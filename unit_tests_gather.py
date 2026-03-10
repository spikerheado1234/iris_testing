import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import _symmetric_memory as symmem

from layers.all_to_all_gather import gather_a2a


def _as_cuda_i64_ptr_tensor(ptrs, device):
    return torch.tensor(list(ptrs), dtype=torch.int64, device=device)


def _alloc_symm_gather_buffers(
    world_size: int,
    e_local: int,
    total_rows: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
    send_counts: torch.Tensor,
    dst_offsets: torch.Tensor,
    tokens: torch.Tensor,
    device: torch.device,
):
    cnts = symmem.empty((world_size, e_local), dtype=torch.int64, device=device)
    cnts_hdl = symmem.rendezvous(cnts, group=dist.group.WORLD)
    cnts_bases = _as_cuda_i64_ptr_tensor(cnts_hdl.buffer_ptrs, device)

    offsets = symmem.empty((world_size,), dtype=torch.int64, device=device)
    offsets_hdl = symmem.rendezvous(offsets, group=dist.group.WORLD)
    offsets_bases = _as_cuda_i64_ptr_tensor(offsets_hdl.buffer_ptrs, device)

    symm_tokens = symmem.empty((total_rows, hidden_dim), dtype=token_dtype, device=device)
    tokens_hdl = symmem.rendezvous(symm_tokens, group=dist.group.WORLD)
    tokens_bases = _as_cuda_i64_ptr_tensor(tokens_hdl.buffer_ptrs, device)

    cnts.copy_(send_counts.to(torch.int64))
    offsets.copy_(dst_offsets.to(torch.int64))
    symm_tokens.copy_(tokens)

    return {
        "cnts": cnts,
        "offsets": offsets,
        "tokens": symm_tokens,
        "cnts_bases": cnts_bases,
        "offsets_bases": offsets_bases,
        "tokens_bases": tokens_bases,
    }


def _init_dist_tcp(rank: int, world_size: int) -> None:
    cache_dir = f"/tmp/triton_cache_{os.getuid()}_{os.getpid()}_rank{rank}"
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )


def _require_dist():
    assert dist.is_initialized()
    assert dist.get_world_size() > 1
    assert torch.cuda.is_available()


def test_gather_a2a(e_local: int = 2, hidden_dim: int = 128, cap: int = 32, threshold: float = 1e-2):
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda")

    # deterministic metadata: every src sends cap rows to every (dst, local expert)
    send_counts = torch.full((world_size, e_local), cap, device=device, dtype=torch.int32)
    dst_offsets = (torch.arange(world_size, device=device, dtype=torch.int32) * (e_local * cap)).contiguous()

    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device=device, dtype=torch.bfloat16)

    # encode source/dst/expert/token into value for exact checking
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    symm = _alloc_symm_gather_buffers(
        world_size=world_size,
        e_local=e_local,
        total_rows=total_rows,
        hidden_dim=hidden_dim,
        token_dtype=tokens.dtype,
        send_counts=send_counts,
        dst_offsets=dst_offsets,
        tokens=tokens,
        device=device,
    )

    dist.barrier()

    gathered = gather_a2a(
        symm["cnts"],
        symm["offsets"],
        symm["tokens"],
        symm["cnts_bases"],
        symm["offsets_bases"],
        symm["tokens_bases"],
        e_local,
    )

    torch.cuda.synchronize()
    dist.barrier()

    # expected shape
    expected_rows = world_size * e_local * cap
    assert gathered.shape == (expected_rows, hidden_dim), (
        f"bad shape: got {gathered.shape}, expected {(expected_rows, hidden_dim)}"
    )

    # build expected packed output:
    # for each src, expert block is contiguous, preserving token order
    expected = torch.empty_like(gathered)
    out_row = 0
    for e in range(e_local):
        for src in range(world_size):
            for t in range(cap):
                val = float(src * 1_000_000 + rank * 10_000 + e * 100 + t)
                expected[out_row].fill_(val)
                out_row += 1

    assert torch.allclose(gathered.float(), expected.float(), rtol=threshold, atol=threshold), (
        f"gather mismatch on rank {rank}"
    )

    if rank == 0:
        print(f"gather_a2a test PASSED (e_local={e_local}, H={hidden_dim}, cap={cap})")


def _worker_gather(local_rank: int, world_size: int) -> None:
    _init_dist_tcp(local_rank, world_size)
    print(f"[rank{local_rank}] init ok, cuda={torch.cuda.current_device()}", flush=True)
    try:
        test_gather_a2a(e_local=2, hidden_dim=128, cap=32)
        test_gather_a2a(e_local=4, hidden_dim=256, cap=16)
        if local_rank == 0:
            print("All gather_a2a tests finished")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ws = torch.cuda.device_count()
    assert ws >= 2, "Need at least 2 GPUs"
    mp.spawn(_worker_gather, args=(ws,), nprocs=ws, join=True)