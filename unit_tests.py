import os
import time
import argparse
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._symmetric_memory as symmem

from layers.all_to_all import custom_a2a
from layers.token_shuffle import shuffle
from layers.expert import expert
from utils import alloc_shmem_buffers, set_seed


# =========================================================
# Part 0. Common helpers
# =========================================================

def _require_cuda():
    assert torch.cuda.is_available(), "CUDA/HIP device is required."


def _device():
    return torch.device("cuda")


def _dtype():
    # keep consistent with your main benchmark
    return torch.bfloat16


def _rand_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_dist_tcp(rank: int, world_size: int, master_port: str = "29500") -> None:
    # Per-process Triton cache, avoids multi-proc cache races
    cache_dir = f"/tmp/triton_cache_{os.getuid()}_{os.getpid()}_rank{rank}"
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
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


def _destroy_dist():
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _spin_wait_counts(buffers, world_size, timeout_s=10.0):
    t0 = time.time()
    while True:
        v = int(buffers.counts_ready.item())
        if v >= world_size:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting counts_ready: {v} < {world_size}")
        time.sleep(0.001)


def _spin_wait_token_sync(buffers, cap: int, world_size: int, tile_size: int = 64, timeout_s: float = 10.0):
    """
    Robust wait helper that tolerates different token_sync layouts.

    Old semantics:
      - token_sync is 1D or 2D and may accumulate to world_size

    New semantics:
      - token_sync is 3D [E_local, world_size, MAX_TILES]
      - each used tile is expected to become 1
    """
    t0 = time.time()
    need_tiles = max((cap + tile_size - 1) // tile_size, 1)

    while True:
        v = buffers.token_sync.detach().cpu()

        ok = False
        if v.dim() == 1:
            ok = bool((v.to(torch.int32) >= world_size).all())
        elif v.dim() == 2:
            # conservative fallback
            ok = bool((v.to(torch.int32) >= 1).all())
        elif v.dim() == 3:
            used = v[:, :, :need_tiles].to(torch.int32)
            ok = bool((used >= 1).all())
        else:
            raise RuntimeError(f"Unexpected token_sync dim={v.dim()} shape={tuple(v.shape)}")

        if ok:
            return

        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting token_sync, shape={tuple(v.shape)}")
        time.sleep(0.001)


def _require_dist():
    assert dist.is_initialized(), "torch.distributed must be initialized."
    assert dist.get_world_size() > 1, "Need WORLD_SIZE > 1."
    assert torch.cuda.is_available(), "CUDA/HIP required."


def _all_gather_tensors(x: torch.Tensor):
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    return gathered


def _make_expert_offsets(local_expert_cnt: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros_like(local_expert_cnt, dtype=torch.int32, device=local_expert_cnt.device)
    if local_expert_cnt.numel() > 1:
        offsets[1:] = local_expert_cnt.cumsum(dim=0)[:-1].to(torch.int32)
    return offsets


def ref_expert(tokens: torch.Tensor, weights: torch.Tensor, local_expert_cnt: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.empty((0, weights.size(-1)), dtype=torch.float32, device="cpu")

    tokens_cpu = tokens.float().cpu()
    weights_cpu = weights.float().cpu()
    cnt_cpu = local_expert_cnt.cpu()

    out_chunks = []
    offset = 0
    for i in range(weights_cpu.size(0)):
        cnt = int(cnt_cpu[i].item())
        if cnt > 0:
            out_chunks.append(torch.matmul(tokens_cpu[offset:offset + cnt], weights_cpu[i]))
            offset += cnt

    if len(out_chunks) == 0:
        return torch.empty((0, weights.size(-1)), dtype=torch.float32, device="cpu")

    return torch.cat(out_chunks, dim=0).float()


# =========================================================
# Part 1. Single-GPU GEMM correctness tests
# =========================================================

def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim, counts=None, seed=42):
    _require_cuda()
    _rand_seed(seed)

    device = _device()
    dtype = _dtype()

    if counts is None:
        expert_token_cnt = torch.randint(
            low=0, high=33,
            size=(num_local_experts,),
            device=device,
            dtype=torch.int32,
        )
    else:
        expert_token_cnt = torch.tensor(counts, device=device, dtype=torch.int32)
        assert expert_token_cnt.numel() == num_local_experts

    total_tokens = int(expert_token_cnt.sum().item())
    tokens = torch.randn((total_tokens, token_hid_dim), device=device, dtype=dtype)
    weights = torch.randn((num_local_experts, token_hid_dim, expert_hid_dim), device=device, dtype=dtype)
    return tokens, weights, expert_token_cnt


def run_one_gemm_case(num_local_experts, token_hid_dim, expert_hid_dim, counts=None, seed=42, atol=2e-2, rtol=2e-2):
    tokens, weights, local_expert_cnt = gen_gemm_input(
        num_local_experts=num_local_experts,
        token_hid_dim=token_hid_dim,
        expert_hid_dim=expert_hid_dim,
        counts=counts,
        seed=seed,
    )

    expert_offsets = _make_expert_offsets(local_expert_cnt)
    assert int(local_expert_cnt.sum().item()) == tokens.size(0)
    for i in range(local_expert_cnt.numel()):
        cnt = int(local_expert_cnt[i].item())
        off = int(expert_offsets[i].item())
        assert off >= 0
        assert cnt >= 0
        assert off + cnt <= tokens.size(0)

    out = expert(tokens, weights, local_expert_cnt, num_local_experts)
    torch.cuda.synchronize()

    ref = ref_expert(tokens, weights, local_expert_cnt).float()
    out_cpu = out.float().cpu()

    assert out_cpu.shape == ref.shape, f"shape mismatch: {tuple(out_cpu.shape)} vs {tuple(ref.shape)}"

    # debug print must be AFTER out_cpu/ref are defined
    offset = 0
    for i in range(local_expert_cnt.numel()):
        cnt = int(local_expert_cnt[i].item())
        if cnt > 0:
            print(f"expert={i}, cnt={cnt}")
            print("custom:")
            print(out_cpu[offset:offset+cnt, :4])
            print("ref:")
            print(ref[offset:offset+cnt, :4])
            offset += cnt

    max_diff = (out_cpu - ref).abs().max().item() if out_cpu.numel() > 0 else 0.0
    print(
        f"[GEMM] E={num_local_experts} H={token_hid_dim} N={expert_hid_dim} "
        f"counts={local_expert_cnt.tolist()} max_diff={max_diff:.6f}",
        flush=True,
    )
    assert torch.allclose(out_cpu, ref, atol=atol, rtol=rtol), f"GEMM mismatch, max_diff={max_diff}"


def test_gemm_small_suite():
    print("=== Part 1: GEMM small correctness suite ===", flush=True)
    # deterministic edge cases
    run_one_gemm_case(2, 64, 64, counts=[0, 0], seed=1)
    run_one_gemm_case(2, 64, 64, counts=[1, 0], seed=2)
    run_one_gemm_case(2, 64, 64, counts=[1, 3], seed=3)
    run_one_gemm_case(4, 128, 64, counts=[0, 1, 2, 7], seed=4)
    run_one_gemm_case(4, 128, 128, counts=[5, 0, 9, 1], seed=5)
    run_one_gemm_case(4, 256, 128, counts=[17, 3, 0, 11], seed=6)


# =========================================================
# Part 2. Single-GPU GEMM stress tests
# =========================================================

def test_gemm_random_stress(num_cases=20):
    print("=== Part 2: GEMM random stress suite ===", flush=True)
    _require_cuda()
    cases = [
        (2, 64, 64),
        (4, 128, 64),
        (4, 128, 128),
        (4, 256, 128),
        (8, 256, 256),
    ]

    for i in range(num_cases):
        E, H, N = cases[i % len(cases)]
        run_one_gemm_case(E, H, N, counts=None, seed=100 + i)


# =========================================================
# Part 3. Multi-GPU custom_a2a correctness
# =========================================================

def test_custom_a2a(world_size: int, e_local: int = 2, hidden_dim: int = 128, cap: int = 32):
    _require_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    experts = e_local * world_size

    # Every sender sends exactly `cap` rows to every (dst, expert) bucket
    send_counts = torch.full((world_size, e_local), cap, device="cuda", dtype=torch.int32)

    # Sender-side packed layout:
    # dst_offsets[dst] = starting row for all experts going to dst
    dst_offsets = (torch.arange(world_size, device="cuda", dtype=torch.int32) * (e_local * cap)).contiguous()

    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)

    # Fill each row with rank/src-dst-expert-token identifiable values
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    buffers = alloc_shmem_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=torch.bfloat16,
    )

    buffers.pca.zero_()
    buffers.counts_ready.zero_()
    buffers.token_sync.zero_()
    buffers.tile_counter.zero_()
    buffers.token_buf.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    out = custom_a2a(
        tokens,
        send_counts,
        dst_offsets,
        buffers.pca,
        buffers.token_buf,
        buffers.counts_ready,
        buffers.token_sync,
        buffers.tile_counter,
        buffers.pca_bases,
        buffers.counts_ready_bases,
        buffers.token_buf_bases,
        buffers.token_sync_bases,
        experts,
        cap,
    )
    torch.cuda.synchronize()

    _spin_wait_counts(buffers, world_size)
    _spin_wait_token_sync(buffers, cap=cap, world_size=world_size)
    torch.cuda.synchronize()
    dist.barrier()

    # Gather sender inputs to build exact expected mapping
    gathered_in = _all_gather_tensors(tokens)

    # Check pca
    exp_pca = torch.full((e_local, world_size), cap, device="cuda", dtype=torch.int32)
    assert torch.equal(buffers.pca, exp_pca), f"[rank{rank}] pca mismatch"

    # Strong block placement check:
    # out[e, src, :, :] must equal the block that src sent to dst=rank for expert e
    dst_base = int(dst_offsets[rank].item())
    for src in range(world_size):
        src_tokens = gathered_in[src]
        for e in range(e_local):
            exp = src_tokens[dst_base + e * cap : dst_base + (e + 1) * cap, :]
            got = out[e, src, :, :]
            assert torch.equal(got, exp), f"[rank{rank}] block mismatch src={src} e={e}"

    if rank == 0:
        print(f"[A2A] PASS e_local={e_local} hidden_dim={hidden_dim} cap={cap}", flush=True)


# =========================================================
# Part 4. Multi-GPU end-to-end smoke:
# custom_a2a -> shuffle -> expert
# =========================================================

def test_end_to_end(world_size: int, e_local: int = 2, hidden_dim: int = 128, out_dim: int = 64, cap: int = 16):
    _require_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    experts = e_local * world_size

    send_counts = torch.full((world_size, e_local), cap, device="cuda", dtype=torch.int32)
    dst_offsets = (torch.arange(world_size, device="cuda", dtype=torch.int32) * (e_local * cap)).contiguous()

    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    # fixed weights, same across ranks
    _rand_seed(1234)
    weights = torch.randn((e_local, hidden_dim, out_dim), device="cuda", dtype=torch.bfloat16)

    buffers = alloc_shmem_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=torch.bfloat16,
    )

    buffers.pca.zero_()
    buffers.counts_ready.zero_()
    buffers.token_sync.zero_()
    buffers.tile_counter.zero_()
    buffers.token_buf.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    out = custom_a2a(
        tokens,
        send_counts,
        dst_offsets,
        buffers.pca,
        buffers.token_buf,
        buffers.counts_ready,
        buffers.token_sync,
        buffers.tile_counter,
        buffers.pca_bases,
        buffers.counts_ready_bases,
        buffers.token_buf_bases,
        buffers.token_sync_bases,
        experts,
        cap,
    )
    torch.cuda.synchronize()

    _spin_wait_counts(buffers, world_size)
    _spin_wait_token_sync(buffers, cap=cap, world_size=world_size)
    packed_tokens = shuffle(buffers.token_buf, buffers.pca, buffers.token_sync, cap)
    torch.cuda.synchronize()

    local_expert_cnt = buffers.pca.sum(dim=1).to(torch.int32)
    out_custom = expert(packed_tokens, weights, local_expert_cnt, experts)
    torch.cuda.synchronize()

    # Build exact packed reference order: pca.view(-1) row-major => expert-major then src-major
    gathered_in = _all_gather_tensors(tokens)
    dst_base = int(dst_offsets[rank].item())

    ref_chunks = []
    for e in range(e_local):
        for src in range(world_size):
            src_tokens = gathered_in[src]
            chunk = src_tokens[dst_base + e * cap : dst_base + (e + 1) * cap, :]
            ref_chunks.append(chunk)

    ref_packed = torch.cat(ref_chunks, dim=0)
    assert torch.equal(packed_tokens, ref_packed), f"[rank{rank}] packed_tokens mismatch"

    ref_out = ref_expert(ref_packed, weights, local_expert_cnt).float()
    out_custom_cpu = out_custom.float().cpu()

    max_diff = (out_custom_cpu - ref_out).abs().max().item() if out_custom_cpu.numel() > 0 else 0.0

    assert out_custom_cpu.shape == ref_out.shape
    assert torch.allclose(out_custom_cpu, ref_out, atol=2e-2, rtol=2e-2), \
        f"[rank{rank}] e2e output mismatch max_diff={max_diff}"

    if rank == 0:
        print(
            f"[E2E] PASS e_local={e_local} hidden_dim={hidden_dim} out_dim={out_dim} cap={cap} "
            f"max_diff={max_diff:.6f}",
            flush=True,
        )


# =========================================================
# Dist worker
# =========================================================

def _worker(rank: int, world_size: int, mode: str, master_port: str):
    _init_dist_tcp(rank, world_size, master_port=master_port)

    try:
        symmem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        dist.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            print(f"[DIST] mode={mode} world_size={world_size}", flush=True)

        if mode in ("a2a", "all"):
            test_custom_a2a(world_size=world_size, e_local=2, hidden_dim=128, cap=32)

        if mode in ("e2e", "all"):
            test_end_to_end(world_size=world_size, e_local=2, hidden_dim=128, out_dim=64, cap=16)

        if rank == 0:
            print("[DIST] PASS", flush=True)

    finally:
        _destroy_dist()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["gemm", "a2a", "e2e", "all"],
        help="gemm = single-GPU only; a2a/e2e/all need multi-GPU",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-port", type=str, default="29500")
    args = parser.parse_args()

    print("RUNNING unit_tests from:", __file__, flush=True)

    if args.mode == "gemm":
        _require_cuda()
        torch.cuda.set_device(0)
        test_gemm_small_suite()
        test_gemm_random_stress(num_cases=20)
        print("[GEMM] PASS", flush=True)
        return

    # multi-GPU modes
    _require_cuda()
    assert torch.cuda.device_count() >= args.world_size, (
        f"Need at least {args.world_size} GPUs, got {torch.cuda.device_count()}"
    )

    mp.spawn(
        _worker,
        args=(args.world_size, args.mode, args.master_port),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()