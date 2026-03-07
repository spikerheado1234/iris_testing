import torch
import torch.distributed as dist

import os
import time
import iris
import torch.multiprocessing as mp
from layers.all_to_all_scatter import custom_a2a
from layers.all_to_all_gather import gather_a2a
# 4debug
from layers.all_to_all_scatter import _LAST_DBG

#from .layers.token_shuffle import shuffle
#from .layers.expert import expert

from utils import alloc_counts_buffers, alloc_token_buffers, _build_dst_offsets
from baseline import exchange_counts_a2a, exchange_payload_a2a, init_baseline_buffers

# 4 debug only
import time
def _spin_wait_counts(cb, world_size, timeout_s=10.0):
    t0 = time.time()
    while True:
        v = int(cb.counts_ready.item())
        if v >= world_size:
            return
        if time.time() - t0 > timeout_s:
           # print what has not written
            pca = cb.pca.detach().cpu().to(torch.int32)  # [E, world]
            col_sums = pca.sum(dim=0).tolist()
            raise RuntimeError(
                f"timeout waiting counts_ready: {v} < {world_size}, "
                f"pca_col_sums={col_sums}"
            )
        time.sleep(0.001)

def _spin_wait_tokens(tb, world_size, dbg=None, timeout_s=10.0):
    t0 = time.time()
    while True:
        ok = bool(torch.all(tb.token_sync == world_size).item())
        if ok:
            return
        if time.time() - t0 > timeout_s:
            msg = f"timeout token_sync: {tb.token_sync.tolist()}"
            if dbg is not None:
                stage, hb, last = dbg
                msg += (
                    f"\nstage=\n{stage.detach().cpu()}\n"
                    f"hb=\n{hb.detach().cpu()}\n"
                    f"last=\n{last.detach().cpu()}\n"
                )
            if _LAST_DBG:
                print("stage:\n", _LAST_DBG["stage"].cpu())
                print("hb:\n",    _LAST_DBG["hb"].cpu())
                print("last:\n",  _LAST_DBG["last"].cpu())
            raise RuntimeError(msg)
        time.sleep(0.001)





## Some simple testing utility functions.

# this two important for IPv6 / hostname  / TCPStore connect errors also some subtle triton errors like cache overwrites of ranks
def _init_dist_tcp(rank: int, world_size: int) -> None:
    # 1) Per-process Triton cache (avoid multi-proc cache races)
    cache_dir = f"/tmp/triton_cache_{os.getuid()}_{os.getpid()}_rank{rank}"
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # 2) Local rendezvous (avoid hostname/IPv6 resolution issues)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # 3) Ensure rank->GPU mapping is explicit
    torch.cuda.set_device(rank)

    # 4) Init PG with explicit device_id to avoid "Guessing device ID" warnings/hangs
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

def _worker(local_rank: int, world_size: int) -> None:
    _init_dist_tcp(local_rank, world_size)
    shmem = _get_shmem()  # init once per process
    print(f"[rank{local_rank}] init ok, cuda={torch.cuda.current_device()}", flush=True)
    try:
        # run tests ONLY inside workers
        print(f"[rank{local_rank}] begin test e_local=2", flush=True)
        test_custom_a2a(shmem, e_local=2, hidden_dim=128, cap=32)
        print(f"[rank{local_rank}] done  test e_local=2", flush=True)

        print(f"[rank{local_rank}] begin test e_local=4", flush=True)
        test_custom_a2a(shmem, e_local=4, hidden_dim=256, cap=16)
        print(f"[rank{local_rank}] done  test e_local=4", flush=True)

        if local_rank == 0:
            print("custom_a2a tests PASSED")
    finally:
        dist.destroy_process_group()

def is_correct(one, two, threshold):
    if one.shape != two.shape:
        return abs(one.sum() - two.sum()) < threshold

    return torch.allclose(one, two, rtol=threshold)

def _require_dist():
    assert dist.is_initialized(), (
        "torch.distributed is not initialized. "
        "Run with torchrun/srun so process group is initialized before running unit tests."
    )
    assert dist.get_world_size() > 1, "Need WORLD_SIZE > 1 for custom_a2a unit test."
    assert torch.cuda.is_available(), "CUDA is required for this test."


def _get_shmem():
    heap_size = int(os.environ.get("IRIS_HEAP_SIZE", str(1 << 30)))
    return iris.iris(heap_size)

def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim):
    expert_token_cnt = torch.randint(low=0, high=100, size=(num_local_experts,))

    tokens = torch.randn(expert_token_cnt.sum(), token_hid_dim)

    weights = torch.randn(num_local_experts, token_hid_dim, expert_hid_dim)

    return tokens, weights



def test_custom_a2a(shmem, e_local: int = 2, hidden_dim: int = 128, cap: int = 32, threshold: float = 1e-2) -> bool:
    """
      Routing pattern:
      Each src sends exactly `cap` rows to every (dst, local_expert).
      This makes expected placement deterministic and easy to check.
    """
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    experts = e_local * world_size

    # deterministic routing metadata
    dest_counts = torch.full((world_size, e_local), cap, device="cuda", dtype=torch.int32)
    dst_offsets = (torch.arange(world_size, device="cuda", dtype=torch.int32) * (e_local * cap)).contiguous()

    # build unique-valued tokens so misplacement is detectable 
    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    # symmetric buffers
    #shmem = _get_shmem()
    cb = alloc_counts_buffers(shmem, world_size=world_size, e_local=e_local)
    tb = alloc_token_buffers(
        shmem,
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=tokens.dtype,
    )

    cb.pca.zero_()
    cb.counts_ready.zero_()
    tb.token_buf.zero_()
    tb.token_sync.zero_()
    #tb.tile_counter.zero_() 
    tile_counter = torch.zeros((e_local, world_size), device="cuda", dtype=torch.int32)
    print(f"[rank{rank}] calling custom_a2a e_local={e_local}", flush=True)
    # the layerop
    # NEW fix with timing stuff
    dummy_recv_wait = torch.zeros((e_local,), dtype=torch.int64, device="cuda")
    debug_time = torch.zeros((world_size, e_local, 3), dtype=torch.int64, device="cuda")
    
    out = custom_a2a(
        tokens,
        dest_counts,
        dst_offsets,
        cb.pca,
        tb.token_buf,
        cb.counts_ready,
        tb.token_sync,
        #tb.tile_counter,
        tile_counter,
        debug_time,
        dummy_recv_wait,
        cb.heap_bases,
        experts,
        cap,
    )
    print(f"[rank{rank}] returned custom_a2a e_local={e_local}", flush=True)

    # wait via sync vars without barrier or sychornization
    #while int(cb.counts_ready.item()) < world_size:
    #    pass
    #while not bool(torch.all(tb.token_sync == world_size).item()):
    #    pass
    _spin_wait_counts(cb, world_size)
    _spin_wait_tokens(tb, world_size)

    
    # gather inputs for expected mapping 
    gathered_in = [torch.empty_like(tokens) for _ in range(world_size)]
    dist.all_gather(gathered_in, tokens)

    # quick smoke: sums
    #total_in_sum = torch.stack([x.sum() for x in gathered_in]).sum()
    #total_out_sum = out.sum()
    #assert is_correct(total_out_sum, total_in_sum, threshold), "SUM sanity check failed"
    dst_base = int(dst_offsets[rank].item())
    expected_local_sum = 0
    for src in range(world_size):
        expected_local_sum += gathered_in[src][dst_base : dst_base + e_local * cap, :].sum()

    total_out_sum = out.sum()
    assert is_correct(total_out_sum, expected_local_sum, threshold), "SUM sanity check failed"


    #strong check: exact block placement for dst = rank
    dst_base = int(dst_offsets[rank].item())
    for src in range(world_size):
        src_tokens = gathered_in[src]
        for e in range(e_local):
            exp = src_tokens[dst_base + e * cap: dst_base + (e + 1) * cap, :]
            got = out[e, src, :, :]
            assert torch.equal(got, exp), f"block mismatch dst={rank} src={src} e={e}"

    return True

   

def test_gemm(total_expert_cnt, token_hid_dim, expert_hid_dim):
    
    tokens, weights = gen_gemm_input(total_expert_cnt, token_hid_dim, expert_hid_dim)

    custom_output = expert(tokens, weights,expert_token_cnt,total_expert_cnt)

    ## We use pytorch as ground truth. ##
    torch_out = []
    tokens_seen = 0
    for i in range(total_expert_cnt):
        torch_out.append(torch.einsum('sd,df->sf', tokens[tokens_seen:expert_token_cnt[i], :], weights[i]))
        tokens_seen += expert_token_cnt[i]

    return is_correct(custom_output, torch.stack(torch_out), 1e-2)


def test_gather_a2a(shmem, e_local: int = 2, hidden_dim: int = 128, cap: int = 32, threshold: float = 1e-2) -> bool:
    """
    Test the gather (pull) all-to-all against the PyTorch baseline.

    Each rank builds a deterministic send payload where every token row
    is filled with a unique value encoding (rank, dst, expert, token_id).
    We run both the custom gather a2a and the baseline a2a, then compare
    the gathered output tensors.
    """
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # --- Build deterministic routing: each rank sends `cap` tokens to every (dst, expert). ---
    send_counts = torch.full((world_size, e_local), cap, device=device, dtype=torch.int32)
    dst_offsets = _build_dst_offsets(send_counts)

    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device=device, dtype=torch.bfloat16)
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    # --- Allocate symmetric memory buffers for the gather a2a. ---
    # cnts and offsets must live in symmetric memory so other ranks can read them.
    shmem_cnts = shmem.zeros((world_size, e_local), dtype=torch.int64, device="cuda")
    shmem_offsets = shmem.zeros((world_size,), dtype=torch.int64, device="cuda")
    shmem_tokens = shmem.zeros((total_rows, hidden_dim), dtype=tokens.dtype, device="cuda")
    heap_bases = shmem.get_heap_bases()

    # Populate symmetric buffers with this rank's data.
    shmem_cnts.copy_(send_counts.to(torch.int64))
    shmem_offsets.copy_(dst_offsets.to(torch.int64))
    shmem_tokens.copy_(tokens)

    # Ensure all ranks have written their symmetric buffers before pulling.
    shmem.barrier()

    # --- Run custom gather a2a. ---
    gathered = gather_a2a(shmem_cnts, shmem_offsets, heap_bases, e_local, shmem_tokens)
    torch.cuda.synchronize()
    shmem.barrier()

    # --- Run baseline a2a to get expected result. ---
    # exchange_counts_a2a gives us recv_counts[src, e] = tokens src sends to our expert e.
    total_recv_upper = world_size * e_local * cap
    base_bufs = init_baseline_buffers(
        world_size, e_local, cap, hidden_dim, tokens.dtype, device, total_recv_upper,
    )
    recv_counts = exchange_counts_a2a(send_counts, base_bufs, strict_capacity=False, capacity=cap)
    recv_flat, _, _ = exchange_payload_a2a(tokens, send_counts, recv_counts, base_bufs)
    torch.cuda.synchronize()

    # recv_flat is packed [src0_e0, src0_e1, ..., src1_e0, ...] which is expert-minor within each src.
    # gathered is packed expert-major: [e0_src0, e0_src1, ..., e1_src0, ...] (from write_meta layout).
    # Reorder baseline into the same expert-major layout for comparison.
    expected = torch.zeros_like(gathered)
    baseline_off = 0
    # Build the same write_meta offsets the kernel uses.
    # expert-major, device-minor: all tokens for expert 0 first, then expert 1, etc.
    write_off = {}
    running = 0
    for e in range(e_local):
        for src in range(world_size):
            c = int(recv_counts[src, e].item())
            write_off[(src, e)] = running
            running += c

    for src in range(world_size):
        for e in range(e_local):
            c = int(recv_counts[src, e].item())
            if c > 0:
                wo = write_off[(src, e)]
                expected[wo:wo + c] = recv_flat[baseline_off:baseline_off + c]
            baseline_off += c

    # --- Compare. ---
    if gathered.shape != expected.shape:
        print(f"[rank{rank}] FAIL: shape mismatch gathered={gathered.shape} expected={expected.shape}")
        return False

    if not torch.allclose(gathered.float(), expected.float(), rtol=threshold, atol=threshold):
        max_diff = (gathered.float() - expected.float()).abs().max().item()
        print(f"[rank{rank}] FAIL: max_diff={max_diff}")
        return False

    if rank == 0:
        print(f"[rank{rank}] gather_a2a test PASSED (e_local={e_local}, H={hidden_dim}, cap={cap})")
    return True


def _worker_gather(local_rank: int, world_size: int) -> None:
    _init_dist_tcp(local_rank, world_size)
    shmem = _get_shmem()
    print(f"[rank{local_rank}] init ok, cuda={torch.cuda.current_device()}", flush=True)
    try:
        test_gather_a2a(shmem, e_local=16, hidden_dim=128, cap=32)
        test_gather_a2a(shmem, e_local=16, hidden_dim=256, cap=16)
        if local_rank == 0:
            print("All gather_a2a tests finished")
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
   
    world_size = 2

    # Scatter (push) a2a tests.
    # mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)

    # Gather (pull) a2a tests.
    mp.spawn(_worker_gather, args=(world_size,), nprocs=world_size, join=True)
    