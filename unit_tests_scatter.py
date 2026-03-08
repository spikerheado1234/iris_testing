import torch
import torch.distributed as dist

import os
import time

import torch.multiprocessing as mp
from layers.all_to_all_scatter import custom_a2a
# 4debug
from layers.all_to_all_scatter import _LAST_DBG

#from .layers.token_shuffle import shuffle
#from .layers.expert import expert

from utils_scatter import alloc_shmem_buffers, set_seed

# 4 debug only
import time
import torch.distributed._symmetric_memory as symmem
print("RUNNING unit_tests from:", __file__, flush=True)


def _spin_wait_tokens(buffers, world_size, timeout_s=10.0):
    t0 = time.time()
    while True:
        v = buffers.token_sync.detach().cpu().to(torch.int32)
        if bool((v >= world_size).all()):
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting token_sync: {v.tolist()} < {world_size}")
        time.sleep(0.001)

def _spin_wait_counts(buffers, world_size, timeout_s=10.0):
    t0 = time.time()
    while True:
        # 直接访问 buffers.counts_ready
        v = int(buffers.counts_ready.item())
        if v >= world_size:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting counts_ready: {v} < {world_size}")
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

    try:
        symmem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        dist.barrier()
        torch.cuda.synchronize()

        print(f"[Rank {local_rank}] Native SymmMem Init OK", flush=True)
        print(f"[Rank {local_rank}] Begin test e_local=2", flush=True)
        test_custom_a2a(world_size=world_size, e_local=2, hidden_dim=128, cap=32)

        if local_rank == 0:
            print("SUCCESS: custom_a2a tests PASSED on devices", flush=True)
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[Rank {local_rank}] destroy_process_group failed: {e}", flush=True)

            
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




def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim):
    expert_token_cnt = torch.randint(low=0, high=100, size=(num_local_experts,))

    tokens = torch.randn(expert_token_cnt.sum(), token_hid_dim)

    weights = torch.randn(num_local_experts, token_hid_dim, expert_hid_dim)

    return tokens, weights



def test_custom_a2a(world_size: int, e_local: int = 2, hidden_dim: int = 128, cap: int = 32, threshold: float = 1e-2) -> bool:
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    experts = e_local * world_size

    dest_counts = torch.full((world_size, e_local), cap, device="cuda", dtype=torch.int32)
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

    buffers = alloc_shmem_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=torch.bfloat16,
    )

    # reset like benchmark
    buffers.pca.zero_()
    buffers.counts_ready.zero_()
    buffers.token_sync.zero_()
    buffers.tile_counter.zero_()
    buffers.token_buf.zero_()
    torch.cuda.synchronize()

    print(f"[rank{rank}] calling custom_a2a e_local={e_local}", flush=True)

    out = custom_a2a(
        tokens, dest_counts, dst_offsets,
        buffers.pca,
        buffers.token_buf,
        buffers.counts_ready,
        buffers.token_sync,
        buffers.tile_counter,
        buffers.pca_bases,
        buffers.counts_ready_bases,
        buffers.token_buf_bases,
        buffers.token_sync_bases,
        experts, cap
    )
    torch.cuda.synchronize()
    _spin_wait_counts(buffers, world_size)
    _spin_wait_tokens(buffers, world_size)
    torch.cuda.synchronize()
    dist.barrier()

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


if __name__ == '__main__':
   
    world_size = 2 
    
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)

    ## Some sample inputs to test out correctness. ##
    #test_gemm(2, 24, 48)
    #test_gemm(5, 128, 128)
    
    #est_custom_a2a(e_local=2, hidden_dim=128, cap=32)
    #test_custom_a2a(e_local=4, hidden_dim=256, cap=16)
    
