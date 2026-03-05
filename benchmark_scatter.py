import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris

import time
from baseline import run_baseline_ref, init_baseline_buffers

from utils import set_seed, gen_local_tokens, gen_router, route_and_pack_padding_free, alloc_shmem_buffers, nvtx_push, nvtx_pop, _allreduce_max_i32, _allreduce_min_i32, _sync_and_check, _build_dst_offsets, _masked_stats

from  layers.all_to_all import custom_a2a, _LAST_DBG

#import torch.cuda.profiler as cuda_profiler
import torch.profiler

WARMUP = int(os.getenv("WARMUP", "5"))
ITERS  = int(os.getenv("ITERS", "20"))
CHECK  = int(os.getenv("CHECK", "1"))  # 1: correctness；0:  perf/sweep
CLEAR_TOKEN_BUF  = int(os.getenv("CLEAR_TOKEN_BUF", "0"))
SKIP_ON_OVERFLOW = int(os.getenv("SKIP_ON_OVERFLOW", "1"))
IRIS_HEAP_GIB    = int(os.getenv("IRIS_HEAP_GIB", "100"))

PROFILE_CUSTOM   = int(os.getenv("PROFILE_CUSTOM", "0")) == 1
PROFILE_BASELINE = int(os.getenv("PROFILE_BASELINE", "0")) == 1
PROFILE_ITERS    = int(os.getenv("PROFILE_ITERS", "3"))
TRACE_DIR        = os.getenv("TRACE_DIR", ".")

BLOCK_E = int(os.getenv("BLOCK_E", "128"))
COUNTS_WARPS = int(os.getenv("COUNTS_WARPS", "4"))
BLOCK_K = int(os.getenv("BLOCK_K", "128"))
TOKENS_WARPS = int(os.getenv("TOKENS_WARPS", "4"))

TEST_MODE = os.getenv("TEST_MODE", "free") # 'free' or 'fixed' for blocks


def _init_iris_shmem():
    heap_size = IRIS_HEAP_GIB * (2**30)
    return iris.iris(heap_size)



# Profile pass (separate from perf timing) so the timing loop is good

def _profile_pass_custom(
    rank: int,
    shmem,
    buffers,
    send_payload,
    send_counts,
    dst_offsets,
    num_experts_total: int,
    capacity: int,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_custom_rank{rank}.json")

    prof = None
    if do_trace:
        os.makedirs(TRACE_DIR, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()

    try:
        for i in range(PROFILE_ITERS):
            if do_trace:
                ctx = torch.profiler.record_function(f"custom_prof_iter_{i}")
                ctx.__enter__()
            

            torch.cuda.synchronize()
            _ = custom_a2a(
                send_payload,
                send_counts,
                dst_offsets,
                buffers.pca,
                buffers.token_buf,
                buffers.counts_ready,
                buffers.token_sync,
                buffers.tile_counter,
                buffers.heap_bases,
                num_experts_total,
                capacity,
                BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
            )
            torch.cuda.synchronize()
            shmem.barrier()
 

            if do_trace:
                ctx.__exit__(None, None, None)
                prof.step()

    finally:
        if do_trace and prof is not None:
            prof.stop()
            prof.export_chrome_trace(trace_path)
            print(f"[trace] wrote {trace_path}", flush=True)


def _profile_pass_baseline(
    rank: int,
    shmem,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    send_payload,
    send_counts,
    base_buffers_perf,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_baseline_rank{rank}.json")

    prof = None
    if do_trace:
        os.makedirs(TRACE_DIR, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()

    try:
        for i in range(PROFILE_ITERS):
            if do_trace:
                ctx = torch.profiler.record_function(f"baseline_prof_iter_{i}")
                ctx.__enter__()

            torch.cuda.synchronize()
            _ = run_baseline_ref(
                rank=rank,
                world_size=world_size,
                e_local=e_local,
                capacity=capacity,
                hidden_dim=hidden_dim,
                send_payload=send_payload,
                send_counts=send_counts,
                buffers=base_buffers_perf,
                do_reorder=False,       # COMM-only
                profile=False,
                strict_capacity=False,
                barrier=False,
            )
            torch.cuda.synchronize()
            shmem.barrier()

            if do_trace:
                ctx.__exit__(None, None, None)
                prof.step()

    finally:
        if do_trace and prof is not None:
            prof.stop()
            prof.export_chrome_trace(trace_path)
            print(f"[trace] wrote {trace_path}", flush=True)



# brand new check compare with comm calc only and better structure
def check_compare(
    rank: int,
    world_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    topk: int,
    e_local: int,
    capacity: int,
    seed: int,
):
    # init dist 
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    set_seed(seed, rank)

    # generate input tokens + routing
    num_experts_total = world_size * e_local
    tokens = gen_local_tokens(batch, seq, hidden_dim, torch.bfloat16, device, seed, rank)
    router = gen_router(hidden_dim, num_experts_total, torch.bfloat16, device, seed)

    if TEST_MODE == "fixed":
        from utils import route_and_pack_fixed
        send_payload, send_counts, _, _ = route_and_pack_fixed(
            tokens, router, topk, world_size, num_experts_total, capacity
        )
    else:
        send_payload, send_counts, _, _ = route_and_pack_padding_free(
            tokens, router, topk, world_size, num_experts_total
        )
    
    send_counts = send_counts.to(torch.int32).contiguous()  # [world, E_local]
    dst_offsets = _build_dst_offsets(send_counts)

    #  gather global counts
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    counts_all = torch.stack(all_send_counts, dim=0).contiguous()  # [src, dst, E_local]

    # expected PCA for correctness
    expected_pca = counts_all[:, rank, :].transpose(0, 1).contiguous()  # [E_local, world]

    # overflow check (prevents deadlocks + unfair comparisons) 
    local_max = expected_pca.max().to(torch.int32)
    global_max = _allreduce_max_i32(local_max).item()

    if global_max > capacity and SKIP_ON_OVERFLOW:
        if rank == 0:
            print(f"[SKIP] max_recv_count={global_max} > CAPACITY={capacity}. "
                  f"Increase CAPACITY or reduce (BATCH, SEQ, TOPK), or set SKIP_ON_OVERFLOW=0.")
        dist.destroy_process_group()
        return

    # init shmem buffers (custom)
    shmem = _init_iris_shmem()
    buffers = alloc_shmem_buffers(
        shmem=shmem,
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=torch.bfloat16,
    )
    # fix the extra cost
    buffers.heap_bases = buffers.heap_bases.to(device, non_blocking=True)
    # Optional clarity
    if CLEAR_TOKEN_BUF:
        buffers.token_buf.zero_()

    # init baseline buffers (perf path: NO reorder)
    total_recv = int(counts_all[:, rank, :].sum().item())
    base_buffers_perf = init_baseline_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=send_payload.dtype,
        device=device,
        total_recv=total_recv,
        allocate_token_buf=False,   # perf: do not allocate token_buf / reorder target
    )
    

       # Perf: COMM-only timing
 
    # Warmup custom (not timed)
    for i in range(WARMUP):
        buffers.pca.zero_()
        buffers.counts_ready.zero_()
        buffers.token_sync.zero_()
        buffers.tile_counter.zero_() 
        if CLEAR_TOKEN_BUF:
            buffers.token_buf.zero_()
        
        _ = custom_a2a(
            send_payload,
            send_counts,
            dst_offsets,
            buffers.pca,
            buffers.token_buf,
            buffers.counts_ready,
            buffers.token_sync,
            buffers.tile_counter,
            buffers.heap_bases,
            num_experts_total,
            capacity,
            BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
        )
        shmem.barrier()
        
      
     
    shmem.barrier()          # local alignment
    dist.barrier()           # one-time global alignment (optional but safe)
    torch.cuda.synchronize()


    # Warmup baseline (not timed) — do_reorder=False, strict_capacity=False
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    warmup_err = None
    for i in range(WARMUP):

        try:
            _ = run_baseline_ref(
                rank=rank,
                world_size=world_size,
                e_local=e_local,
                capacity=capacity,
                hidden_dim=hidden_dim,
                send_payload=send_payload,
                send_counts=send_counts,
                buffers=base_buffers_perf,
                do_reorder=False,        # COMM-only
                profile=False,
                strict_capacity=False,   # perf: don't throw
                barrier=False,           # we sync outside consistently
            )
           
      
        except Exception as e:
            ok.zero_()
            warmup_err = repr(e)
            break

    # Unified convergence point: all ranks must hit this once
    ok_global = _sync_and_check(ok)

    if ok_global == 0:
        if rank == 0:
            print(f"[baseline warmup exception] {warmup_err}")
        dist.destroy_process_group()
        return

    # Align start (do once not in per-iter)
   
    dist.barrier()
    torch.cuda.synchronize()

    # cuda evenets
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    custom_times = []

    for i in range(ITERS):
        buffers.pca.zero_()
        buffers.counts_ready.zero_()
        buffers.token_sync.zero_()
        buffers.tile_counter.zero_()
        if CLEAR_TOKEN_BUF:
            buffers.token_buf.zero_()

        torch.cuda.synchronize()

        start_event.record()

        _ = custom_a2a(
            send_payload,
            send_counts,
            dst_offsets,
            buffers.pca,
            buffers.token_buf,
            buffers.counts_ready,
            buffers.token_sync,
            buffers.tile_counter,
            buffers.heap_bases,
            num_experts_total,
            capacity,
            BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
        )

        end_event.record()
        end_event.synchronize()
        custom_times.append(start_event.elapsed_time(end_event))

        shmem.barrier()
    
    # [NEW] Print AMD inline-asm spin lock cycles output if available
    if "spin_cycles" in _LAST_DBG:
        torch.cuda.synchronize()
        cycles = _LAST_DBG["spin_cycles"].cpu()
        valid_cycles = cycles[cycles > 0] # fliter the extra 0
        if len(valid_cycles) > 0:
            avg_c = valid_cycles.float().mean().item()
            max_c = valid_cycles.max().item()
            # Only print on rank 0 for brevity
            if rank == 0:
                print("\n=== AMD Inline-ASM Profiling (Last Iteration) ===")
            print(f"[Rank {rank}] Spin-lock cycles: AVG = {avg_c:.1f}, MAX = {max_c}")

    # Timed baseline — per-iter timing, sync semantics: safe barrier + cuda sync
   # Align start (do once not in the per-iter)
    dist.barrier()
    torch.cuda.synchronize()

    baseline_times = []
    timed_err = None
    ok.fill_(1)

    for i in range(ITERS):
        try:
            torch.cuda.synchronize()
            start_event.record()
            # fair compare
            
            _ = run_baseline_ref(
                rank=rank,
                world_size=world_size,
                e_local=e_local,
                capacity=capacity,
                hidden_dim=hidden_dim,
                send_payload=send_payload,
                send_counts=send_counts,
                buffers=base_buffers_perf,
                do_reorder=False,
                profile=False,
                strict_capacity=False,
                barrier=False,
            )

            end_event.record()
            end_event.synchronize()
            
            baseline_times.append(start_event.elapsed_time(end_event))
            shmem.barrier()

        except Exception as e:
            ok.zero_()
            timed_err = repr(e)
            break
    # global ok check after loop
    ok_global = _allreduce_min_i32(ok).item()
    if ok_global == 0:
        if rank == 0:
            print(f"[baseline timed exception] {timed_err}")
        dist.destroy_process_group()
        return




    custom_ms = torch.tensor([sum(custom_times) / len(custom_times)], device=device, dtype=torch.float32)
    base_ms   = torch.tensor([sum(baseline_times) / len(baseline_times)], device=device, dtype=torch.float32)

    custom_ms_max = custom_ms.clone()
    base_ms_max   = base_ms.clone()
    dist.all_reduce(custom_ms_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(base_ms_max,   op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== COMM-only Perf (local completion + max-rank reduce) ===")
        print(f"world={world_size} E_local={e_local} CAP={capacity} H={hidden_dim} batch={batch} seq={seq} topk={topk}")
        print(f"custom_ms (max-rank mean):   {custom_ms_max.item():.3f}")
        print(f"baseline_ms (max-rank mean): {base_ms_max.item():.3f}")
        print(f"speedup (baseline/custom):   {base_ms_max.item()/custom_ms_max.item():.3f}x")



# Separate profile passes (NOT included in perf timing)
   
    dist.barrier()
    torch.cuda.synchronize()

    if PROFILE_CUSTOM:
        _profile_pass_custom(
            rank,
            shmem,
            buffers,
            send_payload,
            send_counts,
            dst_offsets,
            num_experts_total,
            capacity,
        )

    dist.barrier()
    torch.cuda.synchronize()

    if PROFILE_BASELINE:
        _profile_pass_baseline(
            rank,
            shmem,
            world_size,
            e_local,
            capacity,
            hidden_dim,
            send_payload,
            send_counts,
            base_buffers_perf,
        )

    # 
    # Correctness: run once, strict
    # 
    if CHECK:
        dist.barrier()
        torch.cuda.synchronize()
        # If overflow would occur, correctness is undefined unless you implement capacity capping.
        if global_max > capacity:
            if rank == 0:
                print(f"[SKIP correctness] max_recv_count={global_max} > CAPACITY={capacity}.")
            dist.destroy_process_group()
            return

        # Run custom once
        buffers.pca.zero_()
        buffers.counts_ready.zero_()
        buffers.token_sync.zero_()
        buffers.tile_counter.zero_() 
        if CLEAR_TOKEN_BUF:
            buffers.token_buf.zero_()
       
      
        _ = custom_a2a(
            send_payload,
            send_counts,
            dst_offsets,
            buffers.pca,
            buffers.token_buf,
            buffers.counts_ready,
            buffers.token_sync,
            buffers.tile_counter,
            buffers.heap_bases,
            num_experts_total,
            capacity,
            BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
        )
      
        torch.cuda.synchronize()

        # Baseline buffers for correctness (needs token_buf)
        base_buffers_ck = init_baseline_buffers(
            world_size=world_size,
            e_local=e_local,
            capacity=capacity,
            hidden_dim=hidden_dim,
            token_dtype=send_payload.dtype,
            device=device,
            total_recv=total_recv,
            allocate_token_buf=True,
        )

        # Run baseline once with reorder + strict checks
        torch_out, _, _ = run_baseline_ref(
            rank=rank,
            world_size=world_size,
            e_local=e_local,
            capacity=capacity,
            hidden_dim=hidden_dim,
            send_payload=send_payload,
            send_counts=send_counts,
            buffers=base_buffers_ck,
            do_reorder=True,
            profile=False,
            strict_capacity=True,
            barrier=False,
        )
        torch_out = base_buffers_ck.token_buf
        dist.barrier()
        torch.cuda.synchronize()

        pca_diff = (buffers.pca - expected_pca).abs().max().to(torch.float32)
        max_diff, sum_triton, sum_torch = _masked_stats(buffers.token_buf, torch_out, expected_pca)

        # Global reductions
        pca_diff_g = pca_diff.clone()
        max_diff_g = max_diff.clone()
        sum_triton_g = sum_triton.clone()
        sum_torch_g  = sum_torch.clone()

        dist.all_reduce(pca_diff_g, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_diff_g, op=dist.ReduceOp.MAX)
        dist.all_reduce(sum_triton_g, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_torch_g,  op=dist.ReduceOp.SUM)

        if rank == 0:
            print("=== Correctness Report ===")
            print(f"PCA max|diff| (global): {pca_diff_g.item():.6g}")
            print(f"TOKEN_BUF max|diff| valid (global): {max_diff_g.item():.6g}")
            print(f"TOKEN_BUF sum(valid) custom (global): {sum_triton_g.item():.6g}")
            print(f"TOKEN_BUF sum(valid) base   (global): {sum_torch_g.item():.6g}")
            ok_pass = (pca_diff_g.item() == 0.0) and (max_diff_g.item() == 0.0) and (
                abs(sum_triton_g.item() - sum_torch_g.item()) < 1e-3
            )
            print("PASS" if ok_pass else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", "8"))
    batch      = int(os.getenv("BATCH", "4"))
    seq        = int(os.getenv("SEQ", "2048"))
    hidden_dim = int(os.getenv("HIDDEN", "4096"))
    topk       = int(os.getenv("TOPK", "2"))
    e_local    = int(os.getenv("E_LOCAL", "4"))
    capacity   = int(os.getenv("CAPACITY", "8096"))
    seed       = int(os.getenv("SEED", "42")) # random ssed

    # Robust defaults for env:// rendezvous
    if not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        check_compare,
        args=(world_size, batch, seq, hidden_dim, topk, e_local, capacity, seed),
        nprocs=world_size,
        join=True,
    )