import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


import time
from baseline import run_baseline_a2a, run_baseline_padding, init_baseline_buffers

from utils_scatter import set_seed, gen_local_tokens, gen_router, route_and_pack_padding_free, alloc_shmem_buffers, nvtx_push, nvtx_pop, _allreduce_max_i32, _allreduce_min_i32, _sync_and_check, _build_dst_offsets, _masked_stats, reset_custom_iter_state

from  layers.all_to_all_scatter import custom_a2a

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





# Profile pass (separate from perf timing) so the timing loop is good

def _profile_pass_custom(
    rank: int,
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
            

            #torch.cuda.synchronize()
            reset_custom_iter_state(buffers, clear_token_buf=False)
            _ = custom_a2a(
                send_payload,
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
                num_experts_total,
                capacity,
                BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
            )
            #torch.cuda.synchronize()
            dist.barrier()
 

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

            #torch.cuda.synchronize()
            _ = run_baseline_a2a(
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
            #torch.cuda.synchronize()
            dist.barrier()

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
    
    # [NEW] 必须显式为当前进程组开启对称内存支持
    try:
        import torch.distributed._symmetric_memory as symmem
        # 这里的 group_name 默认是 "0" 或者从 group 对象获取
        symmem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    except Exception as e:
        if rank == 0: print(f"开启 SymmMem 组权限失败: {e}")
    
    device = torch.device(f"cuda:{rank}")

    set_seed(seed, rank)

    # generate input tokens + routing
    num_experts_total = world_size * e_local
    tokens = gen_local_tokens(batch, seq, hidden_dim, torch.bfloat16, device, seed, rank)
    router = gen_router(hidden_dim, num_experts_total, torch.bfloat16, device, seed)

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
    shmem = None
    buffers = alloc_shmem_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=torch.bfloat16,
    )
    # fix the extra cost noneed without iris
    #buffers.heap_bases = buffers.heap_bases.to(device, non_blocking=True)
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
        force_max_capacity=False ## padding
    )
    
    # NEW Init Baseline Buffers (Padding/Fair) 
    base_buffers_pad = init_baseline_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=send_payload.dtype,
        device=device,
        total_recv=0,               # Ignored when forcing max
        allocate_token_buf=False,
        force_max_capacity=True     # <--- Padding Mode (Allocates Max Size)
    )

    # Perf: COMM-only timing
 
    # Warmup custom (not timed)
    for i in range(WARMUP):
        buffers.pca.zero_()
        buffers.counts_ready.zero_()
        buffers.token_sync.zero_()
        #buffers.tile_counter.zero_() # ALREADY IN ALL2ALL
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
            
            buffers.pca_bases,
            buffers.counts_ready_bases,
            buffers.token_buf_bases,
            buffers.token_sync_bases,

            num_experts_total,
            capacity,
            BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
        )
        dist.barrier()
        
      
     
    dist.barrier()          # local alignment
    #dist.barrier()           # one-time global alignment (optional but safe)
    #torch.cuda.synchronize()


    # Warmup baseline (not timed) — do_reorder=False, strict_capacity=False
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    warmup_err = None
    for i in range(WARMUP):

        try:
            _ = run_baseline_a2a(
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
            dist.barrier()
      
        except Exception as e:
            ok.zero_()
            warmup_err = repr(e)
            break
    # NEW Warmup Baseline Padding (Fair) 
    for i in range(WARMUP):
        try:
            _ = run_baseline_padding(
                rank=rank,
                world_size=world_size,
                e_local=e_local,
                capacity=capacity,
                hidden_dim=hidden_dim,
                send_counts=send_counts,
                buffers=base_buffers_pad, # Use padding buffers
                profile=False,
                barrier=False,
            )
            dist.barrier()
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
    torch.cuda.synchronize()  # flush once so window is clean

    # Custom timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        reset_custom_iter_state(buffers, clear_token_buf=bool(CLEAR_TOKEN_BUF))
        _ = custom_a2a(
            send_payload,
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
            
            num_experts_total,
            capacity,
            BLOCK_E, COUNTS_WARPS, BLOCK_K, TOKENS_WARPS,
        )
        dist.barrier()  # local completion fence (cheap)
    end_event.record()
    end_event.synchronize()
    dist.barrier()

    custom_total_ms = start_event.elapsed_time(end_event)
    custom_avg_ms_local = custom_total_ms / float(ITERS)

    custom_avg_ms = torch.tensor(custom_avg_ms_local, device=device, dtype=torch.float32)
    dist.all_reduce(custom_avg_ms, op=dist.ReduceOp.MAX)

    # Baseline (window timing) 
    dist.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    ok.fill_(1)
    timed_err = None

    start_event.record()
    for _ in range(ITERS):
        try:
            _ = run_baseline_a2a(
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
                barrier=False,  # IMPORTANT: no per-iter dist.barrier
            )
        except Exception as e:
            ok.zero_()
            timed_err = repr(e)
            break
        dist.barrier()
    end_event.record()
    end_event.synchronize()
    dist.barrier()

    ok_global = _allreduce_min_i32(ok).item()
    if ok_global == 0:
        if rank == 0:
            print(f"[baseline timed exception] {timed_err}")
        dist.destroy_process_group()
        return

    baseline_pack_total_ms = start_event.elapsed_time(end_event)
    baseline_pack_avg_ms = torch.tensor(baseline_pack_total_ms / float(ITERS), device=device, dtype=torch.float32)
    dist.all_reduce(baseline_pack_avg_ms, op=dist.ReduceOp.MAX)

    # NEW Baseline Padding (Fair) Timing ###
    dist.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    ok.fill_(1)

    start_event.record()
    for _ in range(ITERS):
        try:
            _ = run_baseline_padding(
                rank=rank,
                world_size=world_size,
                e_local=e_local,
                capacity=capacity,
                hidden_dim=hidden_dim,
                send_counts=send_counts,
                buffers=base_buffers_pad, # Use padding buffers
                profile=False,
                barrier=False,
            )
        except Exception as e:
            ok.zero_()
            break
        dist.barrier()
    end_event.record()
    end_event.synchronize()
    dist.barrier()

    if _allreduce_min_i32(ok).item() == 0:
        if rank == 0: print("[baseline padding timed exception]")
        dist.destroy_process_group()
        return

    baseline_pad_total_ms = start_event.elapsed_time(end_event)
    baseline_pad_avg_ms = torch.tensor(baseline_pad_total_ms / float(ITERS), device=device, dtype=torch.float32)
    dist.all_reduce(baseline_pad_avg_ms, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("=== COMM-only Perf Report ===")
        print(f"Config: world={world_size} E_local={e_local} CAP={capacity} H={hidden_dim} batch={batch} seq={seq} topk={topk}")
        
        c_ms = custom_avg_ms.item()
        pak_ms = baseline_pack_avg_ms.item()
        pad_ms = baseline_pad_avg_ms.item()
        
        print(f"1. Custom (Triton P2P):      {c_ms:.3f} ms")
        print(f"2. Baseline (Padding/Fair):  {pad_ms:.3f} ms")
        print(f"3. Baseline (Packing/SOTA):  {pak_ms:.3f} ms")
        print("-" * 40)
        print(f"Fair Speedup (Pad / Custom): {pad_ms / c_ms:.3f}x  <-- Key Metric (Same Algorithm)")
        print(f"SOTA Speedup (Pak / Custom): {pak_ms / c_ms:.3f}x  <-- Algo Gap")



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
        # buffers.tile_counter.zero_() # ALREADY IN ALL2ALL
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

            buffers.pca_bases,
            buffers.counts_ready_bases,
            buffers.token_buf_bases,
            buffers.token_sync_bases,
            
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
        torch_out, _, _ = run_baseline_a2a(
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