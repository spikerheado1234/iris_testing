import os
from typing import Tuple, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from layers.all_to_all import custom_a2a 
from layers.token_shuffle import shuffle
from layers.expert import expert
import time
from baseline import (
    run_baseline_ref,
    init_baseline_buffers,
    exchange_counts_a2a,
    exchange_payload_a2a,
    reorder_flat_to_token_buf,
)

from utils import (
    set_seed,
    gen_local_tokens,
    gen_router,
    route_and_pack_padding_free,
    alloc_shmem_buffers,
    nvtx_push,
    nvtx_pop,
    _allreduce_max_i32,
    _allreduce_min_i32,
    _sync_and_check,
    _build_dst_offsets,
    _masked_stats,
    reset_custom_iter_state,
    wait_token_sync_ready,
    wait_counts_ready,
)



#import torch.cuda.profiler as cuda_profiler
import torch.profiler

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------


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
CONSUME_TOKEN_SYNC = int(os.getenv("CONSUME_TOKEN_SYNC", "0"))
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# new 4 sweep only
def _emit_result_line(
    rank: int,
    *,
    world_size: int,
    e_local: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    topk: int,
    capacity: int,
    seed: int,
    baseline_ms: Optional[float],
    custom_ms: Optional[float],
    status: str,
):
    if rank != 0:
        return

    experts_total = world_size * e_local
    if baseline_ms is not None and custom_ms is not None and custom_ms > 0:
        speedup = baseline_ms / custom_ms
        baseline_s = f"{baseline_ms:.6f}"
        custom_s = f"{custom_ms:.6f}"
        speedup_s = f"{speedup:.6f}"
    else:
        baseline_s = ""
        custom_s = ""
        speedup_s = ""

    print(
        "RESULT,"
        f"{world_size},{experts_total},{e_local},{topk},{batch},{seq},{hidden_dim},{capacity},{seed},"
        f"{baseline_s},{custom_s},{speedup_s},{status}",
        flush=True,
    )



# ---------------------------------------------------------------------------
# Profile helpers (separate from perf timing)
# ----------------------------------------------------------------
def _profile_pass_custom(
    rank: int,
    buffers,
    send_payload,
    send_counts,
    dst_offsets,
    num_experts_total: int,
    capacity: int,
    world_size: int,
    weights: torch.Tensor,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_custom_rank{rank}.json")

    prof = None
    if do_trace:
        os.makedirs(TRACE_DIR, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()

    # reset as main path
    reset_custom_iter_state(buffers, clear_token_buf=False)
    torch.cuda.synchronize()
    dist.barrier()

    try:
        for i in range(PROFILE_ITERS):
            if do_trace:
                ctx = torch.profiler.record_function(f"scatter_custom_prof_iter_{i}")
                ctx.__enter__()

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

            wait_counts_ready(buffers.counts_ready, world_size)
            wait_token_sync_ready(buffers.token_sync, buffers.pca)

            packed_tokens = shuffle(
                buffers.token_buf,
                buffers.pca,
                buffers.token_sync,
                capacity,
            )
            local_expert_cnt = buffers.pca.sum(dim=1).to(torch.int32)
            _ = expert(packed_tokens, weights, local_expert_cnt, num_experts_total)

            torch.cuda.synchronize()
            dist.barrier()

            if do_trace:
                ctx.__exit__(None, None, None)
                prof.step()

            # new reset + barrier after each iter as main path
            if i != PROFILE_ITERS - 1:
                reset_custom_iter_state(buffers, clear_token_buf=False)
                torch.cuda.synchronize()
                dist.barrier()

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
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    base_buffers_perf,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_scatter_baseline_rank{rank}.json")

    prof = None
    if do_trace:
        os.makedirs(TRACE_DIR, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()

    try:
        for i in range(PROFILE_ITERS):
            if do_trace:
                ctx = torch.profiler.record_function(f"scatter_baseline_prof_iter_{i}")
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
            dist.barrier()

            if do_trace:
                ctx.__exit__(None, None, None)
                prof.step()

    finally:
        if do_trace and prof is not None:
            prof.stop()
            prof.export_chrome_trace(trace_path)
            print(f"[trace] wrote {trace_path}", flush=True)


# ---------------------------------------------------------------------------
# Main benchmark entry point (one per rank)
# ---------------------------------------------------------------------------

def bench_scatter(
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
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
    #important put it outside the module wont works, also ignore the second line cause it never works on gh200
    import torch.distributed._symmetric_memory as symmem
    symmem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    

    device = torch.device(f"cuda:{rank}")

    set_seed(seed, rank)

    # ------------------------------------------------------------------
    # 1. Generate input tokens + routing
    # ------------------------------------------------------------------
    num_experts_total = world_size * e_local
    tokens = gen_local_tokens(batch, seq, hidden_dim, torch.bfloat16, device, seed, rank)
    router = gen_router(hidden_dim, num_experts_total, torch.bfloat16, device, seed)

    send_payload, send_counts, _, _ = route_and_pack_padding_free(
        tokens, router, topk, world_size, num_experts_total
    )
    send_counts = send_counts.to(torch.int32).contiguous()  # [world, E_local]
    dst_offsets = _build_dst_offsets(send_counts)
    total_rows  = send_payload.shape[0]

    #  scatter global counts
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    counts_all = torch.stack(all_send_counts, dim=0).contiguous()  # [src, dst, E_local]

    # expected PCA for correctness
    expected_pca = counts_all[:, rank, :].transpose(0, 1).contiguous()  # [E_local, world]

    # overflow check (prevents deadlocks + unfair comparisons) 
    recv_for_me = counts_all[:, rank, :]  # [src, E_local]
    local_max = expected_pca.max().to(torch.int32)
    global_max = _allreduce_max_i32(local_max).item()

    if global_max > capacity and SKIP_ON_OVERFLOW:
        if rank == 0:
            print(f"[SKIP] max_recv_count={global_max} > CAPACITY={capacity}. "
                  f"Increase CAPACITY or reduce (BATCH, SEQ, TOPK), or set SKIP_ON_OVERFLOW=0.")
        dist.destroy_process_group()
        return

     # ------------------------------------------------------------------
    # 2. Allocate buffers
    # ------------------------------------------------------------------
    buffers = alloc_shmem_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=send_payload.dtype,
    )
    dist.barrier()

    # fix the extra cost noneed without iris
    #buffers.heap_bases = buffers.heap_bases.to(device, non_blocking=True)
    # Optional clarity
    #if CLEAR_TOKEN_BUF:
    #    buffers.token_buf.zero_()

    # init baseline buffers (perf path: NO reorder)
    total_recv = int(recv_for_me.sum().item())
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
    


# ------------------------------------------------------------------
    # 3. Warmup — custom scatter a2a
    # ------------------------------------------------------------------
    # init expert weight for Grouped GEMM 
    expert_hidden_dim = hidden_dim * 4
    weights = torch.randn((e_local, hidden_dim, expert_hidden_dim), dtype=torch.bfloat16, device=device)
    for _ in range(WARMUP):
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

        # Stage-1 must be truly consumed in this benchmark
        wait_counts_ready(buffers.counts_ready, world_size)
        wait_token_sync_ready(buffers.token_sync, buffers.pca)
        packed_tokens = shuffle(
            buffers.token_buf,
            buffers.pca,          # consume real custom counts, not expected_pca
            buffers.token_sync,
            capacity,
        )
        local_expert_cnt = buffers.pca.sum(dim=1).to(torch.int32)
        _ = expert(packed_tokens, weights, local_expert_cnt, num_experts_total)

        torch.cuda.synchronize()
        dist.barrier()

        
      
     
    dist.barrier()          # local alignment
    dist.barrier()           # one-time global alignment (optional but safe)
    torch.cuda.synchronize()


    # ------------------------------------------------------------------
    # 4. Warmup — baseline (call exchange functions directly)
    # ------------------------------------------------------------------
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    warmup_err = None
    for _ in range(WARMUP):
        try:
            recv_counts_b = exchange_counts_a2a(
                send_counts,
                base_buffers_perf,
                strict_capacity=False,
                capacity=capacity,
            )
            recv_flat_b, _, _ = exchange_payload_a2a(
                send_payload,
                send_counts,
                recv_counts_b,
                base_buffers_perf,
            )
            local_expert_cnt_b = recv_counts_b.sum(dim=0).to(torch.int32)
            _ = expert(recv_flat_b, weights, local_expert_cnt_b, num_experts_total)
        except Exception as e:
            ok.zero_()
            warmup_err = repr(e)
            break

    ok_global = _sync_and_check(ok)
    if ok_global == 0:
        if rank == 0:
            print(f"[baseline warmup exception] {warmup_err}")
        _emit_result_line(
            rank,
            world_size=world_size,
            e_local=e_local,
            batch=batch,
            seq=seq,
            hidden_dim=hidden_dim,
            topk=topk,
            capacity=capacity,
            seed=seed,
            baseline_ms=None,
            custom_ms=None,
            status="baseline_warmup_error",
        )
        dist.destroy_process_group()
        return

    dist.barrier()
    torch.cuda.synchronize()
   # ------------------------------------------------------------------
    # 5. Timed — custom scatter a2a (Diagnostic Mode)
    # ------------------------------------------------------------------
    dist.barrier()
    torch.cuda.synchronize()
    
    t0_ev = torch.cuda.Event(enable_timing=True)
    comm_end_ev = torch.cuda.Event(enable_timing=True)
    shuffle_start_ev = torch.cuda.Event(enable_timing=True)
    shuffle_end_ev = torch.cuda.Event(enable_timing=True)
    expert_end_ev = torch.cuda.Event(enable_timing=True)

    custom_comm_ms = []
    custom_host_wait_ms = []
    custom_shuffle_ms = []
    custom_expert_ms = []
    custom_e2e_ms = []

    comm_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()

    # initial reset before the loop
    reset_custom_iter_state(buffers, clear_token_buf=bool(CLEAR_TOKEN_BUF))
    torch.cuda.synchronize()
    dist.barrier()   # everyone reset done before first launch

    for it in range(ITERS):
        t0_ev.record()
        comm_stream.wait_event(t0_ev)

        with torch.cuda.stream(comm_stream):
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
            comm_end_ev.record(comm_stream)

        # get Host time suspect 
        t_wait_start = time.perf_counter()
        wait_counts_ready(buffers.counts_ready, world_size)
        wait_token_sync_ready(buffers.token_sync, buffers.pca)
        t_wait_end = time.perf_counter()
        custom_host_wait_ms.append((t_wait_end - t_wait_start) * 1000.0)

        # wait_counts_ready ends launch following kernels immediately
        with torch.cuda.stream(comp_stream):
            shuffle_start_ev.record(comp_stream)
            packed_tokens = shuffle(
                buffers.token_buf,
                buffers.pca,
                buffers.token_sync,
                capacity,
            )
            shuffle_end_ev.record(comp_stream)

            local_expert_cnt = buffers.pca.sum(dim=1).to(torch.int32)
            _ = expert(packed_tokens, weights, local_expert_cnt, num_experts_total)
            expert_end_ev.record(comp_stream)

        expert_end_ev.synchronize()
        comm_end_ev.synchronize()

        custom_comm_ms.append(t0_ev.elapsed_time(comm_end_ev))
        custom_shuffle_ms.append(shuffle_start_ev.elapsed_time(shuffle_end_ev))
        custom_expert_ms.append(shuffle_end_ev.elapsed_time(expert_end_ev))
        custom_e2e_ms.append(t0_ev.elapsed_time(expert_end_ev))

        # epoch barrier synchornization
        dist.barrier()

        if it != ITERS - 1:
            reset_custom_iter_state(buffers, clear_token_buf=bool(CLEAR_TOKEN_BUF))
            torch.cuda.synchronize()
            dist.barrier()

    # ------------------------------------------------------------------
    # 6. Timed — baseline (Diagnostic Mode)
    # ------------------------------------------------------------------
    base_comm_ms = []
    base_expert_ms = []
    base_e2e_ms = []
    timed_err = None
    ok.fill_(1)

    t0_base = torch.cuda.Event(enable_timing=True)
    t_comm_base = torch.cuda.Event(enable_timing=True)
    t_exp_base = torch.cuda.Event(enable_timing=True)

    for _ in range(ITERS):
        torch.cuda.synchronize()
        dist.barrier()

        t0_base.record()

        try:
            recv_counts_b = exchange_counts_a2a(
                send_counts, base_buffers_perf,
                strict_capacity=False, capacity=capacity,
            )
            recv_flat, _, _ = exchange_payload_a2a(
                send_payload, send_counts, recv_counts_b, base_buffers_perf,
            )
            t_comm_base.record()

            local_expert_cnt_b = recv_counts_b.sum(dim=0).to(torch.int32)
            _ = expert(recv_flat, weights, local_expert_cnt_b, num_experts_total)
            t_exp_base.record()

        except Exception as e:
            ok.zero_()
            timed_err = repr(e)
            break

        t_exp_base.synchronize()
        
        base_comm_ms.append(t0_base.elapsed_time(t_comm_base))
        base_expert_ms.append(t_comm_base.elapsed_time(t_exp_base))
        base_e2e_ms.append(t0_base.elapsed_time(t_exp_base))

    ok_global = _allreduce_min_i32(ok).item()
    if ok_global == 0:
        if rank == 0:
            print(f"[baseline timed exception] {timed_err}")
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------
    # 7. Report perf (Diagnostic Breakdown)
    # ------------------------------------------------------------------
    def get_max_mean(times_list):
        if not times_list:
            return 0.0
        avg = sum(times_list) / len(times_list)
        val = torch.tensor([avg], device=device, dtype=torch.float32)
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        return val.item()

    # Custom times
    c_comm_avg = get_max_mean(custom_comm_ms)
    c_wait_avg = get_max_mean(custom_host_wait_ms)
    c_shuf_avg = get_max_mean(custom_shuffle_ms)
    c_exp_avg  = get_max_mean(custom_expert_ms)
    c_e2e_avg  = get_max_mean(custom_e2e_ms)

    # Baseline times
    b_comm_avg = get_max_mean(base_comm_ms)
    b_exp_avg  = get_max_mean(base_expert_ms)
    b_e2e_avg  = get_max_mean(base_e2e_ms)

    if rank == 0:
        print("\n" + "="*50)
        print(" DIAGNOSTIC BREAKDOWN (max-rank mean over iters) ")
        print("="*50)
        print(f"World: {world_size} | E_local: {e_local} | Cap: {capacity} | H: {hidden_dim} | Batch: {batch} | Seq: {seq}")
        print("-"*50)
        print(f"【Baseline (PyTorch)】")
        print(f"  ├─ Comm (Counts+Tokens): {b_comm_avg:.3f} ms")
        print(f"  ├─ Compute (Expert):     {b_exp_avg:.3f} ms")
        print(f"  └─ End-to-End:           {b_e2e_avg:.3f} ms")
        print("-"*50)
        print(f"【Custom (SymmMem)】")
        print(f"  ├─ Comm (custom_a2a):    {c_comm_avg:.3f} ms")
        print(f"  ├─ Host Polling Wait:    {c_wait_avg:.3f} ms  <-- (If this is high, P2 is mandatory)")
        print(f"  ├─ Shuffle:              {c_shuf_avg:.3f} ms  <-- (If this is high, P4 is mandatory)")
        print(f"  ├─ Compute (Expert):     {c_exp_avg:.3f} ms  <-- (If this is high, P3 is mandatory)")
        print(f"  └─ End-to-End (Overlap): {c_e2e_avg:.3f} ms")
        print("="*50)
        if c_e2e_avg > 0:
            print(f"Speedup (Baseline/Custom E2E): {b_e2e_avg / c_e2e_avg:.3f}x")
        print("="*50 + "\n")

    _emit_result_line(
        rank,
        world_size=world_size,
        e_local=e_local,
        batch=batch,
        seq=seq,
        hidden_dim=hidden_dim,
        topk=topk,
        capacity=capacity,
        seed=seed,
        baseline_ms=b_e2e_avg,
        custom_ms=c_e2e_avg,
        status="ok",
    )

# ------------------------------------------------------------------
    # 8. Optional profile passes
    # -----------------------------------------------------------------
   
    dist.barrier()
    torch.cuda.synchronize()

    if PROFILE_CUSTOM:
        _profile_pass_custom(
            rank,
            buffers,
            send_payload,
            send_counts,
            dst_offsets,
            num_experts_total,
            capacity,
            world_size,
            weights,
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

# ------------------------------------------------------------------
    # 9. Optional correctness check
    # ------------------------------------------------------------------
    if CHECK:
        
        dist.barrier()
        torch.cuda.synchronize()
        # If overflow would occur, correctness is undefined unless you implement capacity capping.
        if global_max > capacity and SKIP_ON_OVERFLOW:
            if rank == 0:
                print(f"[SKIP] max_recv_count={global_max} > CAPACITY={capacity}. "
                    f"Increase CAPACITY or reduce (BATCH, SEQ, TOPK), or set SKIP_ON_OVERFLOW=0.")
            _emit_result_line(
                rank,
                world_size=world_size,
                e_local=e_local,
                batch=batch,
                seq=seq,
                hidden_dim=hidden_dim,
                topk=topk,
                capacity=capacity,
                seed=seed,
                baseline_ms=None,
                custom_ms=None,
                status="skip_overflow",
            )
            dist.destroy_process_group()
            return

        
        
        # Run custom scatter a2a once
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
        wait_counts_ready(buffers.counts_ready, world_size)
        wait_token_sync_ready(buffers.token_sync, buffers.pca)

        torch.cuda.synchronize()
        dist.barrier()

        # Run baseline once and reorder into [e_local, world, capacity, H]
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

        recv_counts = exchange_counts_a2a(
            send_counts, base_buffers_ck, strict_capacity=False, capacity=capacity,
        )
        recv_flat, _, _ = exchange_payload_a2a(
            send_payload, send_counts, recv_counts, base_buffers_ck,
        )
        expected_buf = reorder_flat_to_token_buf(
            recv_flat,
            recv_counts,
            capacity,
            base_buffers_ck.token_buf,
        )
        torch.cuda.synchronize()
        dist.barrier()

        # Compare only valid region:
        # recv_counts is [src, e], while _masked_stats expects [e, src]
        counts_mat = recv_counts.transpose(0, 1).contiguous()

        max_diff_t, custom_sum_t, expected_sum_t = _masked_stats(
            buffers.token_buf,
            expected_buf,
            counts_mat,
        )
        dist.all_reduce(max_diff_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(custom_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(expected_sum_t, op=dist.ReduceOp.SUM)

        if rank == 0:
            print("=== Correctness Report ===")
            print(f"TOKEN max|diff| (global): {max_diff_t.item():.6g}")
            print(f"custom_sum(valid):   {custom_sum_t.item():.6g}")
            print(f"expected_sum(valid): {expected_sum_t.item():.6g}")
            ok_pass = max_diff_t.item() == 0.0
            print("PASS" if ok_pass else "FAIL")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", "4"))
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
        bench_scatter,
        args=(world_size, batch, seq, hidden_dim, topk, e_local, capacity, seed),
        nprocs=world_size,
        join=True,
    )