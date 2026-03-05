"""
Benchmark: AllToAllGatherOp (pull/gather) vs PyTorch baseline all-to-all.

Usage:
    srun -N1 --ntasks-per-node=2 python benchmark_gather.py

Environment variables:
    WORLD_SIZE, BATCH, SEQ, HIDDEN, TOPK, E_LOCAL, CAPACITY, SEED
    WARMUP, ITERS, CHECK, IRIS_HEAP_GIB
    PROFILE_CUSTOM, PROFILE_BASELINE, PROFILE_ITERS, TRACE_DIR
"""

import os
from typing import Tuple, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler

import iris

from baseline import (
    run_baseline_ref,
    init_baseline_buffers,
    exchange_counts_a2a,
    exchange_payload_a2a,
)
from layers.all_to_all_gather import gather_a2a
from utils import (
    set_seed,
    gen_local_tokens,
    gen_router,
    route_and_pack_padding_free,
    _build_dst_offsets,
    _allreduce_max_i32,
    _allreduce_min_i32,
    _sync_and_check,
)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
WARMUP         = int(os.getenv("WARMUP", "5"))
ITERS          = int(os.getenv("ITERS", "20"))
CHECK          = int(os.getenv("CHECK", "1"))
SKIP_ON_OVERFLOW = int(os.getenv("SKIP_ON_OVERFLOW", "1"))
IRIS_HEAP_GIB  = int(os.getenv("IRIS_HEAP_GIB", "100"))

PROFILE_CUSTOM   = int(os.getenv("PROFILE_CUSTOM", "0")) == 1
PROFILE_BASELINE = int(os.getenv("PROFILE_BASELINE", "0")) == 1
PROFILE_ITERS    = int(os.getenv("PROFILE_ITERS", "3"))
TRACE_DIR        = os.getenv("TRACE_DIR", ".")
TEST_MODE = os.getenv("TEST_MODE", "free")
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_iris_shmem():
    heap_size = IRIS_HEAP_GIB * (2 ** 30)
    return iris.iris(heap_size)


def _alloc_shmem_gather_buffers(
    shmem,
    world_size: int,
    e_local: int,
    total_rows: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
    send_counts: torch.Tensor,
    dst_offsets: torch.Tensor,
    tokens: torch.Tensor,
):
    """
    Allocate symmetric memory buffers needed by gather_a2a and populate them
    with this rank's data.  Returns (shmem_cnts, shmem_offsets, shmem_tokens, heap_bases).
    """
    shmem_cnts    = shmem.zeros((world_size, e_local), dtype=torch.int64, device="cuda")
    shmem_offsets = shmem.zeros((world_size,),         dtype=torch.int64, device="cuda")
    shmem_tokens  = shmem.zeros((total_rows, hidden_dim), dtype=token_dtype, device="cuda")
    heap_bases    = shmem.get_heap_bases()

    shmem_cnts.copy_(send_counts.to(torch.int64))
    shmem_offsets.copy_(dst_offsets.to(torch.int64))
    shmem_tokens.copy_(tokens)

    return shmem_cnts, shmem_offsets, shmem_tokens, heap_bases


# ---------------------------------------------------------------------------
# Profile helpers (separate from perf timing)
# ---------------------------------------------------------------------------

def _profile_pass_custom(
    rank: int,
    shmem,
    shmem_cnts,
    shmem_offsets,
    shmem_tokens,
    heap_bases,
    e_local: int,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_gather_custom_rank{rank}.json")

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

    try:
        for i in range(PROFILE_ITERS):
            if do_trace:
                ctx = torch.profiler.record_function(f"gather_custom_prof_iter_{i}")
                ctx.__enter__()

            torch.cuda.synchronize()
            _ = gather_a2a(shmem_cnts, shmem_offsets, heap_bases, e_local, shmem_tokens)
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
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    base_buffers_perf,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_gather_baseline_rank{rank}.json")

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

    try:
        for i in range(PROFILE_ITERS):
            if do_trace:
                ctx = torch.profiler.record_function(f"gather_baseline_prof_iter_{i}")
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
                do_reorder=False,
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


# ---------------------------------------------------------------------------
# Main benchmark entry point (one per rank)
# ---------------------------------------------------------------------------

def bench_gather(
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
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    set_seed(seed, rank)

    # ------------------------------------------------------------------
    # 1. Generate input tokens + routing
    # ------------------------------------------------------------------
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
    total_rows  = send_payload.shape[0]

    # Gather global counts for overflow check + correctness
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    counts_all = torch.stack(all_send_counts, dim=0).contiguous()  # [src, dst, E_local]

    # Overflow check
    recv_for_me = counts_all[:, rank, :]  # [src, E_local]
    local_max = recv_for_me.max().to(torch.int32)
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
    shmem = _init_iris_shmem()

    # Symmetric buffers for gather a2a
    shmem_cnts, shmem_offsets, shmem_tokens, heap_bases = _alloc_shmem_gather_buffers(
        shmem, world_size, e_local, total_rows, hidden_dim,
        send_payload.dtype, send_counts, dst_offsets, send_payload,
    )
    heap_bases = heap_bases.to(device, non_blocking=True)
    shmem.barrier()

    # Baseline buffers (perf path: no reorder)
    total_recv = int(recv_for_me.sum().item())
    base_buffers_perf = init_baseline_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=send_payload.dtype,
        device=device,
        total_recv=total_recv,
        allocate_token_buf=False,
    )

    # ------------------------------------------------------------------
    # 3. Warmup — custom gather a2a
    # ------------------------------------------------------------------
    for _ in range(WARMUP):
        _ = gather_a2a(shmem_cnts, shmem_offsets, heap_bases, e_local, shmem_tokens)
        torch.cuda.synchronize()
        shmem.barrier()

    shmem.barrier()
    dist.barrier()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 4. Warmup — baseline (call exchange functions directly)
    # ------------------------------------------------------------------
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    warmup_err = None
    for _ in range(WARMUP):
        try:
            _ = exchange_counts_a2a(
                send_counts, base_buffers_perf, strict_capacity=False, capacity=capacity,
            )
            _ = exchange_payload_a2a(
                send_payload, send_counts, base_buffers_perf.recv_counts, base_buffers_perf,
            )
        except Exception as e:
            ok.zero_()
            warmup_err = repr(e)
            break

    ok_global = _sync_and_check(ok)
    if ok_global == 0:
        if rank == 0:
            print(f"[baseline warmup exception] {warmup_err}")
        dist.destroy_process_group()
        return

    dist.barrier()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 5. Timed — custom gather a2a
    # ------------------------------------------------------------------
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    custom_times = []

    for _ in range(ITERS):
        torch.cuda.synchronize()
        start_ev.record()

        _ = gather_a2a(shmem_cnts, shmem_offsets, heap_bases, e_local, shmem_tokens)

        end_ev.record()
        end_ev.synchronize()
        custom_times.append(start_ev.elapsed_time(end_ev))
        shmem.barrier()

    # ------------------------------------------------------------------
    # 6. Timed — baseline (direct exchange calls, no internal timing)
    # ------------------------------------------------------------------
    dist.barrier()
    torch.cuda.synchronize()

    baseline_times = []
    timed_err = None
    ok.fill_(1)

    for _ in range(ITERS):
        try:
            torch.cuda.synchronize()
            start_ev.record()

            recv_counts_b = exchange_counts_a2a(
                send_counts, base_buffers_perf,
                strict_capacity=False, capacity=capacity,
            )
            exchange_payload_a2a(
                send_payload, send_counts, recv_counts_b, base_buffers_perf,
            )

            end_ev.record()
            end_ev.synchronize()
            baseline_times.append(start_ev.elapsed_time(end_ev))
            shmem.barrier()

        except Exception as e:
            ok.zero_()
            timed_err = repr(e)
            break

    ok_global = _allreduce_min_i32(ok).item()
    if ok_global == 0:
        if rank == 0:
            print(f"[baseline timed exception] {timed_err}")
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------
    # 7. Report perf
    # ------------------------------------------------------------------
    custom_ms = torch.tensor(
        [sum(custom_times) / len(custom_times)], device=device, dtype=torch.float32,
    )
    base_ms = torch.tensor(
        [sum(baseline_times) / len(baseline_times)], device=device, dtype=torch.float32,
    )

    custom_ms_max = custom_ms.clone()
    base_ms_max   = base_ms.clone()
    dist.all_reduce(custom_ms_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(base_ms_max,   op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Gather A2A — COMM-only Perf (max-rank mean) ===")
        print(f"world={world_size}  E_local={e_local}  CAP={capacity}  "
              f"H={hidden_dim}  batch={batch}  seq={seq}  topk={topk}")
        print(f"custom  (gather a2a):  {custom_ms_max.item():.3f} ms")
        print(f"baseline (PyTorch a2a): {base_ms_max.item():.3f} ms")
        print(f"speedup (baseline/custom): {base_ms_max.item() / custom_ms_max.item():.3f}x")

    # ------------------------------------------------------------------
    # 8. Optional profile passes
    # ------------------------------------------------------------------
    dist.barrier()
    torch.cuda.synchronize()

    if PROFILE_CUSTOM:
        _profile_pass_custom(
            rank, shmem, shmem_cnts, shmem_offsets, shmem_tokens, heap_bases, e_local,
        )

    dist.barrier()
    torch.cuda.synchronize()

    if PROFILE_BASELINE:
        _profile_pass_baseline(
            rank, shmem, world_size, e_local, capacity, hidden_dim,
            send_payload, send_counts, base_buffers_perf,
        )

    # ------------------------------------------------------------------
    # 9. Optional correctness check
    # ------------------------------------------------------------------
    if CHECK:
        dist.barrier()
        torch.cuda.synchronize()

        if global_max > capacity:
            if rank == 0:
                print(f"[SKIP correctness] max_recv_count={global_max} > CAPACITY={capacity}.")
            dist.destroy_process_group()
            return

        # Run custom gather a2a once
        gathered = gather_a2a(shmem_cnts, shmem_offsets, heap_bases, e_local, shmem_tokens)
        torch.cuda.synchronize()
        shmem.barrier()

        # Run baseline once
        base_buffers_ck = init_baseline_buffers(
            world_size=world_size,
            e_local=e_local,
            capacity=capacity,
            hidden_dim=hidden_dim,
            token_dtype=send_payload.dtype,
            device=device,
            total_recv=total_recv,
            allocate_token_buf=False,
        )

        recv_counts = exchange_counts_a2a(
            send_counts, base_buffers_ck, strict_capacity=False, capacity=capacity,
        )
        recv_flat, _, _ = exchange_payload_a2a(
            send_payload, send_counts, recv_counts, base_buffers_ck,
        )
        torch.cuda.synchronize()

        # Reorder baseline recv_flat (src-major, expert-minor) into the same
        # expert-major layout that gather_a2a produces via write_meta.
        expected = torch.zeros_like(gathered)
        write_off = {}
        running = 0
        for e in range(e_local):
            for src in range(world_size):
                c = int(recv_counts[src, e].item())
                write_off[(src, e)] = running
                running += c

        baseline_off = 0
        for src in range(world_size):
            for e in range(e_local):
                c = int(recv_counts[src, e].item())
                if c > 0:
                    wo = write_off[(src, e)]
                    expected[wo:wo + c] = recv_flat[baseline_off:baseline_off + c]
                baseline_off += c

        # Compare
        max_diff = (gathered.float() - expected.float()).abs().max().item()
        gathered_sum = gathered.float().sum().item()
        expected_sum = expected.float().sum().item()

        max_diff_t = torch.tensor([max_diff], device=device, dtype=torch.float32)
        dist.all_reduce(max_diff_t, op=dist.ReduceOp.MAX)

        if rank == 0:
            print("=== Correctness Report ===")
            print(f"TOKEN max|diff| (global): {max_diff_t.item():.6g}")
            ok_pass = max_diff_t.item() == 0.0
            print("PASS" if ok_pass else "FAIL")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", "8"))
    batch      = int(os.getenv("BATCH", "4"))
    seq        = int(os.getenv("SEQ", "2048"))
    hidden_dim = int(os.getenv("HIDDEN", "4096"))
    topk       = int(os.getenv("TOPK", "2"))
    e_local    = int(os.getenv("E_LOCAL", "4"))
    capacity   = int(os.getenv("CAPACITY", "8096"))
    seed       = int(os.getenv("SEED", "42"))

    if not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        bench_gather,
        args=(world_size, batch, seq, hidden_dim, topk, e_local, capacity, seed),
        nprocs=world_size,
        join=True,
    )
