"""
Benchmark: AllToAllGatherOp (pull/gather) vs PyTorch baseline all-to-all.

Usage:
    srun -N1 --ntasks-per-node=2 python benchmark_gather.py

Environment variables:
    WORLD_SIZE, BATCH, SEQ, HIDDEN, TOPK, E_LOCAL, CAPACITY, SEED
    WARMUP, ITERS, CHECK, SKIP_ON_OVERFLOW
    PROFILE_CUSTOM, PROFILE_BASELINE, PROFILE_ITERS, TRACE_DIR
"""


import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler

from baseline import (
    init_baseline_buffers,
    exchange_counts_a2a,
    exchange_payload_a2a,
)
from layers.all_to_all_gather import gather_a2a
from layers.expert import expert
from utils import (
    set_seed,
    gen_local_tokens,
    gen_router,
    route_and_pack_padding_free,
    _build_dst_offsets,
    _allreduce_max_i32,
    _allreduce_min_i32,
    _sync_and_check,
    _alloc_symm_gather_buffers,
)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
WARMUP         = int(os.getenv("WARMUP", "5"))
ITERS          = int(os.getenv("ITERS", "20"))
CHECK          = int(os.getenv("CHECK", "1"))
SKIP_ON_OVERFLOW = int(os.getenv("SKIP_ON_OVERFLOW", "1"))
CLEAR_TOKEN_BUF  = int(os.getenv("CLEAR_TOKEN_BUF", "0"))

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

def _as_cuda_i64_ptr_tensor(ptrs, device):
    return torch.tensor(list(ptrs), dtype=torch.int64, device=device)




# ---------------------------------------------------------------------------
# Profile helpers (separate from perf timing)
# ---------------------------------------------------------------------------

def _profile_pass_custom(
    rank: int,
    symm,
    e_local: int,
    weights: torch.Tensor,
    local_expert_cnt: torch.Tensor,
    num_experts_total: int,
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
            _ = expert(gathered, weights, local_expert_cnt, num_experts_total)

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

def _profile_pass_baseline(
    rank: int,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    base_buffers_perf,
    weights: torch.Tensor,
    num_experts_total: int,
):
    do_trace = (rank == 0)
    trace_path = os.path.join(TRACE_DIR, f"trace_org_baseline_rank{rank}.json")

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
                ctx = torch.profiler.record_function(f"org_baseline_prof_iter_{i}")
                ctx.__enter__()

            torch.cuda.synchronize()
            dist.barrier()

            # Stage 1: counts exchange
            recv_counts_b = exchange_counts_a2a(
                send_counts,
                base_buffers_perf,
                strict_capacity=False,
                capacity=capacity,
            )

            # Stage 2: payload exchange
            recv_flat_b, _, _ = exchange_payload_a2a(
                send_payload,
                send_counts,
                recv_counts_b,
                base_buffers_perf,
            )

            # Stage 3: expert compute
            local_expert_cnt_b = recv_counts_b.sum(dim=0).to(torch.int32)
            _ = expert(recv_flat_b, weights, local_expert_cnt_b, num_experts_total)

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
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
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
        tokens, router, topk, world_size, num_experts_total,
    )
    send_counts = send_counts.to(torch.int32).contiguous()
    dst_offsets = _build_dst_offsets(send_counts)
    total_rows = send_payload.shape[0]

    # Gather global counts for overflow check + local expert counts
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    counts_all = torch.stack(all_send_counts, dim=0).contiguous()  # [src, dst, e_local]

    recv_for_me = counts_all[:, rank, :]  # [src, e_local]
    local_max = recv_for_me.max().to(torch.int32)
    global_max = _allreduce_max_i32(local_max).item()

    if global_max > capacity and SKIP_ON_OVERFLOW:
        if rank == 0:
            print(f"[SKIP] max_recv_count={global_max} > CAPACITY={capacity}. "
                  f"Increase CAPACITY or reduce (BATCH, SEQ, TOPK), or set SKIP_ON_OVERFLOW=0.")
        dist.destroy_process_group()
        return

    # gather path uses packed output directly, so local expert count is known here
    local_expert_cnt = recv_for_me.sum(dim=0).to(torch.int32).contiguous()
    total_recv = int(local_expert_cnt.sum().item())

    # ------------------------------------------------------------------
    # 2. Allocate buffers
    # ------------------------------------------------------------------
    symm = _alloc_symm_gather_buffers(
        world_size,
        e_local,
        total_rows,
        hidden_dim,
        send_payload.dtype,
        send_counts,
        dst_offsets,
        send_payload,
        device,
    )
    dist.barrier()

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

    # expert weights
    expert_hidden_dim = hidden_dim * 4
    weights = torch.randn(
        (e_local, hidden_dim, expert_hidden_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    # ------------------------------------------------------------------
    # 3. Warmup — custom gather a2a + expert
    # ------------------------------------------------------------------
    for _ in range(WARMUP):
        gathered = gather_a2a(
            symm["cnts"],
            symm["offsets"],
            symm["tokens"],
            symm["cnts_bases"],
            symm["offsets_bases"],
            symm["tokens_bases"],
            e_local,
        )
        _ = expert(gathered, weights, local_expert_cnt, num_experts_total)
        torch.cuda.synchronize()
        dist.barrier()

    dist.barrier()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 4. Warmup — baseline a2a + expert
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
    # 5. Timed — custom gather a2a + expert
    # ------------------------------------------------------------------
    custom_comm_ms = []
    custom_expert_ms = []
    custom_e2e_ms = []

    t0_custom = torch.cuda.Event(enable_timing=True)
    t_comm_custom = torch.cuda.Event(enable_timing=True)
    t_exp_custom = torch.cuda.Event(enable_timing=True)

    for _ in range(ITERS):
        torch.cuda.synchronize()
        dist.barrier()

        t0_custom.record()

        gathered = gather_a2a(
            symm["cnts"],
            symm["offsets"],
            symm["tokens"],
            symm["cnts_bases"],
            symm["offsets_bases"],
            symm["tokens_bases"],
            e_local,
        )
        t_comm_custom.record()

        _ = expert(gathered, weights, local_expert_cnt, num_experts_total)
        t_exp_custom.record()

        t_exp_custom.synchronize()

        custom_comm_ms.append(t0_custom.elapsed_time(t_comm_custom))
        custom_expert_ms.append(t_comm_custom.elapsed_time(t_exp_custom))
        custom_e2e_ms.append(t0_custom.elapsed_time(t_exp_custom))

    # ------------------------------------------------------------------
    # 6. Timed — baseline a2a + expert
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
            t_comm_base.record()

            local_expert_cnt_b = recv_counts_b.sum(dim=0).to(torch.int32)
            _ = expert(recv_flat_b, weights, local_expert_cnt_b, num_experts_total)
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
            status="baseline_timed_error",
        )
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------
    # 7. Report perf
    # ------------------------------------------------------------------
    def get_max_mean(times_list):
        if not times_list:
            return 0.0
        avg = sum(times_list) / len(times_list)
        val = torch.tensor([avg], device=device, dtype=torch.float32)
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        return val.item()

    c_comm_avg = get_max_mean(custom_comm_ms)
    c_exp_avg  = get_max_mean(custom_expert_ms)
    c_e2e_avg  = get_max_mean(custom_e2e_ms)

    b_comm_avg = get_max_mean(base_comm_ms)
    b_exp_avg  = get_max_mean(base_expert_ms)
    b_e2e_avg  = get_max_mean(base_e2e_ms)

    if rank == 0:
        print("\n" + "=" * 50)
        print(" GATHER DIAGNOSTIC BREAKDOWN (max-rank mean over iters) ")
        print("=" * 50)
        print(f"World: {world_size} | E_local: {e_local} | Cap: {capacity} | H: {hidden_dim} | Batch: {batch} | Seq: {seq}")
        print("-" * 50)
        print("【Baseline (PyTorch)】")
        print(f"  ├─ Comm (Counts+Tokens): {b_comm_avg:.3f} ms")
        print(f"  ├─ Compute (Expert):     {b_exp_avg:.3f} ms")
        print(f"  └─ End-to-End:           {b_e2e_avg:.3f} ms")
        print("-" * 50)
        print("【Custom (Gather)】")
        print(f"  ├─ Comm (gather_a2a):    {c_comm_avg:.3f} ms")
        print(f"  ├─ Compute (Expert):     {c_exp_avg:.3f} ms")
        print(f"  └─ End-to-End:           {c_e2e_avg:.3f} ms")
        print("=" * 50)
        if c_e2e_avg > 0:
            print(f"Speedup (Baseline/Custom E2E): {b_e2e_avg / c_e2e_avg:.3f}x")
        print("=" * 50 + "\n")

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
    # ------------------------------------------------------------------
    dist.barrier()
    torch.cuda.synchronize()

    if PROFILE_CUSTOM:
        _profile_pass_custom(
            rank,
            symm,
            e_local,
            weights,
            local_expert_cnt,
            num_experts_total,
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
            weights,
            num_experts_total,
        )

    # ------------------------------------------------------------------
    # 9. Optional correctness check (COMM/layout only)
    # ------------------------------------------------------------------
    if CHECK:
        dist.barrier()
        torch.cuda.synchronize()

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
            send_counts,
            base_buffers_ck,
            strict_capacity=False,
            capacity=capacity,
        )
        recv_flat, _, _ = exchange_payload_a2a(
            send_payload,
            send_counts,
            recv_counts,
            base_buffers_ck,
        )
        torch.cuda.synchronize()

        # baseline recv_flat is src-major/expert-minor;
        # gather output is expert-major/src-major
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

        max_diff = (gathered.float() - expected.float()).abs().max().item()
        max_diff_t = torch.tensor([max_diff], device=device, dtype=torch.float32)
        dist.all_reduce(max_diff_t, op=dist.ReduceOp.MAX)

        if rank == 0:
            print("=== Correctness Report ===")
            print(f"TOKEN max|diff| (global): {max_diff_t.item():.6g}")
            print("PASS" if max_diff_t.item() == 0.0 else "FAIL")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", "4"))
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
