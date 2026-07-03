"""
Benchmark: AllToAllGatherOp (pull/gather) vs PyTorch baseline all-to-all.

Usage:
    srun -N1 --ntasks-per-node=2 python benchmark_gather.py

Environment variables:
    WORLD_SIZE, BATCH, SEQ, HIDDEN, TOPK, E_LOCAL, CAPACITY, CAPACITY_PAD,
    CAPACITY_ALIGN, SEED, MODE, WARMUP, ITERS, CHECK, SKIP_ON_OVERFLOW
    RUN_BACKWARD
    PROFILE_CUSTOM, PROFILE_BASELINE, PROFILE_ITERS, TRACE_DIR
"""


import os
import math
import datetime
import csv
import time
from typing import Optional
import triton
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler

from baseline import (
    init_baseline_buffers,
    exchange_counts_a2a,
    exchange_payload_a2a,
    baseline_expert_sequential,
)

from layers.all_to_all_gather import (
    gather_a2a,
    alloc_gather_overlap_workspace,
    reset_gather_overlap_workspace,
    gather_a2a_prepare_metadata_overlap,
    gather_a2a_launch_tiles_overlap,
)

from layers.expert import expert, expert_overlap, CrossLayerExpertPair
from layers.gather_expert_op import gather_expert
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
WARMUP         = int(os.getenv("WARMUP", "10"))
ITERS          = int(os.getenv("ITERS", "50"))
CHECK          = int(os.getenv("CHECK", "1"))
SKIP_ON_OVERFLOW = int(os.getenv("SKIP_ON_OVERFLOW", "1"))
CLEAR_TOKEN_BUF  = int(os.getenv("CLEAR_TOKEN_BUF", "0"))
MODE           = os.getenv("MODE", "all")
RUN_BACKWARD   = int(os.getenv("RUN_BACKWARD", "1")) == 1
USE_FUSED_GATHER_EXPERT = int(os.getenv("USE_FUSED_GATHER_EXPERT", "0")) == 1
BACKWARD_OVERLAP = int(os.getenv("BACKWARD_OVERLAP", "0")) == 1
CROSS_LAYER_BACKWARD = int(os.getenv("CROSS_LAYER_BACKWARD", "0")) == 1
PRINT_GRID = int(os.getenv("PRINT_GRID", "0")) == 1
NCCL_TIMEOUT_S = int(os.getenv("NCCL_TIMEOUT_S", "60"))
DIAG_FORCE_FULL_SYNC = int(os.getenv("DIAG_FORCE_FULL_SYNC", "0")) == 1
DIAG_HEADSTART_US = int(os.getenv("DIAG_HEADSTART_US", "0"))
DEBUG_BACKWARD_STAGES = int(os.getenv("DEBUG_BACKWARD_STAGES", "0")) == 1
OVERLAP_SPIN_PROBE = int(os.getenv("OVERLAP_SPIN_PROBE", "0")) == 1
OVERLAP_PROBE_WAIT_CYCLES = int(os.getenv("OVERLAP_PROBE_WAIT_CYCLES", "1")) == 1
OVERLAP_PROBE_DIR = os.getenv("OVERLAP_PROBE_DIR", "overlap_probe_results")
OVERLAP_PROBE_LABEL = os.getenv("OVERLAP_PROBE_LABEL", "")


def _set_grid_print_active(active: bool):
    if PRINT_GRID:
        os.environ["PRINT_GRID_ACTIVE"] = "1" if active else "0"

PROFILE_CUSTOM   = int(os.getenv("PROFILE_CUSTOM", "0")) == 1
PROFILE_BASELINE = int(os.getenv("PROFILE_BASELINE", "0")) == 1
PROFILE_ITERS    = int(os.getenv("PROFILE_ITERS", "3"))
TRACE_DIR        = os.getenv("TRACE_DIR", ".")

BLOCK_E = int(os.getenv("BLOCK_E", "128"))
COUNTS_WARPS = int(os.getenv("COUNTS_WARPS", "4"))
BLOCK_K = int(os.getenv("BLOCK_K", "128"))
TOKENS_WARPS = int(os.getenv("TOKENS_WARPS", "4"))
CONSUME_TOKEN_SYNC = int(os.getenv("CONSUME_TOKEN_SYNC", "0"))
# GEMM-side sweep dimension.
# If not specified, keep the old behavior: expert_hidden_dim = hidden_dim * 4.
EXPERT_HIDDEN_ENV = os.getenv("EXPERT_HIDDEN", "")
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# new 4 sweep only
def _fmt(x):
    return "" if x is None else f"{float(x):.6f}"


def _safe_ratio(num, den):
    if num is None or den is None or den <= 0:
        return None
    return float(num) / float(den)


def _fmt_stat(stats, idx):
    if stats is None:
        return ""
    return _fmt(stats[idx])


def _median(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return float(values[mid])
    return 0.5 * (float(values[mid - 1]) + float(values[mid]))


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo]) * (1.0 - frac) + float(values[hi]) * frac


def _pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    den_x = sum(v * v for v in dx)
    den_y = sum(v * v for v in dy)
    if den_x <= 0 or den_y <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(den_x * den_y)


def _stat_summary(values):
    if not values:
        return None
    zeros = sum(1 for v in values if v == 0)
    return min(values), _median(values), _percentile(values, 0.95), max(values), zeros / len(values)


def _dump_overlap_probe(
    *,
    rank: int,
    label: str,
    state,
    world_size: int,
    e_local: int,
):
    if not OVERLAP_SPIN_PROBE or state.spin_iters is None:
        return

    torch.cuda.synchronize()
    out_dir = os.path.join(OVERLAP_PROBE_DIR, label or "overlap_probe")
    os.makedirs(out_dir, exist_ok=True)

    spin = state.spin_iters.detach().cpu().tolist()
    wait = state.wait_cycles.detach().cpu().tolist()
    pids = state.prog_pid.detach().cpu().tolist()
    cnts = state.prog_cnt.detach().cpu().tolist()

    max_m_tiles = state.ready.shape[-1]
    num_n_tiles = state.probe_num_n_tiles
    tiles_per_es = max_m_tiles * num_n_tiles
    total = len(pids)
    early = 0
    active_rows = []

    csv_path = os.path.join(out_dir, f"{label or 'overlap_probe'}_rank{rank}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "pid", "e", "src", "tm", "tn", "prog_cnt", "spin_iters", "wait_cycles"])
        for pid, prog_cnt, spin_iters, wait_cycles in zip(pids, cnts, spin, wait):
            pid = int(pid)
            if pid < 0:
                # Unwritten slots should not happen when the kernel runs all programs.
                e = src = tm = tn = -1
            else:
                es_idx = pid // tiles_per_es
                rem = pid % tiles_per_es
                e = es_idx // world_size
                src = es_idx % world_size
                tm = rem // num_n_tiles
                tn = rem % num_n_tiles
            writer.writerow([rank, pid, e, src, tm, tn, prog_cnt, spin_iters, wait_cycles])
            if int(prog_cnt) == -1:
                early += 1
            elif int(prog_cnt) >= 0:
                active_rows.append((pid, int(prog_cnt), int(spin_iters), int(wait_cycles)))

    active = len(active_rows)
    spin_values = [row[2] for row in active_rows if row[2] >= 0]
    wait_values = [row[3] for row in active_rows if row[3] >= 0]
    cnt_values = [row[1] for row in active_rows if row[2] >= 0]
    pid_for_spin = [row[0] for row in active_rows if row[2] >= 0]
    cnt_for_wait = [row[1] for row in active_rows if row[3] >= 0]

    spin_stats = _stat_summary(spin_values)
    wait_stats = _stat_summary(wait_values)
    spin_cnt_corr = _pearson(spin_values, cnt_values)
    wait_cnt_corr = _pearson(wait_values, cnt_for_wait)
    spin_pid_corr = _pearson(spin_values, pid_for_spin)

    def _stats_text(stats):
        if stats is None:
            return "unavailable"
        mn, med, p95, mx, frac_zero = stats
        return f"min={mn} median={med:.3f} p95={p95:.3f} max={mx} frac_zero={frac_zero:.6f}"

    print(
        f"[OVERLAP_PROBE_SUMMARY][{label or 'overlap_probe'}][rank{rank}] "
        f"csv={csv_path} total_programs={total} early_return={early} active={active} "
        f"wait_cycles_available={bool(wait_values)} "
        f"spin_iters({_stats_text(spin_stats)}) "
        f"wait_cycles({_stats_text(wait_stats)}) "
        f"corr_spin_cnt={'' if spin_cnt_corr is None else f'{spin_cnt_corr:.6f}'} "
        f"corr_wait_cnt={'' if wait_cnt_corr is None else f'{wait_cnt_corr:.6f}'} "
        f"corr_spin_pid={'' if spin_pid_corr is None else f'{spin_pid_corr:.6f}'}",
        flush=True,
    )


def _emit_header_line(rank: int):
    if rank != 0:
        return

    print(
        "HEADER_V2,"
        "world_size,experts_total,e_local,topk,batch,seq,"
        "hidden_dim,expert_hidden_dim,capacity,seed,mode,"
        "tpe_mean,tpe_max,tpe_min,src_expert_max,"
        "baseline_comm_p50,baseline_comm_p90,baseline_comm_p99,"
        "baseline_expert_p50,baseline_expert_p90,baseline_expert_p99,"
        "baseline_e2e_p50,baseline_e2e_p90,baseline_e2e_p99,"
        "custom_no_overlap_comm_p50,custom_no_overlap_comm_p90,custom_no_overlap_comm_p99,"
        "custom_no_overlap_reorder_p50,custom_no_overlap_reorder_p90,custom_no_overlap_reorder_p99,"
        "custom_no_overlap_expert_p50,custom_no_overlap_expert_p90,custom_no_overlap_expert_p99,"
        "custom_no_overlap_e2e_p50,custom_no_overlap_e2e_p90,custom_no_overlap_e2e_p99,"
        "custom_metadata_p50,custom_metadata_p90,custom_metadata_p99,"
        "custom_overlapped_p50,custom_overlapped_p90,custom_overlapped_p99,"
        "custom_overlap_e2e_p50,custom_overlap_e2e_p90,custom_overlap_e2e_p99,"
        "custom_backward_forward_p50,custom_backward_forward_p90,custom_backward_forward_p99,"
        "custom_backward_backward_p50,custom_backward_backward_p90,custom_backward_backward_p99,"
        "custom_backward_e2e_p50,custom_backward_e2e_p90,custom_backward_e2e_p99,"
        "no_overlap_e2e_speedup_p50,overlap_e2e_speedup_p50,status",
        flush=True,
    )


def _emit_result_line(
    rank: int,
    *,
    world_size: int,
    e_local: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    expert_hidden_dim: Optional[int],
    topk: int,
    capacity: int,
    seed: int,
    mode: str,
    # Real routing statistics. These are the y-axis quantities.
    tpe_mean: Optional[float] = None,          # mean tokens per local expert, across ranks
    tpe_max: Optional[float] = None,           # max tokens per local expert, across ranks
    tpe_min: Optional[float] = None,           # min tokens per local expert, across ranks
    src_expert_max: Optional[float] = None,    # max tokens for one src->local-expert pair
    # Timing breakdowns.
    baseline_comm_ms=None,
    baseline_expert_ms=None,
    baseline_e2e_ms=None,
    custom_no_overlap_comm_ms=None,
    custom_no_overlap_reorder_ms=None,
    custom_no_overlap_expert_ms=None,
    custom_no_overlap_e2e_ms=None,
    custom_metadata_ms=None,
    custom_overlapped_ms=None,
    custom_overlap_e2e_ms=None,
    custom_backward_forward_ms=None,
    custom_backward_backward_ms=None,
    custom_backward_e2e_ms=None,
    status: str,
):
    if rank != 0:
        return

    experts_total = world_size * e_local
    no_overlap_speedup = _safe_ratio(
        None if baseline_e2e_ms is None else baseline_e2e_ms[0],
        None if custom_no_overlap_e2e_ms is None else custom_no_overlap_e2e_ms[0],
    )
    overlap_speedup = _safe_ratio(
        None if baseline_e2e_ms is None else baseline_e2e_ms[0],
        None if custom_overlap_e2e_ms is None else custom_overlap_e2e_ms[0],
    )

    print(
        "RESULT_V2,"
        f"{world_size},{experts_total},{e_local},{topk},{batch},{seq},"
        f"{hidden_dim},{expert_hidden_dim if expert_hidden_dim is not None else ''},{capacity},{seed},{mode},"
        f"{_fmt(tpe_mean)},{_fmt(tpe_max)},{_fmt(tpe_min)},{_fmt(src_expert_max)},"
        f"{_fmt_stat(baseline_comm_ms, 0)},{_fmt_stat(baseline_comm_ms, 1)},{_fmt_stat(baseline_comm_ms, 2)},"
        f"{_fmt_stat(baseline_expert_ms, 0)},{_fmt_stat(baseline_expert_ms, 1)},{_fmt_stat(baseline_expert_ms, 2)},"
        f"{_fmt_stat(baseline_e2e_ms, 0)},{_fmt_stat(baseline_e2e_ms, 1)},{_fmt_stat(baseline_e2e_ms, 2)},"
        f"{_fmt_stat(custom_no_overlap_comm_ms, 0)},{_fmt_stat(custom_no_overlap_comm_ms, 1)},{_fmt_stat(custom_no_overlap_comm_ms, 2)},"
        f"{_fmt_stat(custom_no_overlap_reorder_ms, 0)},{_fmt_stat(custom_no_overlap_reorder_ms, 1)},{_fmt_stat(custom_no_overlap_reorder_ms, 2)},"
        f"{_fmt_stat(custom_no_overlap_expert_ms, 0)},{_fmt_stat(custom_no_overlap_expert_ms, 1)},{_fmt_stat(custom_no_overlap_expert_ms, 2)},"
        f"{_fmt_stat(custom_no_overlap_e2e_ms, 0)},{_fmt_stat(custom_no_overlap_e2e_ms, 1)},{_fmt_stat(custom_no_overlap_e2e_ms, 2)},"
        f"{_fmt_stat(custom_metadata_ms, 0)},{_fmt_stat(custom_metadata_ms, 1)},{_fmt_stat(custom_metadata_ms, 2)},"
        f"{_fmt_stat(custom_overlapped_ms, 0)},{_fmt_stat(custom_overlapped_ms, 1)},{_fmt_stat(custom_overlapped_ms, 2)},"
        f"{_fmt_stat(custom_overlap_e2e_ms, 0)},{_fmt_stat(custom_overlap_e2e_ms, 1)},{_fmt_stat(custom_overlap_e2e_ms, 2)},"
        f"{_fmt_stat(custom_backward_forward_ms, 0)},{_fmt_stat(custom_backward_forward_ms, 1)},{_fmt_stat(custom_backward_forward_ms, 2)},"
        f"{_fmt_stat(custom_backward_backward_ms, 0)},{_fmt_stat(custom_backward_backward_ms, 1)},{_fmt_stat(custom_backward_backward_ms, 2)},"
        f"{_fmt_stat(custom_backward_e2e_ms, 0)},{_fmt_stat(custom_backward_e2e_ms, 1)},{_fmt_stat(custom_backward_e2e_ms, 2)},"
        f"{_fmt(no_overlap_speedup)},{_fmt(overlap_speedup)},"
        f"{status}",
        flush=True,
    )


def _as_cuda_i64_ptr_tensor(ptrs, device):
    return torch.tensor(list(ptrs), dtype=torch.int64, device=device)

def _global_token_stats(local_expert_cnt: torch.Tensor, recv_for_me: torch.Tensor):
    """
    local_expert_cnt: [E_local], actual total tokens handled by each local expert on this rank.
    recv_for_me:      [world_size, E_local], tokens from each src rank to each local expert.

    Returns:
        tpe_mean:       global mean tokens per expert across all ranks
        tpe_max:        global max tokens per expert across all ranks
        tpe_min:        global min tokens per expert across all ranks
        src_expert_max: global max tokens for a single src->expert pair
    """
    vals = local_expert_cnt.to(torch.float32)

    local_sum = vals.sum()
    local_count = torch.tensor([vals.numel()], dtype=torch.float32, device=vals.device)
    local_max = vals.max()
    local_min = vals.min()

    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(local_min, op=dist.ReduceOp.MIN)

    tpe_mean = (local_sum / local_count).item()
    tpe_max = local_max.item()
    tpe_min = local_min.item()

    src_expert_max = recv_for_me.to(torch.float32).max()
    dist.all_reduce(src_expert_max, op=dist.ReduceOp.MAX)

    return tpe_mean, tpe_max, tpe_min, src_expert_max.item()


def _percentile_from_sorted(vals, pct: float) -> float:
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int((len(vals) - 1) * pct + 0.999999)))
    return float(vals[idx])


def _max_rank_p50_p90_p99(times_list, device):
    if not times_list:
        return None
    vals = sorted(float(x) for x in times_list)
    stats = torch.tensor(
        [
            _percentile_from_sorted(vals, 0.50),
            _percentile_from_sorted(vals, 0.90),
            _percentile_from_sorted(vals, 0.99),
        ],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return tuple(float(x) for x in stats.tolist())


def _build_gather_to_src_major_plan(recv_counts: torch.Tensor):
    """
    gather_a2a output layout is expert-major/src-major.
    baseline_expert_sequential expects src-major/expert-minor.
    """
    world_size, e_local = recv_counts.shape

    gather_off = {}
    running = 0
    for e in range(e_local):
        for src in range(world_size):
            c = int(recv_counts[src, e].item())
            gather_off[(src, e)] = running
            running += c

    out_off = 0
    plan = []
    for src in range(world_size):
        for e in range(e_local):
            c = int(recv_counts[src, e].item())
            go = gather_off[(src, e)]
            plan.append((out_off, go, c))
            out_off += c

    return plan


def _reorder_gathered_to_src_major(gathered: torch.Tensor, plan, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    reordered = torch.empty_like(gathered) if out is None else out
    for dst_start, src_start, count in plan:
        if count > 0:
            reordered[dst_start:dst_start + count] = gathered[src_start:src_start + count]
    return reordered


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
            _ = baseline_expert_sequential(recv_flat_b, recv_counts_b, weights)

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
    capacity_env: str,
    capacity_pad: float,
    capacity_align: int,
    auto_capacity: bool,
    seed: int,
):
    capacity = None if auto_capacity else int(capacity_env)
    if capacity_align <= 0:
        raise ValueError(f"CAPACITY_ALIGN must be positive, got {capacity_align}")
    if capacity_pad < 1.0:
        raise ValueError(f"CAPACITY_PAD must be >= 1.0, got {capacity_pad}")

    if MODE not in ("all", "baseline", "custom_no_overlap", "custom_overlap", "custom_backward"):
        raise ValueError(
            f"MODE must be one of all/baseline/custom_no_overlap/custom_overlap/custom_backward, got {MODE!r}"
        )

    torch.cuda.set_device(rank)
    _set_grid_print_active(False)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
        timeout=datetime.timedelta(seconds=NCCL_TIMEOUT_S),
    )
    import torch.distributed._symmetric_memory as symmem
    symmem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    _emit_header_line(rank)

    run_baseline = MODE in ("all", "baseline")
    run_no_overlap = MODE in ("all", "custom_no_overlap")
    run_overlap = MODE in ("all", "custom_overlap")
    run_backward = RUN_BACKWARD and MODE in ("all", "custom_no_overlap", "custom_backward")

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

    # gather path uses packed output directly, so local expert count is known here
    local_expert_cnt = recv_for_me.sum(dim=0).to(torch.int32).contiguous()
    total_recv = int(local_expert_cnt.sum().item())
    gather_to_src_major_plan = _build_gather_to_src_major_plan(recv_for_me)

    # Real measured y-axis for the heatmap-style analysis.
    # Do this outside timed region; .item() here is fine.
    tpe_mean, tpe_max, tpe_min, src_expert_max = _global_token_stats(
        local_expert_cnt=local_expert_cnt,
        recv_for_me=recv_for_me,
    )
    tpe_max_i = int(tpe_max)

    if auto_capacity:
        capacity = math.ceil(tpe_max_i * capacity_pad / capacity_align) * capacity_align
        if rank == 0:
            print(
                f"[AUTO_CAPACITY] tpe_max={tpe_max_i} pad={capacity_pad} "
                f"align={capacity_align} -> CAPACITY={capacity}",
                flush=True,
            )
    elif tpe_max_i > capacity and SKIP_ON_OVERFLOW:
        if rank == 0:
            print(f"[SKIP] tpe_max={tpe_max_i} > CAPACITY={capacity}. "
                  f"Increase CAPACITY or reduce (BATCH, SEQ, TOPK), or set SKIP_ON_OVERFLOW=0.")
        dist.destroy_process_group()
        return

    # Some backward kernels read CAPACITY from the process environment.
    # Keep it numeric after AUTO resolution so forward/backward agree.
    os.environ["CAPACITY"] = str(capacity)

    #max tile and workplace for overlap
    block_m = 64
    max_cnt_for_me_t = recv_for_me.max().to(torch.int64)
    dist.all_reduce(max_cnt_for_me_t, op=dist.ReduceOp.MAX)
    max_cnt_global = max(int(max_cnt_for_me_t.item()), 1)
    max_tiles = triton.cdiv(max_cnt_global, block_m)

    overlap_ws = None
    if run_overlap:
        overlap_ws = alloc_gather_overlap_workspace(
            world_size=world_size,
            e_local=e_local,
            total_recv=total_recv,
            hidden_dim=hidden_dim,
            token_dtype=send_payload.dtype,
            device=device,
            max_tiles=max_tiles,
            block_m=block_m,
        )
        dist.barrier()  # Wait for all ranks to finish overlap workspace allocation.
    # ------------------------------------------------------------------
    # 2. Allocate buffers
    # ------------------------------------------------------------------
    need_custom_buffers = (
        run_no_overlap
        or run_overlap
        or run_backward
        or PROFILE_CUSTOM
        or (CHECK and (run_no_overlap or run_overlap))
    )
    symm = None
    if need_custom_buffers:
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
        dist.barrier()  # Wait for all ranks to finish symmetric memory allocation.
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
    # This is the GEMM-side x-axis in the professor's heatmap.
    expert_hidden_dim = int(EXPERT_HIDDEN_ENV) if EXPERT_HIDDEN_ENV else hidden_dim * 4

    weights = torch.randn(
        (e_local, hidden_dim, expert_hidden_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    if CROSS_LAYER_BACKWARD:
        expert_offsets = torch.zeros_like(local_expert_cnt, dtype=torch.int32, device=device)
        if e_local > 1:
            expert_offsets[1:] = local_expert_cnt.cumsum(dim=0)[:-1].to(torch.int32)

        X_L = torch.randn(
            (total_recv, hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        X_L1 = torch.randn(
            (total_recv, hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        W_L = torch.randn(
            (e_local, hidden_dim, expert_hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        W_L1 = torch.randn(
            (e_local, hidden_dim, hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        dY_L = torch.randn(
            (total_recv, expert_hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        )

        pair = CrossLayerExpertPair(expert, expert)

        for _ in range(WARMUP):
            pair.backward_serial(
                dY_L,
                X_L,
                X_L1,
                W_L,
                W_L1,
                expert_offsets,
                expert_offsets,
                local_expert_cnt,
                local_expert_cnt,
            )
        torch.cuda.synchronize()

        times_serial = []
        for _ in range(ITERS):
            dist.barrier()
            t0 = time.perf_counter()
            pair.backward_serial(
                dY_L,
                X_L,
                X_L1,
                W_L,
                W_L1,
                expert_offsets,
                expert_offsets,
                local_expert_cnt,
                local_expert_cnt,
            )
            torch.cuda.synchronize()
            times_serial.append((time.perf_counter() - t0) * 1000.0)

        for _ in range(WARMUP):
            pair.backward_pipeline(
                dY_L,
                X_L,
                X_L1,
                W_L,
                W_L1,
                expert_offsets,
                expert_offsets,
                local_expert_cnt,
                local_expert_cnt,
            )
        torch.cuda.synchronize()

        times_pipeline = []
        for _ in range(ITERS):
            dist.barrier()
            t0 = time.perf_counter()
            pair.backward_pipeline(
                dY_L,
                X_L,
                X_L1,
                W_L,
                W_L1,
                expert_offsets,
                expert_offsets,
                local_expert_cnt,
                local_expert_cnt,
            )
            times_pipeline.append((time.perf_counter() - t0) * 1000.0)

        serial_stats = _max_rank_p50_p90_p99(times_serial, device)
        pipeline_stats = _max_rank_p50_p90_p99(times_pipeline, device)
        if rank == 0:
            serial_p50 = serial_stats[0] if serial_stats is not None else 0.0
            pipeline_p50 = pipeline_stats[0] if pipeline_stats is not None else 0.0
            speedup = serial_p50 / pipeline_p50 if pipeline_p50 > 0 else 0.0
            print(
                f"[CROSS_LAYER] serial_p50={serial_p50:.3f}ms  "
                f"pipeline_p50={pipeline_p50:.3f}ms  speedup={speedup:.3f}x",
                flush=True,
            )
        dist.destroy_process_group()
        return

     #new 4 overlap
    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    metadata_ready = torch.cuda.Event()
    producer_fully_done = torch.cuda.Event()
    compute_done = torch.cuda.Event()

    # ------------------------------------------------------------------
    # 3. Warmup - custom gather a2a + expert
    # ------------------------------------------------------------------
    if run_overlap:
        for it in range(WARMUP):
            print(
                f"[rank{rank}] iter={it} BEFORE reset_gather_overlap_workspace",
                flush=True,
            )
            with torch.cuda.stream(comm_stream):
                reset_gather_overlap_workspace(overlap_ws)
            print(
                f"[rank{rank}] iter={it} AFTER reset_gather_overlap_workspace",
                flush=True,
            )

            with torch.cuda.stream(comm_stream):
                print(
                    f"[rank{rank}] iter={it} BEFORE gather_a2a_prepare_metadata_overlap",
                    flush=True,
                )
                state = gather_a2a_prepare_metadata_overlap(
                    symm["cnts"],
                    symm["offsets"],
                    symm["tokens"],
                    symm["cnts_bases"],
                    symm["offsets_bases"],
                    e_local,
                    workspace=overlap_ws,
                    comm_stream=comm_stream,
                )
                print(
                    f"[rank{rank}] iter={it} AFTER gather_a2a_prepare_metadata_overlap",
                    flush=True,
                )

                metadata_ready.record(comm_stream)

                print(
                    f"[rank{rank}] iter={it} BEFORE gather_a2a_launch_tiles_overlap",
                    flush=True,
                )
                gather_a2a_launch_tiles_overlap(
                    symm["tokens_bases"],
                    state,
                    comm_stream=comm_stream,
                )
                print(
                    f"[rank{rank}] iter={it} AFTER gather_a2a_launch_tiles_overlap",
                    flush=True,
                )
                producer_fully_done.record(comm_stream)

            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(metadata_ready)
                if DIAG_FORCE_FULL_SYNC:
                    compute_stream.wait_event(producer_fully_done)
                if DIAG_HEADSTART_US > 0:
                    torch.cuda._sleep(DIAG_HEADSTART_US)
                print(
                    f"[rank{rank}] iter={it} BEFORE expert_overlap",
                    flush=True,
                )
                _ = expert_overlap(state, weights, num_experts_total)
                print(
                    f"[rank{rank}] iter={it} AFTER expert_overlap",
                    flush=True,
                )
                compute_done.record(compute_stream)

            print(
                f"[rank{rank}] iter={it} BEFORE compute_done.synchronize",
                flush=True,
            )
            compute_done.synchronize()
            print(
                f"[rank{rank}] iter={it} AFTER compute_done.synchronize",
                flush=True,
            )

    if run_no_overlap:
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
            gathered_reordered = _reorder_gathered_to_src_major(gathered, gather_to_src_major_plan)
            _ = baseline_expert_sequential(gathered_reordered, recv_for_me, weights)
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 4. Warmup - baseline a2a + expert
    # ------------------------------------------------------------------
    ok = torch.tensor([1], device=device, dtype=torch.int32)
    warmup_err = None
    if run_baseline:
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
                _ = baseline_expert_sequential(recv_flat_b, recv_counts_b, weights)
                torch.cuda.synchronize()
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
            expert_hidden_dim=expert_hidden_dim,
            topk=topk,
            capacity=capacity,
            seed=seed,
            mode=MODE,
            tpe_mean=tpe_mean,
            tpe_max=tpe_max,
            tpe_min=tpe_min,
            src_expert_max=src_expert_max,
            status="baseline_warmup_error",
        )
        dist.destroy_process_group()
        return

    dist.barrier()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 5. Timed - custom gather a2a + expert
    # ------------------------------------------------------------------
    custom_metadata_ms = []
    custom_overlapped_ms = []
    custom_overlap_e2e_ms = []

    t0_custom = torch.cuda.Event(enable_timing=True)
    t_comm_custom = torch.cuda.Event(enable_timing=True)
    t_exp_custom = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    dist.barrier()

    if run_overlap:
        overlap_probe_dumped = False
        for _ in range(ITERS):
            with torch.cuda.stream(comm_stream):
                reset_gather_overlap_workspace(overlap_ws)

            t0_custom.record()

            with torch.cuda.stream(comm_stream):
                state = gather_a2a_prepare_metadata_overlap(
                    symm["cnts"],
                    symm["offsets"],
                    symm["tokens"],
                    symm["cnts_bases"],
                    symm["offsets_bases"],
                    e_local,
                    workspace=overlap_ws,
                    comm_stream=comm_stream,
                )

                metadata_ready.record(comm_stream)
                t_comm_custom.record(comm_stream)

                gather_a2a_launch_tiles_overlap(
                    symm["tokens_bases"],
                    state,
                    comm_stream=comm_stream,
                )
                producer_fully_done.record(comm_stream)

            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(metadata_ready)
                if DIAG_FORCE_FULL_SYNC:
                    compute_stream.wait_event(producer_fully_done)
                if DIAG_HEADSTART_US > 0:
                    torch.cuda._sleep(DIAG_HEADSTART_US)
                _set_grid_print_active(_ == 0)
                _ = expert_overlap(state, weights, num_experts_total)
                _set_grid_print_active(False)
                t_exp_custom.record(compute_stream)
                compute_done.record(compute_stream)

            compute_done.synchronize()
            if OVERLAP_SPIN_PROBE and not overlap_probe_dumped:
                probe_label = OVERLAP_PROBE_LABEL or (
                    f"E{world_size * e_local}_EL{e_local}_K{topk}_B{batch}_S{seq}_H{hidden_dim}"
                )
                _dump_overlap_probe(
                    rank=rank,
                    label=probe_label,
                    state=state,
                    world_size=world_size,
                    e_local=e_local,
                )
                overlap_probe_dumped = True
            if OVERLAP_SPIN_PROBE and overlap_probe_dumped:
                break
            custom_metadata_ms.append(t0_custom.elapsed_time(t_comm_custom))
            custom_overlapped_ms.append(t_comm_custom.elapsed_time(t_exp_custom))
            custom_overlap_e2e_ms.append(t0_custom.elapsed_time(t_exp_custom))

    if not OVERLAP_SPIN_PROBE:
        # ------------------------------------------------------------------
        # 5b. Timed - custom no-overlap gather + baseline expert
        # ------------------------------------------------------------------
        custom_no_overlap_comm_ms = []
        custom_no_overlap_reorder_ms = []
        custom_no_overlap_expert_ms = []
        custom_no_overlap_e2e_ms = []

        t0_no = torch.cuda.Event(enable_timing=True)
        t_comm_no = torch.cuda.Event(enable_timing=True)
        t_reorder_no = torch.cuda.Event(enable_timing=True)
        t_exp_no = torch.cuda.Event(enable_timing=True)
        no_overlap_reorder_buf = torch.empty(
            (total_recv, hidden_dim),
            dtype=send_payload.dtype,
            device=device,
        ) if run_no_overlap else None

        torch.cuda.synchronize()
        dist.barrier()

        if run_no_overlap:
            for _ in range(ITERS):
                t0_no.record()
                gathered = gather_a2a(
                    symm["cnts"],
                    symm["offsets"],
                    symm["tokens"],
                    symm["cnts_bases"],
                    symm["offsets_bases"],
                    symm["tokens_bases"],
                    e_local,
                )
                t_comm_no.record()

                gathered_reordered = _reorder_gathered_to_src_major(
                    gathered,
                    gather_to_src_major_plan,
                    no_overlap_reorder_buf,
                )
                t_reorder_no.record()
                _ = baseline_expert_sequential(gathered_reordered, recv_for_me, weights)
                t_exp_no.record()
                t_exp_no.synchronize()

                custom_no_overlap_comm_ms.append(t0_no.elapsed_time(t_comm_no))
                custom_no_overlap_reorder_ms.append(t_comm_no.elapsed_time(t_reorder_no))
                custom_no_overlap_expert_ms.append(t_reorder_no.elapsed_time(t_exp_no))
                custom_no_overlap_e2e_ms.append(t0_no.elapsed_time(t_exp_no))

        # ------------------------------------------------------------------
        # 6. Timed - baseline a2a + expert
        # ------------------------------------------------------------------
        base_comm_ms = []
        base_expert_ms = []
        base_e2e_ms = []
        timed_err = None
        ok.fill_(1)

        t0_base = torch.cuda.Event(enable_timing=True)
        t_comm_base = torch.cuda.Event(enable_timing=True)
        t_exp_base = torch.cuda.Event(enable_timing=True)
    
        # same stuff as custom
        torch.cuda.synchronize()
        dist.barrier()
        if run_baseline:
            for _ in range(ITERS):
                #torch.cuda.synchronize()
                #dist.barrier()

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

                    _ = baseline_expert_sequential(recv_flat_b, recv_counts_b, weights)
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
                expert_hidden_dim=expert_hidden_dim,
                topk=topk,
                capacity=capacity,
                seed=seed,
                mode=MODE,
                tpe_mean=tpe_mean,
                tpe_max=tpe_max,
                tpe_min=tpe_min,
                src_expert_max=src_expert_max,
                status="baseline_timed_error",
            )
            dist.destroy_process_group()
            return

        # ------------------------------------------------------------------
        # 6b. Timed - custom autograd forward + backward
        # ------------------------------------------------------------------
        custom_backward_forward_ms = []
        custom_backward_backward_ms = []
        custom_backward_e2e_ms = []

        if run_backward:
            symm["tokens"].requires_grad_(True)
            bw_weights = weights.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            dist.barrier()

            for _ in range(WARMUP):
                symm["tokens"].grad = None
                bw_weights.grad = None
                if USE_FUSED_GATHER_EXPERT and MODE == "custom_backward":
                    out = gather_expert(
                        symm["cnts"],
                        symm["offsets"],
                        symm["tokens"],
                        symm["cnts_bases"],
                        symm["offsets_bases"],
                        symm["tokens_bases"],
                        e_local,
                        bw_weights,
                        local_expert_cnt,
                        recv_for_me,
                    )
                    if DEBUG_BACKWARD_STAGES:
                        print(
                            f"[DEBUG_BACKWARD_STAGES][rank{rank}] "
                            "after fused gather+expert forward (warmup)",
                            flush=True,
                        )
                        torch.cuda.synchronize()
                else:
                    gathered = gather_a2a(
                        symm["cnts"],
                        symm["offsets"],
                        symm["tokens"],
                        symm["cnts_bases"],
                        symm["offsets_bases"],
                        symm["tokens_bases"],
                        e_local,
                    )
                    if DEBUG_BACKWARD_STAGES:
                        print(f"[DEBUG_BACKWARD_STAGES][rank{rank}] after gather_a2a forward (warmup)", flush=True)
                        torch.cuda.synchronize()
                    out = expert(gathered, bw_weights, local_expert_cnt, recv_for_me)
                    if DEBUG_BACKWARD_STAGES:
                        print(f"[DEBUG_BACKWARD_STAGES][rank{rank}] after expert forward (warmup)", flush=True)
                        torch.cuda.synchronize()
                loss = out.float().sum()
                if DEBUG_BACKWARD_STAGES:
                    print(f"[DEBUG_BACKWARD_STAGES][rank{rank}] immediately before loss.backward() (warmup)", flush=True)
                    torch.cuda.synchronize()
                loss.backward()
                torch.cuda.synchronize()

            t0_bw = torch.cuda.Event(enable_timing=True)
            t_fwd_bw = torch.cuda.Event(enable_timing=True)
            t_bwd_bw = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            dist.barrier()

            for _ in range(ITERS):
                symm["tokens"].grad = None
                bw_weights.grad = None

                t0_bw.record()
                if USE_FUSED_GATHER_EXPERT and MODE == "custom_backward":
                    out = gather_expert(
                        symm["cnts"],
                        symm["offsets"],
                        symm["tokens"],
                        symm["cnts_bases"],
                        symm["offsets_bases"],
                        symm["tokens_bases"],
                        e_local,
                        bw_weights,
                        local_expert_cnt,
                        recv_for_me,
                    )
                    if DEBUG_BACKWARD_STAGES:
                        print(
                            f"[DEBUG_BACKWARD_STAGES][rank{rank}] "
                            "after fused gather+expert forward (timed)",
                            flush=True,
                        )
                        torch.cuda.synchronize()
                else:
                    gathered = gather_a2a(
                        symm["cnts"],
                        symm["offsets"],
                        symm["tokens"],
                        symm["cnts_bases"],
                        symm["offsets_bases"],
                        symm["tokens_bases"],
                        e_local,
                    )
                    if DEBUG_BACKWARD_STAGES:
                        print(f"[DEBUG_BACKWARD_STAGES][rank{rank}] after gather_a2a forward (timed)", flush=True)
                        torch.cuda.synchronize()
                    out = expert(gathered, bw_weights, local_expert_cnt, recv_for_me)
                    if DEBUG_BACKWARD_STAGES:
                        print(f"[DEBUG_BACKWARD_STAGES][rank{rank}] after expert forward (timed)", flush=True)
                        torch.cuda.synchronize()
                loss = out.float().sum()
                t_fwd_bw.record()

                if DEBUG_BACKWARD_STAGES:
                    print(f"[DEBUG_BACKWARD_STAGES][rank{rank}] immediately before loss.backward() (timed)", flush=True)
                    torch.cuda.synchronize()
                _set_grid_print_active(_ == 0)
                loss.backward()
                _set_grid_print_active(False)
                t_bwd_bw.record()
                t_bwd_bw.synchronize()

                custom_backward_forward_ms.append(t0_bw.elapsed_time(t_fwd_bw))
                custom_backward_backward_ms.append(t_fwd_bw.elapsed_time(t_bwd_bw))
                custom_backward_e2e_ms.append(t0_bw.elapsed_time(t_bwd_bw))

        # ------------------------------------------------------------------
        # 7. Report perf
        # ------------------------------------------------------------------
        custom_metadata_stats = _max_rank_p50_p90_p99(custom_metadata_ms, device)
        custom_overlapped_stats = _max_rank_p50_p90_p99(custom_overlapped_ms, device)
        custom_overlap_e2e_stats = _max_rank_p50_p90_p99(custom_overlap_e2e_ms, device)

        custom_no_overlap_comm_stats = _max_rank_p50_p90_p99(custom_no_overlap_comm_ms, device)
        custom_no_overlap_reorder_stats = _max_rank_p50_p90_p99(custom_no_overlap_reorder_ms, device)
        custom_no_overlap_expert_stats = _max_rank_p50_p90_p99(custom_no_overlap_expert_ms, device)
        custom_no_overlap_e2e_stats = _max_rank_p50_p90_p99(custom_no_overlap_e2e_ms, device)
        custom_backward_forward_stats = _max_rank_p50_p90_p99(custom_backward_forward_ms, device)
        custom_backward_backward_stats = _max_rank_p50_p90_p99(custom_backward_backward_ms, device)
        custom_backward_e2e_stats = _max_rank_p50_p90_p99(custom_backward_e2e_ms, device)

        b_comm_stats = _max_rank_p50_p90_p99(base_comm_ms, device)
        b_exp_stats = _max_rank_p50_p90_p99(base_expert_ms, device)
        b_e2e_stats = _max_rank_p50_p90_p99(base_e2e_ms, device)

        if rank == 0:
            print("\n" + "=" * 50)
            print(" GATHER DIAGNOSTIC BREAKDOWN (max-rank p50/p90/p99 over iters) ")
            print("=" * 50)
            print(
                f"World: {world_size} | E_local: {e_local} | "
                f"Cap: {capacity} | H: {hidden_dim} | Expert_H: {expert_hidden_dim} | "
                f"Batch: {batch} | Seq: {seq} | TopK: {topk} | Mode: {MODE}"
            )
            print(
                f"Actual tokens/expert: mean={tpe_mean:.1f}, "
                f"max={tpe_max:.1f}, min={tpe_min:.1f}, "
                f"max src->expert={src_expert_max:.1f}"
            )
            print("Note: custom_overlap metadata/overlapped breakdown is not comparable to baseline breakdown; E2E is primary.")
            print("-" * 50)
            if b_e2e_stats is not None:
                print("[Baseline (PyTorch)]")
                print(f"  Comm p50/p90/p99:    {b_comm_stats[0]:.3f}/{b_comm_stats[1]:.3f}/{b_comm_stats[2]:.3f} ms")
                print(f"  Expert p50/p90/p99:  {b_exp_stats[0]:.3f}/{b_exp_stats[1]:.3f}/{b_exp_stats[2]:.3f} ms")
                print(f"  E2E p50/p90/p99:     {b_e2e_stats[0]:.3f}/{b_e2e_stats[1]:.3f}/{b_e2e_stats[2]:.3f} ms")
                print("-" * 50)
            if custom_no_overlap_e2e_stats is not None:
                print("[Custom no-overlap]")
                print(f"  Comm p50/p90/p99:    {custom_no_overlap_comm_stats[0]:.3f}/{custom_no_overlap_comm_stats[1]:.3f}/{custom_no_overlap_comm_stats[2]:.3f} ms")
                print(f"  Reorder p50/p90/p99: {custom_no_overlap_reorder_stats[0]:.3f}/{custom_no_overlap_reorder_stats[1]:.3f}/{custom_no_overlap_reorder_stats[2]:.3f} ms")
                print(f"  Expert p50/p90/p99:  {custom_no_overlap_expert_stats[0]:.3f}/{custom_no_overlap_expert_stats[1]:.3f}/{custom_no_overlap_expert_stats[2]:.3f} ms")
                print(f"  E2E p50/p90/p99:     {custom_no_overlap_e2e_stats[0]:.3f}/{custom_no_overlap_e2e_stats[1]:.3f}/{custom_no_overlap_e2e_stats[2]:.3f} ms")
                print("-" * 50)
            if custom_overlap_e2e_stats is not None:
                print("[Custom overlap]")
                print(f"  Metadata p50/p90/p99:   {custom_metadata_stats[0]:.3f}/{custom_metadata_stats[1]:.3f}/{custom_metadata_stats[2]:.3f} ms")
                print(f"  Overlapped p50/p90/p99: {custom_overlapped_stats[0]:.3f}/{custom_overlapped_stats[1]:.3f}/{custom_overlapped_stats[2]:.3f} ms")
                print(f"  E2E p50/p90/p99:        {custom_overlap_e2e_stats[0]:.3f}/{custom_overlap_e2e_stats[1]:.3f}/{custom_overlap_e2e_stats[2]:.3f} ms")
                print("-" * 50)
            if custom_backward_e2e_stats is not None:
                print("[Custom backward autograd]")
                print(f"  Forward p50/p90/p99:  {custom_backward_forward_stats[0]:.3f}/{custom_backward_forward_stats[1]:.3f}/{custom_backward_forward_stats[2]:.3f} ms")
                print(f"  Backward p50/p90/p99: {custom_backward_backward_stats[0]:.3f}/{custom_backward_backward_stats[1]:.3f}/{custom_backward_backward_stats[2]:.3f} ms")
                print(f"  E2E p50/p90/p99:      {custom_backward_e2e_stats[0]:.3f}/{custom_backward_e2e_stats[1]:.3f}/{custom_backward_e2e_stats[2]:.3f} ms")
            print("=" * 50)
            if b_e2e_stats is not None and custom_no_overlap_e2e_stats is not None and custom_no_overlap_e2e_stats[0] > 0:
                print(f"Speedup (Baseline/Custom no-overlap E2E p50): {b_e2e_stats[0] / custom_no_overlap_e2e_stats[0]:.3f}x")
            if b_e2e_stats is not None and custom_overlap_e2e_stats is not None and custom_overlap_e2e_stats[0] > 0:
                print(f"Speedup (Baseline/Custom overlap E2E p50): {b_e2e_stats[0] / custom_overlap_e2e_stats[0]:.3f}x")
            print("=" * 50 + "\n")
        _emit_result_line(
            rank,
            world_size=world_size,
            e_local=e_local,
            batch=batch,
            seq=seq,
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            topk=topk,
            capacity=capacity,
            seed=seed,
            mode=MODE,
            tpe_mean=tpe_mean,
            tpe_max=tpe_max,
            tpe_min=tpe_min,
            src_expert_max=src_expert_max,
            baseline_comm_ms=b_comm_stats,
            baseline_expert_ms=b_exp_stats,
            baseline_e2e_ms=b_e2e_stats,
            custom_no_overlap_comm_ms=custom_no_overlap_comm_stats,
            custom_no_overlap_reorder_ms=custom_no_overlap_reorder_stats,
            custom_no_overlap_expert_ms=custom_no_overlap_expert_stats,
            custom_no_overlap_e2e_ms=custom_no_overlap_e2e_stats,
            custom_metadata_ms=custom_metadata_stats,
            custom_overlapped_ms=custom_overlapped_stats,
            custom_overlap_e2e_ms=custom_overlap_e2e_stats,
            custom_backward_forward_ms=custom_backward_forward_stats,
            custom_backward_backward_ms=custom_backward_backward_stats,
            custom_backward_e2e_ms=custom_backward_e2e_stats,
            status="ok",
        )
        dist.barrier()
        torch.cuda.synchronize()

        if PROFILE_CUSTOM and need_custom_buffers:
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
        if CHECK and (run_no_overlap or run_overlap):
            dist.barrier()
            torch.cuda.synchronize()

            if (not auto_capacity) and tpe_max_i > capacity and SKIP_ON_OVERFLOW:
                if rank == 0:
                    print(f"[SKIP] tpe_max={tpe_max_i} > CAPACITY={capacity}. "
                          f"Increase CAPACITY or reduce (BATCH, SEQ, TOPK), or set SKIP_ON_OVERFLOW=0.")
                _emit_result_line(
                    rank,
                    world_size=world_size,
                    e_local=e_local,
                    batch=batch,
                    seq=seq,
                    hidden_dim=hidden_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    topk=topk,
                    capacity=capacity,
                    seed=seed,
                    mode=MODE,
                    tpe_mean=tpe_mean,
                    tpe_max=tpe_max,
                    tpe_min=tpe_min,
                    src_expert_max=src_expert_max,
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
    capacity_env = os.environ.get("CAPACITY", "AUTO")
    capacity_pad = float(os.environ.get("CAPACITY_PAD", "1.10"))
    capacity_align = int(os.environ.get("CAPACITY_ALIGN", "64"))
    auto_capacity = (capacity_env.upper() == "AUTO")
    if not auto_capacity:
        int(capacity_env)  # validate before spawning ranks
    seed       = int(os.getenv("SEED", "42"))

    if not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        bench_gather,
        args=(world_size, batch, seq, hidden_dim, topk, e_local, capacity_env, capacity_pad, capacity_align, auto_capacity, seed),
        nprocs=world_size,
        join=True,
    )
