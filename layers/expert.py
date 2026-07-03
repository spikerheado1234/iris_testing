import os

import torch
import torch.distributed as dist
import triton

from kernels import grouped_gemm_rect, grouped_gemm_overlap
from backward_kernels import (
    grouped_gemm_dx,
    grouped_gemm_dw_fused,
    grouped_gemm_dx_tiled,
    grouped_gemm_dw_overlap,
    grouped_gemm_dw_partial,
    grouped_gemm_dw_reduce,
)


def _grid_num_programs(grid):
    if isinstance(grid, tuple):
        total = 1
        for dim in grid:
            total *= int(dim)
        return total
    return int(grid)


_GRID_PRINTED = set()


def _print_grid(kernel_name, grid):
    if os.getenv("PRINT_GRID", "0") != "1":
        return
    if os.getenv("PRINT_GRID_ACTIVE", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    if kernel_name in _GRID_PRINTED:
        return
    _GRID_PRINTED.add(kernel_name)
    grid_size = _grid_num_programs(grid)
    cu_count = 304
    print(
        f"[GRID] kernel={kernel_name} grid={grid_size} "
        f"cu_count={cu_count} grid_per_cu={grid_size / cu_count:.2f}",
        flush=True,
    )


class Expert(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tokens,              # [S, H]
        weights,             # [E, H, N]
        local_expert_cnt,    # [E]
        aggregrate_exp_cnt,  # unused for now
    ):
        device = tokens.device
        E = weights.size(0)
        H = weights.size(1)
        N = weights.size(2)
        S = tokens.size(0)

        # packed offsets for expert-major token layout
        expert_offsets = torch.zeros_like(local_expert_cnt, dtype=torch.int32, device=device)
        if E > 1:
            expert_offsets[1:] = local_expert_cnt.cumsum(dim=0)[:-1].to(torch.int32)

        ctx.save_for_backward(tokens, weights, local_expert_cnt, expert_offsets)
        ctx.local_expert_cnts = aggregrate_exp_cnt if torch.is_tensor(aggregrate_exp_cnt) else None
        ctx.E = E
        ctx.H = H
        ctx.N = N

        if tokens.numel() == 0:
            return torch.empty(
                (0, N),
                dtype=tokens.dtype,
                device=tokens.device,
            )

        if os.environ.get("GGEMM_VALIDATE", "0") == "1":
            linear_out_elems = int(S) * N
            if linear_out_elems >= (1 << 31):
                print(
                    f"[GGEMM_RECT] int32 overflow risk avoided by int64 index math: "
                    f"S={S} * N={N} = {linear_out_elems} >= 2^31",
                    flush=True,
                )

        out = torch.empty((S, N), dtype=tokens.dtype, device=device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # phase-1 no-tolist schedule:
        # rectangular launch, kernel decodes pid -> (g, tm, tn)
        max_m_tiles = triton.cdiv(S, BLOCK_M)
        n_tiles = triton.cdiv(N, BLOCK_N)
        total_tiles = E * max_m_tiles * n_tiles

        if total_tiles == 0:
            return out

        grid = (total_tiles,)

        if os.environ.get("DEBUG_GGEMM", "0") == "1":
            rank = dist.get_rank() if dist.is_initialized() else -1
            print(f"[GGEMM_RECT][rank{rank}] S={S}, E={E}, H={H}, N={N}", flush=True)
            print(f"[GGEMM_RECT][rank{rank}] max_m_tiles={max_m_tiles}, n_tiles={n_tiles}, total_tiles={total_tiles}", flush=True)
            print(f"[GGEMM_RECT][rank{rank}] local_expert_cnt={local_expert_cnt.tolist()}", flush=True)
            print(f"[GGEMM_RECT][rank{rank}] expert_offsets={expert_offsets.tolist()}", flush=True)

        if os.environ.get("GGEMM_VALIDATE", "0") == "1":
            assert int(local_expert_cnt.sum().item()) == S, (
                f"sum(local_expert_cnt)={int(local_expert_cnt.sum().item())}, S={S}"
            )
            for i in range(E):
                cnt = int(local_expert_cnt[i].item())
                off = int(expert_offsets[i].item())
                assert cnt >= 0
                assert off >= 0
                assert off + cnt <= S, f"expert {i}: off={off}, cnt={cnt}, S={S}"

        num_sm = int(os.environ.get("GGEMM_NUM_SM", "1"))

        grouped_gemm_rect[grid](
            tokens_ptr=tokens,
            weights_ptr=weights,
            out_ptr=out,
            expert_tkn_cnt=local_expert_cnt,
            expert_offsets=expert_offsets,
            expert_cnt=E,
            hidden_dim=H,
            expert_hidden_dim=N,
            max_m_tiles=max_m_tiles,
            num_n_tiles=n_tiles,
            NUM_SM=num_sm,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
        )

        return out

    @staticmethod
    def backward(ctx, do):
        debug_stages = os.environ.get("DEBUG_BACKWARD_STAGES", "0") == "1"
        rank = dist.get_rank() if dist.is_initialized() else -1

        def _stage_print(stage, message, **fields):
            if not debug_stages:
                return
            suffix = ""
            if fields:
                suffix = " " + " ".join(f"{key}={value}" for key, value in fields.items())
            print(
                f"[DEBUG_BACKWARD_STAGES][rank{rank}] {stage}: {message}{suffix}",
                flush=True,
            )

        def _stage_sync(stage, message):
            if not debug_stages:
                return
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(
                    f"[DEBUG_BACKWARD_STAGES][rank{rank}] {stage}: FAILED during synchronize: {exc}",
                    flush=True,
                )
                raise
            _stage_print(stage, message)

        _stage_print("B0", "first line of Expert.backward")

        tokens, weights, local_expert_cnt, expert_offsets = ctx.saved_tensors
        E = ctx.E
        H = ctx.H
        N = ctx.N
        S = tokens.size(0)
        capacity = int(os.environ.get("CAPACITY", str(S)))

        d_out = do.contiguous()
        d_tokens = torch.empty_like(tokens)

        debug_backward = os.environ.get("DEBUG_BACKWARD", "0") == "1"
        sync_backward = (
            os.environ.get("SYNC_BACKWARD_KERNELS", "0") == "1"
            or os.environ.get("AMD_SERIALIZE_KERNEL", "") in ("1", "3")
        )
        use_parallel_backward = (
            not debug_stages
            and not sync_backward
            and os.environ.get("DEBUG_DW_BOUNDS", "0") != "1"
        )
        use_backward_overlap = (
            not debug_stages
            and not sync_backward
            and os.environ.get("BACKWARD_OVERLAP", "0") == "1"
        )

        cnt_last = 0
        cnt_max = 0
        cnt_sum = 0
        off_last = 0
        tail = 0
        offsets_monotonic = True
        tail_ok = True
        max_cnt_ok = True
        if debug_stages:
            try:
                if local_expert_cnt.numel() > 0:
                    cnt_last = int(local_expert_cnt[-1].item())
                    cnt_max = int(local_expert_cnt.max().item())
                    cnt_sum = int(local_expert_cnt.sum().item())
                    off_last = int(expert_offsets[-1].item())
                    tail = off_last + cnt_last
                    if expert_offsets.numel() > 1:
                        offsets_monotonic = bool(
                            (expert_offsets[1:] >= expert_offsets[:-1]).all().item()
                        )
                tail_ok = tail <= S
                max_cnt_ok = cnt_max <= capacity
                _stage_print(
                    "B1",
                    "after metadata/tail/capacity sync",
                    capacity=capacity,
                    S=S,
                    expert_offsets_last=off_last,
                    local_expert_cnt_last=cnt_last,
                    tail=tail,
                    local_expert_cnt_max=cnt_max,
                    local_expert_cnt_sum=cnt_sum,
                    expert_offsets_monotonic=offsets_monotonic,
                    tail_le_S=tail_ok,
                    max_cnt_le_capacity=max_cnt_ok,
                )
            except Exception as exc:
                print(
                    f"[DEBUG_BACKWARD_STAGES][rank{rank}] B1: FAILED during metadata/tail/capacity sync: {exc}",
                    flush=True,
                )
                raise

        if debug_backward:
            local_expert_cnts = ctx.local_expert_cnts
            if local_expert_cnts is None:
                local_expert_cnts = local_expert_cnt.unsqueeze(0)
            print(
                f"[BACKWARD_DEBUG][rank{rank}] "
                f"local_expert_cnt(dtype={local_expert_cnt.dtype}, device={local_expert_cnt.device}, "
                f"shape={tuple(local_expert_cnt.shape)}, contiguous={local_expert_cnt.is_contiguous()}) "
                f"local_expert_cnts(dtype={local_expert_cnts.dtype}, device={local_expert_cnts.device}, "
                f"shape={tuple(local_expert_cnts.shape)}, contiguous={local_expert_cnts.is_contiguous()}) "
                f"expert_offsets(dtype={expert_offsets.dtype}, device={expert_offsets.device}, "
                f"shape={tuple(expert_offsets.shape)}, contiguous={expert_offsets.is_contiguous()})",
                flush=True,
            )
            if local_expert_cnt.numel() > 0:
                cnt_min = int(local_expert_cnt.min().item())
                cnt_max = int(local_expert_cnt.max().item())
                cnt_sum = int(local_expert_cnt.sum().item())
                tail = int(expert_offsets[-1].item()) + int(local_expert_cnt[-1].item())
            else:
                cnt_min = cnt_max = cnt_sum = tail = 0
            print(
                f"[BACKWARD_DEBUG][rank{rank}] "
                f"local_expert_cnt(min={cnt_min}, max={cnt_max}, sum={cnt_sum}) "
                f"expert_offsets[-1]+local_expert_cnt[-1]={tail} "
                f"gathered_tokens.shape[0]={tokens.shape[0]} "
                f"d_out.shape[0]={d_out.shape[0]} capacity={capacity}",
                flush=True,
            )

        if local_expert_cnt.numel() > 0:
            if not debug_stages:
                tail = int(expert_offsets[-1].item()) + int(local_expert_cnt[-1].item())
            assert tail <= tokens.shape[0], (
                f"expert tail {tail} exceeds tokens.shape[0]={tokens.shape[0]}"
            )
            if not debug_stages:
                cnt_max = int(local_expert_cnt.max().item())
            assert cnt_max <= capacity, (
                f"local_expert_cnt.max()={cnt_max} exceeds capacity={capacity}"
            )

        BLOCK_M = 64
        BLOCK_H = 64
        BLOCK_N = 64
        BLOCK_K = 32

        h_tiles = triton.cdiv(H, BLOCK_H)
        n_tiles = triton.cdiv(N, BLOCK_N)
        max_m_tiles = triton.cdiv(S, BLOCK_M)
        hn_tiles = h_tiles * n_tiles
        m_tiles_total = triton.cdiv(cnt_max, BLOCK_M) if cnt_max > 0 else 0
        m_tile_batch = max(1, int(os.environ.get("DW_M_TILE_BATCH", "1")))
        hn_tile_batch = max(1, int(os.environ.get("DW_HN_TILE_BATCH", "64")))
        debug_stage1_expert = int(os.environ.get("DEBUG_DW_STAGE1_EXPERT_MASK", "-1"))
        debug_stage2_expert = int(os.environ.get("DEBUG_DW_STAGE2_EXPERT_MASK", "-1"))
        debug_stage2_hn = int(os.environ.get("DEBUG_DW_STAGE2_HN_MASK", "-1"))
        skip_stage2 = os.environ.get("DEBUG_DW_SKIP_STAGE2", "0") == "1"
        stage2_no_store = os.environ.get("DEBUG_DW_STAGE2_NO_STORE", "0") == "1"
        raw_stage2_m_loop_limit = int(os.environ.get("DEBUG_DW_STAGE2_M_LOOP_LIMIT", "0"))
        stage2_m_loop_limit = (
            min(m_tile_batch, raw_stage2_m_loop_limit)
            if raw_stage2_m_loop_limit > 0
            else m_tile_batch
        )
        stage2_store_block = max(1, int(os.environ.get("DW_STAGE2_STORE_BLOCK", "256")))
        stage2_store_chunks = triton.cdiv(BLOCK_H * BLOCK_N, stage2_store_block)
        zero_partial = os.environ.get("DEBUG_DW_ZERO_PARTIAL", "1") == "1"
        if raw_stage2_m_loop_limit == 1:
            assert stage2_m_loop_limit == 1, (
                f"DEBUG_DW_STAGE2_M_LOOP_LIMIT=1 but effective loop limit is {stage2_m_loop_limit}"
            )

        if S > 0 and not use_parallel_backward:
            _stage_print(
                "B2",
                "before grouped_gemm_dx",
                capacity=capacity,
                S=S,
                expert_offsets_last=off_last,
                local_expert_cnt_last=cnt_last,
                tail=tail,
                local_expert_cnt_max=cnt_max,
                local_expert_cnt_sum=cnt_sum,
                expert_offsets_monotonic=offsets_monotonic,
                tail_le_S=tail_ok,
                max_cnt_le_capacity=max_cnt_ok,
            )
            grid_dx = (E * max_m_tiles * h_tiles,)
            _print_grid("grouped_gemm_dx", grid_dx)
            grouped_gemm_dx[grid_dx](
                d_out_ptr=d_out,
                weights_ptr=weights,
                d_tokens_ptr=d_tokens,
                expert_tkn_cnt=local_expert_cnt,
                expert_offsets=expert_offsets,
                expert_cnt=E,
                hidden_dim=H,
                expert_hidden_dim=N,
                capacity=capacity,
                max_m_tiles=max_m_tiles,
                num_h_tiles=h_tiles,
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_H=BLOCK_H,
                BLOCK_SIZE_K=BLOCK_K,
            )
            if debug_stages:
                _stage_sync("B3", "after grouped_gemm_dx launch and synchronize")
            elif sync_backward:
                torch.cuda.synchronize()

        _stage_print(
            "B4",
            "before grouped_gemm_dw",
            capacity=capacity,
            S=S,
            expert_offsets_last=off_last,
            local_expert_cnt_last=cnt_last,
            tail=tail,
            local_expert_cnt_max=cnt_max,
            local_expert_cnt_sum=cnt_sum,
            expert_offsets_monotonic=offsets_monotonic,
            tail_le_S=tail_ok,
            max_cnt_le_capacity=max_cnt_ok,
        )
        if os.environ.get("DEBUG_DW_BOUNDS", "0") == "1":
            try:
                partial_numel = E * m_tile_batch * hn_tile_batch * BLOCK_H * BLOCK_N
                partial_mb = partial_numel * 4.0 / (1024.0 * 1024.0)
                accum_numel = E * H * N
                accum_mb = accum_numel * 4.0 / (1024.0 * 1024.0)
                print(
                    f"[DEBUG_DW_BOUNDS][rank{rank}] "
                    f"max_cnt={cnt_max} m_tiles_total={m_tiles_total} "
                    f"M_TILE_BATCH={m_tile_batch} HN_TILE_BATCH={hn_tile_batch} "
                    f"H_tiles={h_tiles} EH_tiles={n_tiles} HN_tiles={hn_tiles} "
                    f"BLOCK_M={BLOCK_M} BLOCK_H={BLOCK_H} BLOCK_EH={BLOCK_N} "
                    f"partial_shape=({E},{m_tile_batch},{hn_tile_batch},{BLOCK_H},{BLOCK_N}) "
                    f"partial_numel={partial_numel} partial_estimated_mb={partial_mb:.2f} "
                    f"accum_shape=({E},{H},{N}) accum_numel={accum_numel} "
                    f"accum_estimated_mb={accum_mb:.2f} "
                    f"max_cnt_for_me={cnt_max} tail={tail} "
                    f"raw_m_loop_limit={raw_stage2_m_loop_limit} "
                    f"m_loop_limit={stage2_m_loop_limit} "
                    f"DEBUG_DW_ZERO_PARTIAL={zero_partial}",
                    flush=True,
                )
                cnts_host = local_expert_cnt.detach().cpu().tolist()
                offsets_host = expert_offsets.detach().cpu().tolist()
                print(
                    f"[DEBUG_DW_BOUNDS][rank{rank}] "
                    f"local_expert_cnt={cnts_host} expert_offsets={offsets_host}",
                    flush=True,
                )
                for expert_id, (cnt_i, token_base_i) in enumerate(zip(cnts_host, offsets_host)):
                    cnt_i = int(cnt_i)
                    token_base_i = int(token_base_i)
                    effective_cnt_i = min(cnt_i, capacity)
                    token_tail_i = token_base_i + cnt_i
                    effective_tail_i = token_base_i + effective_cnt_i
                    print(
                        f"[DEBUG_DW_BOUNDS][rank{rank}] "
                        f"S={S} capacity={capacity} expert_id={expert_id} "
                        f"cnt={cnt_i} effective_cnt={effective_cnt_i} "
                        f"token_base={token_base_i} token_base_plus_cnt={token_tail_i} "
                        f"token_base_plus_effective_cnt={effective_tail_i} "
                        f"x_shape={tuple(tokens.shape)} d_out_shape={tuple(d_out.shape)} "
                        f"d_weights_shape={tuple(weights.shape)} "
                        f"tail_le_S={token_tail_i <= S} effective_tail_le_S={effective_tail_i <= S} "
                        f"cnt_le_capacity={cnt_i <= capacity}",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[DEBUG_DW_BOUNDS][rank{rank}] FAILED during host-side bounds sync: {exc}",
                    flush=True,
                )
                raise

        d_weights = torch.zeros_like(weights)
        if os.environ.get("DEBUG_DW_BOUNDS", "0") == "1":
            print(
                f"[DEBUG_DW_BOUNDS][rank{rank}] "
                f"DW_DWEIGHTS_INIT=zeros_like weights_shape={tuple(weights.shape)} "
                f"d_weights_shape={tuple(d_weights.shape)} dtype={d_weights.dtype} "
                f"zeroed_each_backward=True full_overwrite_expected=True fused_dw=True",
                flush=True,
            )
        if use_backward_overlap:
            overlap_max_m_tiles = triton.cdiv(int(capacity), BLOCK_M)
            dx_counter = torch.zeros(
                (E, overlap_max_m_tiles),
                dtype=torch.int32,
                device=tokens.device,
            )

            current_stream = torch.cuda.current_stream()
            stream_dx = torch.cuda.Stream(device=tokens.device)
            stream_dw = torch.cuda.Stream(device=tokens.device)
            stream_dx.wait_stream(current_stream)
            stream_dw.wait_stream(current_stream)

            with torch.cuda.stream(stream_dx):
                if S > 0:
                    grid_dx = (E * overlap_max_m_tiles * h_tiles,)
                    _print_grid("grouped_gemm_dx", grid_dx)
                    grouped_gemm_dx_tiled[grid_dx](
                        d_out_ptr=d_out,
                        weights_ptr=weights,
                        d_tokens_ptr=d_tokens,
                        expert_tkn_cnt=local_expert_cnt,
                        expert_offsets=expert_offsets,
                        dx_counter_ptr=dx_counter,
                        expert_cnt=E,
                        hidden_dim=H,
                        expert_hidden_dim=N,
                        capacity=capacity,
                        max_m_tiles=overlap_max_m_tiles,
                        num_h_tiles=h_tiles,
                        BLOCK_SIZE_M=BLOCK_M,
                        BLOCK_SIZE_H=BLOCK_H,
                        BLOCK_SIZE_K=BLOCK_K,
                    )

            with torch.cuda.stream(stream_dw):
                if cnt_max > 0:
                    grid_dw = (E, h_tiles, n_tiles)
                    grouped_gemm_dw_overlap[grid_dw](
                        tokens_ptr=tokens,
                        d_out_ptr=d_out,
                        d_weights_ptr=d_weights,
                        expert_tkn_cnt=local_expert_cnt,
                        expert_offsets=expert_offsets,
                        dx_counter_ptr=dx_counter,
                        expert_cnt=E,
                        hidden_dim=H,
                        expert_hidden_dim=N,
                        capacity=capacity,
                        total_tokens=S,
                        max_m_tiles=overlap_max_m_tiles,
                        num_h_tiles=h_tiles,
                        BLOCK_SIZE_H=BLOCK_H,
                        BLOCK_SIZE_N=BLOCK_N,
                        BLOCK_SIZE_M=BLOCK_M,
                    )

            current_stream.wait_stream(stream_dx)
            current_stream.wait_stream(stream_dw)
            torch.cuda.synchronize()
        elif use_parallel_backward:
            current_stream = torch.cuda.current_stream()
            stream_dx = torch.cuda.Stream(device=tokens.device)
            stream_dw = torch.cuda.Stream(device=tokens.device)
            stream_dx.wait_stream(current_stream)
            stream_dw.wait_stream(current_stream)

            with torch.cuda.stream(stream_dx):
                if S > 0:
                    grid_dx = (E * max_m_tiles * h_tiles,)
                    _print_grid("grouped_gemm_dx", grid_dx)
                    grouped_gemm_dx[grid_dx](
                        d_out_ptr=d_out,
                        weights_ptr=weights,
                        d_tokens_ptr=d_tokens,
                        expert_tkn_cnt=local_expert_cnt,
                        expert_offsets=expert_offsets,
                        expert_cnt=E,
                        hidden_dim=H,
                        expert_hidden_dim=N,
                        capacity=capacity,
                        max_m_tiles=max_m_tiles,
                        num_h_tiles=h_tiles,
                        BLOCK_SIZE_M=BLOCK_M,
                        BLOCK_SIZE_H=BLOCK_H,
                        BLOCK_SIZE_K=BLOCK_K,
                    )

            with torch.cuda.stream(stream_dw):
                if cnt_max > 0:
                    dw_grid = (E, h_tiles, n_tiles)
                    _print_grid("grouped_gemm_dw_fused", dw_grid)
                    grouped_gemm_dw_fused[dw_grid](
                        tokens_ptr=tokens,
                        d_out_ptr=d_out,
                        d_weights_ptr=d_weights,
                        expert_tkn_cnt=local_expert_cnt,
                        expert_offsets=expert_offsets,
                        expert_cnt=E,
                        hidden_dim=H,
                        expert_hidden_dim=N,
                        capacity=capacity,
                        total_tokens=S,
                        BLOCK_SIZE_H=BLOCK_H,
                        BLOCK_SIZE_N=BLOCK_N,
                        BLOCK_SIZE_M=BLOCK_M,
                    )

            current_stream.wait_stream(stream_dx)
            current_stream.wait_stream(stream_dw)
            torch.cuda.current_stream().synchronize()
        elif cnt_max > 0:
            dw_grid = (E, h_tiles, n_tiles)
            _print_grid("grouped_gemm_dw_fused", dw_grid)
            if os.environ.get("DEBUG_DW_BOUNDS", "0") == "1":
                print(
                    f"[DEBUG_DW_BOUNDS][rank{rank}] "
                    f"DW_FUSED_LAUNCH grid={dw_grid} max_cnt={cnt_max} "
                    f"m_tiles_total={m_tiles_total} H_tiles={h_tiles} EH_tiles={n_tiles} "
                    f"BLOCK_M={BLOCK_M} BLOCK_H={BLOCK_H} BLOCK_EH={BLOCK_N}",
                    flush=True,
                )
            grouped_gemm_dw_fused[dw_grid](
                tokens_ptr=tokens,
                d_out_ptr=d_out,
                d_weights_ptr=d_weights,
                expert_tkn_cnt=local_expert_cnt,
                expert_offsets=expert_offsets,
                expert_cnt=E,
                hidden_dim=H,
                expert_hidden_dim=N,
                capacity=capacity,
                total_tokens=S,
                BLOCK_SIZE_H=BLOCK_H,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_M=BLOCK_M,
            )
            if debug_stages:
                _stage_sync("B5", "after grouped_gemm_dw launch and synchronize")
            else:
                torch.cuda.synchronize()
        elif not use_parallel_backward:
            if debug_stages:
                _stage_sync("B5", "after grouped_gemm_dw launch and synchronize")
            else:
                torch.cuda.synchronize()

        _stage_print(
            "B6",
            "before scatter/all-to-all/backward communication (none in Expert.backward)",
            capacity=capacity,
            S=S,
            expert_offsets_last=off_last,
            local_expert_cnt_last=cnt_last,
            tail=tail,
            local_expert_cnt_max=cnt_max,
            local_expert_cnt_sum=cnt_sum,
            expert_offsets_monotonic=offsets_monotonic,
            tail_le_S=tail_ok,
            max_cnt_le_capacity=max_cnt_ok,
        )
        _stage_sync(
            "B7",
            "after scatter/all-to-all/backward communication and synchronize (none in Expert.backward)",
        )

        return d_tokens, d_weights, None, None

# new stuff for overlap
def expert_overlap(
    state,               # GatherOverlapState
    weights,             # [E, H, N]
    aggregrate_exp_cnt,  # Keep this slot for API compatibility.
):
    if state.gathered_tokens.numel() == 0:
        return torch.empty(
            (0, weights.size(-1)),
            dtype=state.gathered_tokens.dtype,
            device=state.gathered_tokens.device,
        )

    device = state.gathered_tokens.device
    E = weights.size(0)
    H = weights.size(1)
    N = weights.size(2)
    S = state.gathered_tokens.size(0)
    W = state.world_size

    out = torch.empty((S, N), dtype=state.gathered_tokens.dtype, device=device)

    BLOCK_M = state.block_m
    BLOCK_N = 64
    BLOCK_K = 32

    max_m_tiles = state.ready.shape[-1]
    num_n_tiles = triton.cdiv(N, BLOCK_N)

    total_tiles = E * W * max_m_tiles * num_n_tiles
    if total_tiles == 0:
        return out

    grid = (total_tiles,)
    enable_overlap_probes = os.environ.get("OVERLAP_SPIN_PROBE", "0") == "1"
    enable_wait_cycles = os.environ.get("OVERLAP_PROBE_WAIT_CYCLES", "1") == "1"
    diag_max_spin = int(os.environ.get("DIAG_MAX_SPIN", "0"))
    if enable_overlap_probes:
        spin_iters = torch.full((total_tiles,), -2, dtype=torch.int32, device=device)
        wait_cycles = torch.full((total_tiles,), -2, dtype=torch.int64, device=device)
        prog_pid = torch.full((total_tiles,), -2, dtype=torch.int32, device=device)
        prog_cnt = torch.full((total_tiles,), -2, dtype=torch.int32, device=device)
    else:
        spin_iters = torch.empty((0,), dtype=torch.int32, device=device)
        wait_cycles = torch.empty((0,), dtype=torch.int64, device=device)
        prog_pid = torch.empty((0,), dtype=torch.int32, device=device)
        prog_cnt = torch.empty((0,), dtype=torch.int32, device=device)
    if diag_max_spin > 0:
        diag_timeout_flag = torch.zeros((1,), dtype=torch.int32, device=device)
        diag_timeout_pid = torch.full((1,), -1, dtype=torch.int32, device=device)
        diag_timeout_e = torch.full((1,), -1, dtype=torch.int32, device=device)
        diag_timeout_src = torch.full((1,), -1, dtype=torch.int32, device=device)
        diag_timeout_tm = torch.full((1,), -1, dtype=torch.int32, device=device)
        diag_timeout_cnt = torch.full((1,), -1, dtype=torch.int32, device=device)
    else:
        diag_timeout_flag = torch.empty((0,), dtype=torch.int32, device=device)
        diag_timeout_pid = diag_timeout_flag
        diag_timeout_e = diag_timeout_flag
        diag_timeout_src = diag_timeout_flag
        diag_timeout_tm = diag_timeout_flag
        diag_timeout_cnt = diag_timeout_flag
    state.spin_iters = spin_iters if enable_overlap_probes else None
    state.wait_cycles = wait_cycles if enable_overlap_probes else None
    state.prog_pid = prog_pid if enable_overlap_probes else None
    state.prog_cnt = prog_cnt if enable_overlap_probes else None
    state.probe_total_tiles = total_tiles if enable_overlap_probes else 0
    state.probe_num_n_tiles = num_n_tiles if enable_overlap_probes else 0

    _print_grid("grouped_gemm_overlap", grid)
    grouped_gemm_overlap[grid](
        tokens_ptr=state.gathered_tokens,
        weights_ptr=weights,
        out_ptr=out,
        counts_by_src=state.local_expert_cnts,   # [W, E]
        write_meta=state.write_meta,             # [W, E]
        ready=state.ready,                       # [E, W, MAX_TILES]
        spin_iters=spin_iters,
        wait_cycles=wait_cycles,
        prog_pid=prog_pid,
        prog_cnt=prog_cnt,
        diag_timeout_flag=diag_timeout_flag,
        diag_timeout_pid=diag_timeout_pid,
        diag_timeout_e=diag_timeout_e,
        diag_timeout_src=diag_timeout_src,
        diag_timeout_tm=diag_timeout_tm,
        diag_timeout_cnt=diag_timeout_cnt,
        world_size=W,
        expert_cnt=E,
        hidden_dim=H,
        expert_hidden_dim=N,
        max_m_tiles=max_m_tiles,
        num_n_tiles=num_n_tiles,
        ENABLE_PROBES=enable_overlap_probes,
        ENABLE_WAIT_CYCLES=enable_wait_cycles,
        DIAG_MAX_SPIN=diag_max_spin,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )

    if diag_max_spin > 0 and int(diag_timeout_flag.item()) == 1:
        pid, e, src, tm, cnt = torch.cat(
            (
                diag_timeout_pid,
                diag_timeout_e,
                diag_timeout_src,
                diag_timeout_tm,
                diag_timeout_cnt,
            )
        ).cpu().tolist()
        print(
            f"[DIAG_TIMEOUT][rank{dist.get_rank()}] "
            f"pid={pid} e={e} src={src} tm={tm} cnt={cnt}",
            flush=True,
        )

    return out

expert = Expert.apply


class CrossLayerExpertPair:
    """
    Experimental cross-layer backward pipelining for two adjacent local expert
    layers. Existing Expert/autograd paths are untouched; this helper launches
    the same dX and fused dW Triton kernels directly under an explicit schedule.

    Pipeline order:
      stream_0: dX_L -> event_dX_done -> dW_L
      stream_1: wait event_dX_done -> dX_L1 -> dW_L1

    This overlaps dW_L with dX_L1 on GPU. The communication push/pull step is
    intentionally not embedded here because the current push_back kernel needs
    AllToAllGatherOp metadata that is not part of this standalone local pair
    benchmark.
    """

    def __init__(self, expert_L=None, expert_L1=None):
        self.expert_L = expert_L
        self.expert_L1 = expert_L1
        self.stream_0 = torch.cuda.Stream()
        self.stream_1 = torch.cuda.Stream()
        self.event_dX_done = torch.cuda.Event()

    @staticmethod
    def _launch_layer_backward(d_out, tokens, weights, expert_offsets, cnt):
        E = weights.size(0)
        H = weights.size(1)
        N = weights.size(2)
        S = tokens.size(0)
        BLOCK_M = 64
        BLOCK_H = 64
        BLOCK_N = 64
        BLOCK_K = 32

        cnt = cnt.to(torch.int32).contiguous()
        expert_offsets = expert_offsets.to(torch.int32).contiguous()
        d_out = d_out.contiguous()

        d_tokens = torch.empty((S, H), dtype=tokens.dtype, device=tokens.device)
        d_weights = torch.zeros_like(weights)

        h_tiles = triton.cdiv(H, BLOCK_H)
        n_tiles = triton.cdiv(N, BLOCK_N)
        max_m_tiles = triton.cdiv(S, BLOCK_M)
        cnt_max = int(cnt.max().item()) if cnt.numel() > 0 else 0
        capacity = int(os.environ.get("CAPACITY", str(max(cnt_max, 1))))

        if S > 0:
            grid_dx = (E * max_m_tiles * h_tiles,)
            _print_grid("grouped_gemm_dx", grid_dx)
            grouped_gemm_dx[grid_dx](
                d_out_ptr=d_out,
                weights_ptr=weights,
                d_tokens_ptr=d_tokens,
                expert_tkn_cnt=cnt,
                expert_offsets=expert_offsets,
                expert_cnt=E,
                hidden_dim=H,
                expert_hidden_dim=N,
                capacity=capacity,
                max_m_tiles=max_m_tiles,
                num_h_tiles=h_tiles,
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_H=BLOCK_H,
                BLOCK_SIZE_K=BLOCK_K,
            )

        if cnt_max > 0:
            dw_grid = (E, h_tiles, n_tiles)
            _print_grid("grouped_gemm_dw_fused", dw_grid)
            grouped_gemm_dw_fused[dw_grid](
                tokens_ptr=tokens,
                d_out_ptr=d_out,
                d_weights_ptr=d_weights,
                expert_tkn_cnt=cnt,
                expert_offsets=expert_offsets,
                expert_cnt=E,
                hidden_dim=H,
                expert_hidden_dim=N,
                capacity=capacity,
                total_tokens=S,
                BLOCK_SIZE_H=BLOCK_H,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_M=BLOCK_M,
            )

        return d_tokens, d_weights

    def backward_serial(
        self,
        dY_L,
        X_L,
        X_L1,
        W_L,
        W_L1,
        expert_offsets_L,
        expert_offsets_L1,
        cnt_L,
        cnt_L1,
    ):
        dX_L, dW_L = self._launch_layer_backward(
            dY_L, X_L, W_L, expert_offsets_L, cnt_L
        )
        torch.cuda.synchronize()
        dX_L1, dW_L1 = self._launch_layer_backward(
            dX_L, X_L1, W_L1, expert_offsets_L1, cnt_L1
        )
        torch.cuda.synchronize()
        return dX_L1, dW_L, dW_L1

    def backward_pipeline(
        self,
        dY_L,
        X_L,
        X_L1,
        W_L,
        W_L1,
        expert_offsets_L,
        expert_offsets_L1,
        cnt_L,
        cnt_L1,
    ):
        dX_L = torch.empty((X_L.size(0), W_L.size(1)), dtype=X_L.dtype, device=X_L.device)
        dX_L1 = torch.empty((X_L1.size(0), W_L1.size(1)), dtype=X_L1.dtype, device=X_L1.device)
        dW_L = torch.zeros_like(W_L)
        dW_L1 = torch.zeros_like(W_L1)

        BLOCK_M = 64
        BLOCK_H = 64
        BLOCK_N = 64
        BLOCK_K = 32

        cnt_L = cnt_L.to(torch.int32).contiguous()
        cnt_L1 = cnt_L1.to(torch.int32).contiguous()
        expert_offsets_L = expert_offsets_L.to(torch.int32).contiguous()
        expert_offsets_L1 = expert_offsets_L1.to(torch.int32).contiguous()
        dY_L = dY_L.contiguous()

        E_L, H_L, N_L = W_L.shape
        E_L1, H_L1, N_L1 = W_L1.shape
        S_L = X_L.size(0)
        S_L1 = X_L1.size(0)
        if H_L != H_L1:
            raise ValueError(
                f"Layer L hidden_dim ({H_L}) must match Layer L1 hidden_dim ({H_L1})"
            )

        h_tiles_L = triton.cdiv(H_L, BLOCK_H)
        n_tiles_L = triton.cdiv(N_L, BLOCK_N)
        h_tiles_L1 = triton.cdiv(H_L1, BLOCK_H)
        n_tiles_L1 = triton.cdiv(N_L1, BLOCK_N)
        max_m_tiles_L = triton.cdiv(S_L, BLOCK_M)
        max_m_tiles_L1 = triton.cdiv(S_L1, BLOCK_M)
        cnt_max_L = int(cnt_L.max().item()) if cnt_L.numel() > 0 else 0
        cnt_max_L1 = int(cnt_L1.max().item()) if cnt_L1.numel() > 0 else 0
        capacity = int(os.environ.get("CAPACITY", str(max(cnt_max_L, cnt_max_L1, 1))))

        current_stream = torch.cuda.current_stream()
        self.stream_0.wait_stream(current_stream)
        self.stream_1.wait_stream(current_stream)

        with torch.cuda.stream(self.stream_0):
            if S_L > 0:
                grid_dx_L = (E_L * max_m_tiles_L * h_tiles_L,)
                _print_grid("grouped_gemm_dx", grid_dx_L)
                grouped_gemm_dx[grid_dx_L](
                    d_out_ptr=dY_L,
                    weights_ptr=W_L,
                    d_tokens_ptr=dX_L,
                    expert_tkn_cnt=cnt_L,
                    expert_offsets=expert_offsets_L,
                    expert_cnt=E_L,
                    hidden_dim=H_L,
                    expert_hidden_dim=N_L,
                    capacity=capacity,
                    max_m_tiles=max_m_tiles_L,
                    num_h_tiles=h_tiles_L,
                    BLOCK_SIZE_M=BLOCK_M,
                    BLOCK_SIZE_H=BLOCK_H,
                    BLOCK_SIZE_K=BLOCK_K,
                )
            self.event_dX_done.record(self.stream_0)
            if cnt_max_L > 0:
                dw_grid_L = (E_L, h_tiles_L, n_tiles_L)
                _print_grid("grouped_gemm_dw_fused", dw_grid_L)
                grouped_gemm_dw_fused[dw_grid_L](
                    tokens_ptr=X_L,
                    d_out_ptr=dY_L,
                    d_weights_ptr=dW_L,
                    expert_tkn_cnt=cnt_L,
                    expert_offsets=expert_offsets_L,
                    expert_cnt=E_L,
                    hidden_dim=H_L,
                    expert_hidden_dim=N_L,
                    capacity=capacity,
                    total_tokens=S_L,
                    BLOCK_SIZE_H=BLOCK_H,
                    BLOCK_SIZE_N=BLOCK_N,
                    BLOCK_SIZE_M=BLOCK_M,
                )

        with torch.cuda.stream(self.stream_1):
            self.stream_1.wait_event(self.event_dX_done)
            if S_L1 > 0:
                grid_dx_L1 = (E_L1 * max_m_tiles_L1 * h_tiles_L1,)
                _print_grid("grouped_gemm_dx", grid_dx_L1)
                grouped_gemm_dx[grid_dx_L1](
                    d_out_ptr=dX_L,
                    weights_ptr=W_L1,
                    d_tokens_ptr=dX_L1,
                    expert_tkn_cnt=cnt_L1,
                    expert_offsets=expert_offsets_L1,
                    expert_cnt=E_L1,
                    hidden_dim=H_L1,
                    expert_hidden_dim=N_L1,
                    capacity=capacity,
                    max_m_tiles=max_m_tiles_L1,
                    num_h_tiles=h_tiles_L1,
                    BLOCK_SIZE_M=BLOCK_M,
                    BLOCK_SIZE_H=BLOCK_H,
                    BLOCK_SIZE_K=BLOCK_K,
                )

        current_stream.wait_stream(self.stream_0)
        current_stream.wait_stream(self.stream_1)
        if cnt_max_L1 > 0:
            dw_grid_L1 = (E_L1, h_tiles_L1, n_tiles_L1)
            _print_grid("grouped_gemm_dw_fused", dw_grid_L1)
            grouped_gemm_dw_fused[dw_grid_L1](
                tokens_ptr=X_L1,
                d_out_ptr=dX_L,
                d_weights_ptr=dW_L1,
                expert_tkn_cnt=cnt_L1,
                expert_offsets=expert_offsets_L1,
                expert_cnt=E_L1,
                hidden_dim=H_L1,
                expert_hidden_dim=N_L1,
                capacity=capacity,
                total_tokens=S_L1,
                BLOCK_SIZE_H=BLOCK_H,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_M=BLOCK_M,
            )
        torch.cuda.synchronize()
        return dX_L1, dW_L, dW_L1
