import os

import torch
import torch.distributed as dist
import triton

from backward_kernels import (
    grouped_gemm_dw_fused,
    grouped_gemm_dx,
    token_exchange_push_back,
)
from kernels import (
    counts_exchange_pull,
    grouped_gemm_rect,
    roll_cum_sum,
    token_exchange_pull,
)
from layers.all_to_all_gather import _get_backward_tokens_workspace


class GatherExpertOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cnts,
        offsets,
        tokens,
        cnts_bases,
        offsets_bases,
        tokens_bases,
        e_local,
        weights,
        local_expert_cnt,
        aggregrate_exp_cnt,
    ):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = tokens.device
        backward_ws = _get_backward_tokens_workspace(tokens, world_size)

        local_expert_cnts = torch.zeros(
            (world_size, e_local), dtype=torch.int64, device=device
        )
        local_expert_offset_idxs = torch.zeros(
            (world_size,), dtype=torch.int64, device=device
        )
        cnt_exchange_sync = torch.zeros((1,), dtype=torch.int32, device=device)

        block_m = 64
        counts_exchange_pull[(world_size,)](
            cnts_bases=cnts_bases,
            offsets_bases=offsets_bases,
            local_expert_cnts=local_expert_cnts,
            local_expert_offset_idxs=local_expert_offset_idxs,
            cnt_exchange_sync=cnt_exchange_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_M=block_m,
        )

        read_meta, write_meta = roll_cum_sum(
            local_expert_cnts, local_expert_offset_idxs
        )
        total_recv = int(local_expert_cnts.sum().item())
        max_cnt = max(int(local_expert_cnts.max().item()), 1)
        gathered_tokens = torch.empty(
            (total_recv, tokens.shape[-1]), dtype=tokens.dtype, device=device
        )
        token_sync = torch.zeros(
            (world_size, e_local), dtype=torch.int32, device=device
        )

        token_exchange_pull[(world_size, e_local, max_cnt)](
            tokens_bases=tokens_bases,
            read_meta=read_meta,
            write_meta=write_meta,
            local_expert_cnts=local_expert_cnts,
            gathered_tokens=gathered_tokens,
            token_sync=token_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            hidden_dim=tokens.shape[-1],
            BLOCK_K=256,
            num_warps=4,
        )

        expert_cnt = weights.size(0)
        hidden_dim = weights.size(1)
        expert_hidden_dim = weights.size(2)
        expert_offsets = torch.zeros_like(
            local_expert_cnt, dtype=torch.int32, device=device
        )
        if expert_cnt > 1:
            expert_offsets[1:] = local_expert_cnt.cumsum(dim=0)[:-1].to(
                torch.int32
            )

        out = torch.empty(
            (total_recv, expert_hidden_dim),
            dtype=gathered_tokens.dtype,
            device=device,
        )
        block_n = 64
        block_k = 32
        max_m_tiles = triton.cdiv(total_recv, block_m)
        num_n_tiles = triton.cdiv(expert_hidden_dim, block_n)
        total_tiles = expert_cnt * max_m_tiles * num_n_tiles
        if total_tiles > 0:
            grouped_gemm_rect[(total_tiles,)](
                tokens_ptr=gathered_tokens,
                weights_ptr=weights,
                out_ptr=out,
                expert_tkn_cnt=local_expert_cnt,
                expert_offsets=expert_offsets,
                expert_cnt=expert_cnt,
                hidden_dim=hidden_dim,
                expert_hidden_dim=expert_hidden_dim,
                max_m_tiles=max_m_tiles,
                num_n_tiles=num_n_tiles,
                NUM_SM=int(os.environ.get("GGEMM_NUM_SM", "1")),
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
            )

        ctx.save_for_backward(
            read_meta,
            write_meta,
            local_expert_cnts,
            gathered_tokens,
            weights,
            local_expert_cnt,
            expert_offsets,
        )
        ctx.d_tokens_bases = backward_ws["d_tokens_bases"]
        ctx.d_tokens = backward_ws["d_tokens"]
        ctx.world_size = world_size
        ctx.e_local = e_local
        ctx.expert_cnt = expert_cnt
        ctx.hidden_dim = hidden_dim
        ctx.expert_hidden_dim = expert_hidden_dim
        ctx.total_recv = total_recv
        ctx.tokens_shape = tokens.shape
        ctx.aggregrate_exp_cnt = (
            aggregrate_exp_cnt if torch.is_tensor(aggregrate_exp_cnt) else None
        )

        return out

    @staticmethod
    def backward(ctx, d_out):
        (
            read_meta,
            write_meta,
            local_expert_cnts,
            gathered_tokens,
            weights,
            local_expert_cnt,
            expert_offsets,
        ) = ctx.saved_tensors

        debug_stages = os.environ.get("DEBUG_BACKWARD_STAGES", "0") == "1"
        sync_backward = (
            os.environ.get("SYNC_BACKWARD_KERNELS", "0") == "1"
            or os.environ.get("AMD_SERIALIZE_KERNEL", "") in ("1", "3")
        )
        debug_bounds = os.environ.get("DEBUG_DW_BOUNDS", "0") == "1"
        use_parallel = not debug_stages and not sync_backward and not debug_bounds
        skip_push_back = os.environ.get("DEBUG_SKIP_PUSH_BACK", "0") == "1"
        rank = dist.get_rank() if dist.is_initialized() else -1

        def stage_sync(stage, message):
            if not debug_stages:
                return
            torch.cuda.synchronize()
            print(
                f"[DEBUG_BACKWARD_STAGES][rank{rank}] {stage}: {message}",
                flush=True,
            )

        d_out = d_out.contiguous()
        d_gathered_tokens = torch.empty_like(gathered_tokens)
        d_weights = torch.zeros_like(weights)

        expert_cnt = ctx.expert_cnt
        hidden_dim = ctx.hidden_dim
        expert_hidden_dim = ctx.expert_hidden_dim
        total_recv = ctx.total_recv
        capacity = int(os.environ.get("CAPACITY", str(total_recv)))

        block_m = 64
        block_h = 64
        block_n = 64
        block_k = 32
        h_tiles = triton.cdiv(hidden_dim, block_h)
        n_tiles = triton.cdiv(expert_hidden_dim, block_n)
        max_m_tiles = triton.cdiv(total_recv, block_m)
        max_comm_cnt = max(int(local_expert_cnts.max().item()), 1)
        max_comm_tiles = triton.cdiv(max_comm_cnt, block_m)
        local_cnt_max = (
            int(local_expert_cnt.max().item())
            if local_expert_cnt.numel() > 0
            else 0
        )

        def launch_dx():
            if total_recv == 0:
                return
            grouped_gemm_dx[(expert_cnt * max_m_tiles * h_tiles,)](
                d_out_ptr=d_out,
                weights_ptr=weights,
                d_tokens_ptr=d_gathered_tokens,
                expert_tkn_cnt=local_expert_cnt,
                expert_offsets=expert_offsets,
                expert_cnt=expert_cnt,
                hidden_dim=hidden_dim,
                expert_hidden_dim=expert_hidden_dim,
                capacity=capacity,
                max_m_tiles=max_m_tiles,
                num_h_tiles=h_tiles,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_H=block_h,
                BLOCK_SIZE_K=block_k,
            )

        def launch_push_back():
            if skip_push_back:
                return
            token_exchange_push_back[
                (ctx.world_size, ctx.e_local, max_comm_tiles)
            ](
                d_tokens_bases=ctx.d_tokens_bases,
                read_meta=read_meta,
                write_meta=write_meta,
                local_expert_cnts=local_expert_cnts,
                d_gathered_tokens=d_gathered_tokens,
                src_rank=rank,
                world_size=ctx.world_size,
                e_local=ctx.e_local,
                hidden_dim=ctx.tokens_shape[-1],
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_K=256,
                num_warps=4,
            )

        def launch_dw():
            if local_cnt_max == 0:
                return
            grouped_gemm_dw_fused[(expert_cnt, h_tiles, n_tiles)](
                tokens_ptr=gathered_tokens,
                d_out_ptr=d_out,
                d_weights_ptr=d_weights,
                expert_tkn_cnt=local_expert_cnt,
                expert_offsets=expert_offsets,
                expert_cnt=expert_cnt,
                hidden_dim=hidden_dim,
                expert_hidden_dim=expert_hidden_dim,
                capacity=capacity,
                total_tokens=total_recv,
                BLOCK_SIZE_H=block_h,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_M=block_m,
            )

        if use_parallel:
            current_stream = torch.cuda.current_stream()
            stream_comm = torch.cuda.Stream(device=gathered_tokens.device)
            stream_dw = torch.cuda.Stream(device=gathered_tokens.device)
            stream_comm.wait_stream(current_stream)
            stream_dw.wait_stream(current_stream)

            with torch.cuda.stream(stream_comm):
                launch_dx()
                launch_push_back()

            with torch.cuda.stream(stream_dw):
                launch_dw()

            current_stream.wait_stream(stream_comm)
            current_stream.wait_stream(stream_dw)
        else:
            launch_dx()
            stage_sync("B3", "after grouped_gemm_dx launch and synchronize")
            if sync_backward and not debug_stages:
                torch.cuda.synchronize()

            launch_push_back()
            stage_sync("B4", "after token_exchange_push_back and synchronize")
            if sync_backward and not debug_stages:
                torch.cuda.synchronize()

            launch_dw()
            if debug_stages:
                stage_sync(
                    "B5", "after grouped_gemm_dw_fused launch and synchronize"
                )
            else:
                torch.cuda.synchronize()

        d_tokens = ctx.d_tokens.clone()
        torch.cuda.current_stream().synchronize()
        return (
            None,
            None,
            d_tokens,
            None,
            None,
            None,
            None,
            d_weights,
            None,
            None,
        )


gather_expert = GatherExpertOp.apply
