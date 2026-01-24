import torch
import torch.distributed as dist

# from .kernels import counts_exchange_kernel, tokens_exchange_kernel, build_expert_offsets


from ..kernels import counts_exchange_kernel, tokens_exchange_kernel
from ..utils import build_expert_offsets, _assert_cuda_int32

class AllToAllOp(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
       
        # Local buffers that contain messages that will be exchanged to other devices
        # (Local → Remote: this rank produces these; kernels read them and put to peers.)
       
        tokens,       # packed send payload for this rank
                      # Packing convention: DEST-major then EXPERT-local-major (must match dst_offsets/dest_counts).
        dest_counts,  # dest_counts[dst, e] = number of token rows this rank sends to dst's local expert e
        dst_offsets,  # dst_offsets[dst] = starting row offset in `tokens` for destination rank dst
        
        # Local buffers used to store/aggregate results
        # These buffers are potentially written to by other devices
        # (Remote → Local: peers put into these buffers via symmetric memory.)
       
        local_pca,    # local_pca[e, src] is written by remote src ranks in Stage-1 (counts exchange).
        token_buf,    # token_buf[e, src, :, :] is written by remote src ranks in Stage-2 (token exchange).

        # (Remote atomics update these; local consumers spin/wait on them.)
        
        counts_ready, # Stage-1 completion counter on *this* rank.
                      # Each remote src does atomic_add(+1, release) after writing its column of local_pca.
        token_sync,   # Stage-2 per-expert completion counters on *this* rank.
                      # Each remote src does atomic_add(+1, release) for each expert after writing token_buf.

        # Heap bases pointing to symmetric memory arrays of all devices participating in EP.
        # (Used by iris.put / iris.atomic_add to address peers' symmetric allocations.)
        heap_bases,   # [world] pointer-like tensor / address list (implementation-dependent)

        # Local variables used.
        
        experts,       # total number of experts in the EP group (global)
        capacity,      # CAP: max rows per (dst, expert) stored in token_buf; must match token_buf.shape[2]
    ):
        """
    Custom Shmem-based AllToAll for Expert parallelism.
    This implements a two-stage AllToAll.

    Args:
        ctx: context used to store data for bwd pass (unused; backward not implemented).
        tokens (Tensor): local packed send payload ([sum_send, H]).
            tokens are packed DEST-major then EXPERT-local-major; must match dest_counts/dst_offsets.
        dest_counts (Tensor): int32 routing counts ([world_size, E_local]).
            dest_counts[dst, e] = x means this rank sends x token rows to dst's local expert e.
            In Stage-1, this value is written to destination's local_pca[e, src_rank].
        dst_offsets (Tensor): int32 destination offsets ([world_size]).
            dst_offsets[dst] is the starting row offset in `tokens` for the segment sent to rank dst.
        local_pca (Tensor): local physical counts array ([E_local, world_size]) (symmetric).
            local_pca[e, src] = x means remote src will write x tokens belonging to local expert e on this rank.
        counts_ready (Tensor): Stage-1 completion counter ([1]) int32 (symmetric).
            Each remote src atomically increments counts_ready by 1 (release) after finishing writes to local_pca[:, src].
            Any consumer reading local_pca must wait until counts_ready == world_size.
        token_buf (Tensor): Stage-2 receive buffer ([E_local, world_size, CAP, H]) (symmetric).
            Remote src ranks write token rows into token_buf[e, src, :, :] for each local expert e.
        token_sync (Tensor): Stage-2 per-expert completion counters ([E_local]) int32 (symmetric).
            Each remote src atomically increments token_sync[e] by 1 (release) after finishing writes to token_buf[e, src, :, :].
            Any consumer reading token_buf for expert e must wait until token_sync[e] == world_size.
        heap_bases (Tensor): addresses/bases of devices' symmetric heaps (implementation-dependent).
            Used by iris.put / iris.atomic_add to address symmetric allocations on remote devices.
        experts (int): total number of experts in the EP group (global).
            Must be divisible by world_size; E_local = experts // world_size.
        capacity (int): CAP for token_buf (must match token_buf.shape[2]).
            Chosen as a safe upper bound and allocated symmetrically ahead of time; reused across iterations.
    """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        assert experts % world_size == 0, 'EP-group size unevenly calculated.'

        e_local = experts // world_size
        hidden_dim = tokens.shape[-1]

        # Stage-1: counts exchange
        counts_exchange_kernel[(world_size,)](
            dest_counts,     # send_counts_ptr
            local_pca,       # pca_ptr on destination
            counts_ready,    # counts_ready_ptr on destination
            heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_E=128,
            num_warps=4,
        )

        # Stage-2: token exchange
        expert_offs = build_expert_offsets(dest_counts)  # prefix offsets per (dst, expert) :contentReference[oaicite:11]{index=11}
        tokens_exchange_kernel[(world_size, e_local)](
            tokens,
            dest_counts,
            dst_offsets,
            expert_offs,
            token_buf,
            token_sync,
            heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            CAP=capacity,
            hidden_dim=hidden_dim,
            BLOCK_M=32,
            BLOCK_K=128,
            num_warps=8,
        )

        #return only the layer output (token_buf).
        return token_buf

    @staticmethod
    def backward(ctx, *grad_outputs): #better for many writebacks right？
        raise NotImplementedError("Backward pass not implemented yet.")

custom_a2a = AllToAllOp.apply
