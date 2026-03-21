import torch
import torch.distributed as dist

from kernels import token_shuffle_kernel


class TokenShuffle(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        dispatched_tokens, 
        pca
    ):
        """
        Reshuffles tokens from uneven all-to-all to eliminate zero-padding.

        Args:
            ctx: context used for pytorch bwd/fwd.
            dispatched_tokens (Tensor): [E, world_size, capacity, hidden_dim]-sized tensor.
            pca (Tensor): [E, world_size]-sized physical_counts_array tensor.
        """
        e_local, world_size, capacity, hidden_dim = dispatched_tokens.shape

        pca_flat = pca.view(-1)
        pca_offsets = torch.zeros_like(pca_flat)
        pca_offsets[1:] = pca_flat.cumsum(dim=0)[:-1].to(torch.int32)
        
        total_tkn_cnt = int(pca_flat.sum().item())
        
        if total_tkn_cnt == 0:
            return torch.empty((0, hidden_dim), dtype=dispatched_tokens.dtype, device=dispatched_tokens.device)

        shuffled_tokens = torch.zeros((total_tkn_cnt, hidden_dim), dtype=dispatched_tokens.dtype, device=dispatched_tokens.device)

        grid = (e_local, world_size, capacity)

        token_shuffle_kernel[grid](
            pca_offsets, 
            pca,
            dispatched_tokens, 
            shuffled_tokens,
            e_local, 
            world_size,
            capacity, 
            hidden_dim,
            BLOCK_X=128
        ) 

        return shuffled_tokens

    @staticmethod
    def backward(ctx):
        raise NotImplementedError("Backward pass not implemented yet.")

shuffle = TokenShuffle.apply