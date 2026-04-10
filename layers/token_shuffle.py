import torch
import torch.distributed as dist

from kernels import token_shuffle


class TokenShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                dispatched_tokens, 
                pca, 
                token_sync, 
                capacity
                ):
        """
        Reshuffles tokens from uneven all-to-all to eliminate zero-padding.

        Args:
            ctx: context used for pytorch bwd/fwd.
            dispatched_tokens (Tensor): [E, world_size, capacity, hidden_dim]-sized tensor
                that stores the dispatched tokens (after all-to-all shuffling).
            pca (Tensor): [E, world_size]-sized physical_counts_array tensor that determines
                number of incoming tokens routed to the current rank for an expert. 
                pca[i, j] = x indicates device j is routing x tokens to expert i on the current rank.
            token_sync (Tensor): [E]-sized array indicating synchronization variables for when the current
                tokens have successfully routed to expert i on the current rank.
        """

        e_local, world_size, _, hidden_dim = dispatched_tokens.shape

        pca_flat = pca.view(-1)
        pca_offsets = torch.zeros_like(pca_flat)
        pca_offsets[1:] = pca_flat.cumsum(dim=0)[:-1].to(torch.int32)

        total_tkn_cnt = int(pca_flat.sum().item())

        if total_tkn_cnt == 0:
            return torch.empty((0, hidden_dim), dtype=dispatched_tokens.dtype, device=dispatched_tokens.device)

        shuffled_tokens = torch.zeros(
            (total_tkn_cnt, hidden_dim),
            dtype=dispatched_tokens.dtype,
            device=dispatched_tokens.device,
        )

        # allocated capacity is still needed as the stride in token_buffer
        TILE_SIZE = 64
        MAX_TILES = (capacity + TILE_SIZE - 1) // TILE_SIZE

        # only launch as many token-id programs as are actually needed locally
        active_cap = max(int(pca.max().item()), 1)
        grid = (e_local, world_size, active_cap)

        token_shuffle[grid](
            pca_offsets,
            pca,
            dispatched_tokens,
            shuffled_tokens,
            token_sync,
            e_local,
            world_size,
            capacity,      # keep real allocated stride here
            hidden_dim,
            128,
            TILE_SIZE,
            MAX_TILES,
        )
        return shuffled_tokens

    @staticmethod
    def backward(ctx):
        pass

shuffle = TokenShuffle.apply