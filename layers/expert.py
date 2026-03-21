import torch
import torch.nn.functional as F

from kernels import grouped_gemm
import torch.distributed as dist
class Expert(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        tokens, 
        weights,
        local_expert_cnt,
        aggregate_exp_cnt
    ):
        """
        Expert GeMM compute stage. Computes a single layer MLP for all local experts.

        Args:
            ctx: context required for torch fwd/bwd pass.
            tokens (Tensor): [S, hidden_dim]-sized tensor. S is the packed size of the tensor.
            weights (Tensor): [E, hidden_dim, ffn_hidden_dim]-sized tensor representation MLP weights per expert.
            local_expert_cnt (Tensor): [E]-sized tensor representing the total number of tokens.
            aggregate_exp_cnt (int): the total number of experts across all devices.
        """
        if tokens.numel() == 0:
            return torch.empty((0, weights.size(-1)), dtype=tokens.dtype, device=tokens.device)

        world_size = dist.get_world_size()
        e_local = aggregate_exp_cnt // world_size

        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 64
        
        if tokens.is_cuda:
            NUM_SM = torch.cuda.get_device_properties(tokens.device).multi_processor_count
        else:
            NUM_SM = 142

        gemm_grid = (NUM_SM,)
        processed_tokens = torch.zeros((tokens.shape[0], weights.size(-1)), dtype=tokens.dtype, device=tokens.device)

        grouped_gemm[gemm_grid](
            tokens,
            weights, 
            processed_tokens, 
            local_expert_cnt, 
            NUM_SM=NUM_SM,
            expert_cnt=e_local,
            hidden_dim=tokens.size(-1),
            expert_hidden_dim=weights.size(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )

        return processed_tokens 

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError('Backward pass not implemented yet.')

expert = Expert.apply