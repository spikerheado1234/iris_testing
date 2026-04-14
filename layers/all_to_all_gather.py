import torch
import torch.distributed as dist

from kernels import counts_exchange_pull, roll_cum_sum, token_exchange_pull
from utils import nvtx_push, nvtx_pop


class AllToAllGatherOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        cnts,              # local symm tensor [W, E]
        offsets,           # local symm tensor [W]
        tokens,            # local symm tensor [T, H]
        cnts_bases,        # [W] int64 on cuda
        offsets_bases,     # [W] int64 on cuda
        tokens_bases,      # [W] int64 on cuda
        e_local,
        
    ):
        """
        Pull-style all-to-all gather implemented on top of Pytorch symmetric memory.
        Fllowed by the orginal gather design

        On rank d, the operation proceeds in three stages:

        1. Metadata pull:
           The first kernel pulls from every peer the routing metadata relevant to rank d:
             - local_expert_cnts[i, j]: number of tokens that peer i sends to expert j on rank d
             - local_expert_offset_idxs[i]: starting row in peer i's packed send buffer for tokens routed to rank d

        2. Local metadata preparation:
           `roll_cum_sum` converts the pulled metadata into:
             - read_meta[i, j]: starting row in peer i's send buffer for expert j destined for rank d
             - write_meta[i, j]: starting row in rank d's packed receive buffer for tokens pulled
               from peer i for expert j

        3. Token pull:
           The second kernel uses the metadata above to pull token rows directly from each peer's
           symmetric token buffer into a packed local receive buffer (`gathered_tokens`).

        Such may keep the gather path padding-free after metadata preparation 
        """
        
        ## New sort of initilization $$
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = tokens.device 


        ## We first instantiate local buffers. ##
        local_expert_cnts = torch.zeros((world_size, e_local), dtype=torch.int64, device=device)
        local_expert_offset_idxs = torch.zeros((world_size,), dtype=torch.int64, device=device)
        cnt_exchange_sync = torch.zeros((1,), dtype=torch.int32, device=device)


        ## Next, launch the first metadata exchange. ##
        BLOCK_M = 64
        nvtx_push("Stage1_Meta_Pull")
        counts_exchange_pull[(world_size,)](
            cnts_bases=cnts_bases,
            offsets_bases=offsets_bases,
            local_expert_cnts=local_expert_cnts,
            local_expert_offset_idxs=local_expert_offset_idxs,
            cnt_exchange_sync=cnt_exchange_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_M=BLOCK_M,
        )
        nvtx_pop()


        ## Then, intermediate ops. ##
        nvtx_push("Stage1.5_Roll_Cum_Sum")
        read_meta, write_meta = roll_cum_sum(local_expert_cnts, local_expert_offset_idxs)

        total_recv = int(local_expert_cnts.sum().item())
        gathered_tokens = torch.empty((total_recv, tokens.shape[-1]), dtype=tokens.dtype, device=device)
        token_sync = torch.zeros((world_size, e_local), dtype=torch.int32, device=device)
        
        max_cnt = max(int(local_expert_cnts.max().item()), 1)
        nvtx_pop()

        ## Finally, launch the next kernel. ##
        nvtx_push("Stage2_Token_Pull")
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
              
        ## wait for everything finished ##
        #target = local_expert_cnts.to(torch.int32)
        #while not bool(torch.equal(token_sync, target)):
        #    pass
        torch.cuda.synchronize(device)
        nvtx_pop()

        return gathered_tokens

    @staticmethod
    def backward(ctx, grads):
        raise NotImplementedError('Backwards pass not supported yet!')


gather_a2a = AllToAllGatherOp.apply