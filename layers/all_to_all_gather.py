import torch
import torch.distributed as dist

from kernels import counts_exchange_pull, roll_cum_sum, token_exchange_pull


class AllToAllGatherOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        cnts,
        offsets,
        heap_bases,
        e_local,
        tokens 
    ):
        """
        Simpler "pull" all-to-all. In this version, device i gathers the relevant tokens from all other devices.
        A benefit of this is that achieving a truly padding free pipeline is now tenable (with little effort) through
        an intermediate metadata preparation step. 

        The flow of the code is as follows:

        `counts_exchange_pull` populates two pieces of metadata: `local_expert_cnts` and `local_expert_offset_idxs`. 

        `local_expert_cnts` -> [world_size, expert_pack] sized tensor. If it resides on device d, then [i, j] = x indicates
        that device i will send x tokens to expert j that resides on device d.

        `local_expert_offset_idxs` -> [world_size] sized tensor. If it resides on device d, then [i] = x indicates that 
        device i's first token to route to d resides at index x.

        Next, an intermediate metadata preparation step prepares two tensors:
            1. read_meta -> [world_size, expert_pack] sized tensor. Here, [i, j] = x on device d indicates that in device i's token buffer
                                the first token to be routed to d of expert j resides at index x.
            2. write_meta -> [world_size, expert_pack] sized tensor. Here, [i, j] = x on device d indicates that in device d's token buffer
                                that the index of the first token of expert j routed to d from device i should be at index x.

        Finally, the token_exchange_pull kernel exchanges the physical tokens. It ingests the metadata and has simple read rules
        where device i will pull in the data from all other devices into its token buffer.
        """

        ## We first instantiate local buffers. ##
        local_expert_cnts = torch.zeros(dist.get_world_size(), e_local, dtype=torch.int64).to(tokens.device)
        local_expert_offset_idxs = torch.zeros(dist.get_world_size(), dtype=torch.int64).to(tokens.device)
        cnt_exchange_sync = torch.zeros(1, dtype=torch.int32).to(tokens.device)

        ## Next, launch the first metadata exchange. ##
        BLOCK_M = 64
        counts_exchange_pull[(dist.get_world_size(),)](
            cnts=cnts,
            offsets=offsets,
            local_expert_cnts=local_expert_cnts,
            local_expert_offset_idxs=local_expert_offset_idxs,
            cnt_exchange_sync=cnt_exchange_sync,
            heap_bases=heap_bases,
            src_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            e_local=e_local,
            BLOCK_M=BLOCK_M
        )
        ## Then, intermediate ops. ##
        read_meta, write_meta = roll_cum_sum(local_expert_cnts, local_expert_offset_idxs)

        gathered_tokens = torch.zeros(local_expert_cnts.sum(), tokens.shape[-1], dtype=tokens.dtype).to(tokens.device)
        token_sync = torch.zeros(dist.get_world_size(), e_local, dtype=torch.int64).to(tokens.device)
        
        ## Finally, launch the next kernel. ##
        token_exchange_pull[(dist.get_world_size(), e_local, local_expert_cnts.max())](
            tokens=tokens,
            read_meta=read_meta,
            write_meta=write_meta,
            local_expert_cnts=local_expert_cnts,
            gathered_tokens=gathered_tokens,
            token_sync=token_sync,
            heap_bases=heap_bases,
            src_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            e_local=e_local,
            hidden_dim=tokens.shape[-1],
            BLOCK_M=1024,
            num_warps=8 
        )

        return gathered_tokens

    @staticmethod
    def backward(ctx, grads):
        raise NotImplementedError('Backwards pass not supported yet!')


gather_a2a = AllToAllGatherOp.apply