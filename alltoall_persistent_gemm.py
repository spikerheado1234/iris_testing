"""
A lot of this code is taken from: 
https://github.com/ROCm/iris/blob/main/examples/07_gemm_all_scatter/gemm_all_scatter.py
"""
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import gen_tensor

import triton
import triton.language as tl

import sys
import os

import iris

## Experimental preamble to enable a more general and unbalanced a2a.
@triton.jit
def alltoalldispatch_preamble(
    A, B, META, NUMS_flag, heap_bases,
    cur_rank: tl.constexpr, world_size: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    N: tl.constexpr, EP_SIZE: tl.constexpr):
    """
    This is a fairly non-intuitive kernel. It is meant to emulate an `all_to_all_single` 
    from pytorch (link here: https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single).

    A -> [a1, a2, ...., ak] -> these are the physical values that need to be scattered to the different GPUs.
    B -> [0... 0] 0 initialized array of size world_size that determines how many tokens 
                        should be routed from dev_other -> curr_dev. 
    META -> [m1, m2, ... mf] -> of size: num_experts. This determines the number of tokens routed to expert mi 
                                    from this device.
    NUMS_flag -> This is a unit-sized array which will symbolize when the comms has completed.
    This kernel should be launched with grid-size = NUM_EXPERTS, without this invariant, it will fail.
    """
    pid = tl.program_id(0)
    ## Let's code a little defensively for now. ##
    tl.device_assert(NUM_EXPERTS % EP_SIZE == 0)

    ## We can later replace this loop with block-level parallelism: launch world_size blocks. ##
    for device_id in tl.range(world_size):
        ## Here we may to have make a decision as to which
        ##  expert to route to.

        ## First, figure out the device id this expert belongs to.
        chunk_size : tl.constexpr = NUM_EXPERTS // EP_SIZE

        ## Extract the count. ##
        ptrs = tl.arange(0, chunk_size) + chunk_size * device_id
        cnt = tl.sum(tl.load(META + ptrs, mask=ptrs<NUM_EXPERTS))
        
        ## Can we make this better? Use device_ids only and not anything else?
        iris.atomic_add(
            B + cur_rank, 
            cnt,
            cur_rank,
            device_id,
            heap_bases,
            mask=None, ## Should be a legal call since we're not doing anything special here.
            sem="acquire",
            scope="sys"
            )
        iris.atomic_add(
            NUMS_flag,
            1,
            cur_rank,
            device_id,
            heap_bases,
            mask=None,
            sem="release",
            scope="sys"
        )

    ## Is this needed since we synchronize using shmem.barier() later? ##
    ## Experiments indicate that we do need them. Seems like in-flight 
    ##  values will not be flushed and observed out of the kernel.
    world_size_i32 = tl.full([], world_size, dtype=tl.int32)
    while tl.load(NUMS_flag) != world_size_i32:
        pass

## Currently only supports transmission of fixed, transmit_size packets. ##
@triton.jit
def alltoalldispatch_main(
    routed_token_buffer, input_dev_tokens, 
    DATA_flag, stride_am, stride_ak, heap_bases, 
    token_cnt: tl.constexpr, outgoing_buffer_size: tl.constexpr,
    hidden_dim: tl.constexpr, transmit_size: tl.constexpr, 
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    NUM_EXPERTS: tl.constexpr, EP_SIZE: tl.constexpr 
):
    """
    This is the main kernel that physically transmits the data over.
    routed_token_buffer -> the array of tokens that we paste into (tokens come from other devices)
    input_dev_tokens -> the array of tokens currently residing on this device.
    DATA_flag -> unit-sized array that will determine when comms have finished.
    transmit_size -> number of tokens from Device A -> B per message (we fix this for ease of implementation).
    cur_rank -> self-explanatory.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    #ptrs = tl.arange(0, transmit_size)[:, None] * stride_am + tl.arange(0, hidden_dim)[None, :] * stride_ak
    non_local_ptrs = tl.arange(0, transmit_size)[:, None] * stride_am + tl.arange(0, hidden_dim)[None, :] * stride_ak + cur_rank * transmit_size * stride_am

    ## Optimise this out into multiple blocks once finished base implementation with correctness tested. ##
    device_id = pid
    ptrs = tl.arange(0, transmit_size)[:, None] * stride_am + tl.arange(0, hidden_dim)[None, :] * stride_ak + device_id * transmit_size * stride_am
    #for device_id in tl.range(world_size):
    iris.put(
        input_dev_tokens + ptrs,
        routed_token_buffer + non_local_ptrs,
        cur_rank,
        device_id,
        heap_bases,
        mask=tl.arange(0, transmit_size)[:, None] + cur_rank < outgoing_buffer_size, 
        )
    iris.atomic_add(
        DATA_flag,
        1,
        cur_rank,
        device_id,
        heap_bases,
        mask=None,
        sem="release",
        scope="sys" 
    )

    #world_size_i32 = tl.full([], world_size, dtype=tl.int32)
    #while tl.load(DATA_flag) != world_size_i32:
    #    pass


def callee(
    rank: int, batch: int, seq: int, hidden_dim: int, num_experts: int,
    world_size: int, general_a2a: bool 
    ):
    """
    This is the callee function for the Shmem-based all-to-all + gemm kernels.

    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
    meta: num_experts sized tensor representing the number of tokens routed to expert_i.
    general_a2a: Flag to trigger unbalanced a2a capability, currently not working.
    """
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )
    heap_size = 2**30 ## 1 GiB symmetric heap.
    shmem = iris.iris(heap_size)
    tokens, meta = gen_tensor(
        batch, seq, hidden_dim, 
        world_size, num_experts, 
        rank)

    if general_a2a:
        ## Instantiate shmem based heap regions over here. ##
        NUMS_flag = shmem.zeros(1, dtype=torch.int32, device="cuda")
        device_cnts = shmem.zeros(world_size, device="cuda")

        ## First, we have to call an alltoall that will aggregrate token level information
        ##   to instantiate buffer sizes.
        alltoalldispatch_preamble[(1,1,1)](
            tokens, device_cnts, meta, NUMS_flag, shmem.get_heap_bases(), 
            dist.get_rank(), world_size, num_experts, tokens.shape[0], world_size
        ) 

        ## Next, we instantiate token buffers accordingly for the next phase of the all-to-all + gemm. ##
        routed_token_buffer = shmem.zeros(int(round(device_cnts.sum().item())), tokens.shape[-1])
    else:
        ## For now, we fix a fix sized buffer to transmit over, otherwise things get far too complicated.
        assert tokens.shape[0] % world_size == 0, 'Tensor sizes not properly shaped.'
        transmit_size = tokens.shape[0] // world_size 
        routed_token_buffer = shmem.zeros(transmit_size * world_size, tokens.shape[-1], dtype=tokens.dtype, device="cuda")
        DATA_flag = shmem.zeros(world_size, device="cuda")
        outgoing_buffer_size = transmit_size * world_size

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    ## Now, we launch the main all-to-all kernel + persistent gemm. ##
    with torch.cuda.stream(s1):
        alltoalldispatch_main[(world_size,1,1)](
            routed_token_buffer, tokens, DATA_flag,
            tokens.stride(0), tokens.stride(1), shmem.get_heap_bases(), 
            tokens.shape[0], transmit_size * world_size, 
            tokens.shape[-1], transmit_size, rank, 
            world_size, num_experts, world_size
        )
    with torch.cuda.stream(s2):
        ## Call the persistent Gemm here that does the MLP compute. ##
        pass

    s1.synchronize()
    s2.synchronize()
    shmem.barrier()
    print(f'[rank: {rank}], routed_token_buffer sum: {routed_token_buffer.sum()}')
    dist.destroy_process_group()

if __name__ == "__main__":
    ## Input parameters. ##
    world_size, batch, seq, hidden_dim = 8, 8, 4, 4
    num_experts = world_size * 2
    ## A custom test case for convenience. ##
    mp.spawn(callee, args=(batch, seq, hidden_dim, num_experts, world_size, False), nprocs=world_size, join=True)
