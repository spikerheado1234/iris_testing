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

## Experimental preamble to enable a more general and unbalanced a2a. Currently a WIP. ##
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
    NUM_EXPERTS: tl.constexpr, EP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr 
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
    device_id = pid
    num_progs = tl.num_programs(0)
    tl.device_assert(transmit_size % BLOCK_SIZE == 0)

    non_local_ptrs = tl.arange(0, BLOCK_SIZE)[:, None] * stride_am + tl.arange(0, hidden_dim)[None, :] * stride_ak + cur_rank * transmit_size * stride_am
    ptrs = tl.arange(0, BLOCK_SIZE)[:, None] * stride_am + tl.arange(0, hidden_dim)[None, :] * stride_ak + device_id * transmit_size * stride_am

    for iter in tl.range(tl.cdiv(transmit_size, BLOCK_SIZE)):
        iris.put(
            input_dev_tokens + ptrs,
            routed_token_buffer + non_local_ptrs,
            cur_rank,
            device_id,
            heap_bases,
            mask=tl.arange(0, BLOCK_SIZE)[:, None] + cur_rank < outgoing_buffer_size, 
            )

        non_local_ptrs += BLOCK_SIZE * stride_am
        ptrs += BLOCK_SIZE * stride_am
    
    ## We have to incr the atomic flag only once at the end once the shift is successful. ##
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

def run(
    rank: int, tokens: torch.tensor, meta: torch.tensor, batch: int, seq: int, hidden_dim: int, num_experts: int,
    world_size: int, shmem, general_a2a: bool 
):
    """
    This is the callee function for the Shmem-based all-to-all + gemm kernels.

    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
    meta: num_experts sized tensor representing the number of tokens routed to expert_i.
    general_a2a: Flag to trigger unbalanced a2a capability, currently not working.
    """
    device_id = rank % torch.cuda.device_count()

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
            world_size, num_experts, world_size, BLOCK_SIZE=256  ## Temporarily put 256 but autotune out in the future.
        )
    with torch.cuda.stream(s2):
        ## Call the persistent Gemm here that does the MLP compute. ##
        NUM_REM_SMS = 100 - world_size

    torch.cuda.synchronize()
    shmem.barrier()
    #print(f'[rank: {rank}], summed tensor: {routed_token_buffer.sum()}')