import torch.multiprocessing as mp

"""
A lot of this code is taken from: 
https://github.com/ROCm/iris/blob/main/examples/07_gemm_all_scatter/gemm_all_scatter.py
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

import triton
import triton.language as tl

import sys
import os

import iris
"""
First exchange token size data.

## We set number of blocks to number of devices to send the information to.

Suppose we have the following experts:
Ex1, Ex2, Ex3, Ex4.

1 & 2 on one device.

3 & 4 on one device.

We should launch two blocks.

On device 0:
block 1 transmits the information to "local" (alt. also recieves stuff locally).

block 2 transmits the information to device 2 (alt. also recieves stuff locally).

We must use iris.atomic_adds for convenience.


NUMS_global -> array consisting of integers that determine the number of tokens routed to an expert from this TB (dev.).
"""
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
    NUMS_flag -> This is an array of world_size (0 initialized) as flags to understand when comms have finished. 
    This kernel should be launched with grid-size = NUM_EXPERTS, without this invariant, it will fail.
    """
    pid = tl.program_id(0)
    ## Let's code a little defensively for now. ##
    tl.device_assert(world_size == tl.num_programs(0))
    tl.device_assert(NUM_EXPERTS % EP_SIZE == 0)

    for device_id in tl.range(world_size):
        ## Here we may to have make a decision as to which
        ##  expert to route to.

        ## First, figure out the device id this expert belongs to.
        chunk_size : tl.constexpr = NUM_EXPERTS // EP_SIZE

        ## Extract the count. ##
        ptrs = tl.arange(0, chunk_size) + chunk_size * device_id
        cnt = tl.sum(tl.load(META + ptrs, mask= ptrs < world_size))

        ## Can we make this better? Use device_ids only and not anything else?
        if device_id == cur_rank:
            tl.atomic_add(B+device_id, cnt)
            tl.atomic_add(NUMS_flag, 1)
        else:
            iris.atomic_add(
                B + device_id, 
                cnt,
                cur_rank,
                device_id,
                heap_bases,
                mask=None ## Should be a legal call since we're not doing anything special here.
                )
            iris.atomic_add(
                NUMS_flag,
                1,
                cur_rank,
                device_id,
                heap_bases,
                mask=None
            )

def alltoalldispatch_main(
    A, B, META, DATA_flag, stride_am, stride_ak
):
    """
    This is the main kernel that physically transmits the data over.
    A -> vectors to transmit over.
    B -> empty array zero-initialized to plunk all the data into.
    """
    pass

def gen_tensor(
    batch: int, seq: int, hidden_dim: int, 
    world_size: int, num_experts: int, 
    rank: int) -> tuple[torch.tensor, torch.tensor]:
    torch.manual_seed(rank)
    tokens = torch.randn(batch*seq, hidden_dim).to("cuda" if torch.cuda.is_available() else "cpu")

    ## We first do a common load tensor. ##
    assert (batch*seq) % num_experts == 0, 'must be evenly divisible'
    meta = torch.tensor([(batch*seq) // num_experts for _ in range(num_experts)]).to("cuda" if torch.cuda.is_available() else "cpu")

    return tokens, meta

def callee(
    rank: int, batch: int, seq: int, hidden_dim: int, num_experts: int,
    world_size: int
    ):
    """
    This is the callee function for the Shmem-based all-to-all + gemm kernels.

    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
    meta: num_experts sized tensor representing the number of tokens routed to expert_i.
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
    tokens, meta = gen_tensor(batch, seq, hidden_dim, world_size, num_experts, rank)

    device_cnts = torch.zeros(world_size).to(tokens.device)

    ## Instantiate shmem based heap regions over here. ##
    NUMS_flag = shmem.zeros(num_experts)

    ## First, we have to call an alltoall that will aggregrate token level information
    ##   to instantiate buffer sizes.
    alltoalldispatch_preamble[(world_size,1,1)](
        tokens, device_cnts, meta, NUMS_flag, shmem.get_heap_bases(), 
        dist.get_rank(), world_size, num_experts, tokens.shape[0], num_experts // world_size 
    ) 

    iris.barrier()
    ## Let's print device_cnts at the end. ##
    print(device_cnts)

    ## Next, we instantiate token buffers accordingly for the next phase of the all-to-all + gemm.
    routed_token_buffer = shmem.zeros(device_cnts.sum(), tokens.shape[-1])
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    ## Now, we launch the main all-to-all kernel + persistent gemm.
    with torch.cuda.stream(s1):
        ## Call the main all-to-all over here that transmits the data. ##
        pass

    with torch.cuda.stream(s2):
        ## Call the persistent Gemm here that does the MLP compute. ##
        pass

    shmem.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    ## Input parameters. ##
    world_size, batch, seq, hidden_dim = 2, 2, 2, 4  
    num_experts = world_size * 2
    ## A custom test case for convenience. ##
    mp.spawn(callee, args=(batch, seq, hidden_dim, num_experts, world_size), nprocs=world_size, join=True)
