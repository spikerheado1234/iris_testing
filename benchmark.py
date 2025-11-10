import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import gen_tensor
from alltoall_persistent_gemm import run as CustomA2A 
from baseline import run as TorchA2A

import triton
import triton.language as tl

import sys
import os
import time

import iris

def callee(
    rank: int, batch: int, seq: int, hidden_dim: int, num_experts: int,
    world_size: int, opt: bool 
    ):
    """
    This is the callee function for the Shmem-based all-to-all + gemm kernels.

    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
    meta: num_experts sized tensor representing the number of tokens routed to expert_i.
    opt: control whether to trigger Custom A2A or baseline, respectively.
    """
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )

    heap_size = (2**30)*5 ## 1 GiB symmetric heap.
    shmem = iris.iris(heap_size)

    tokens, meta = gen_tensor(
        batch, seq, hidden_dim, 
        world_size, num_experts, 
        rank)

    ## Warmup. ##
    for _ in range(5):
        if opt:
            CustomA2A(
                rank, tokens, meta, batch, 
                seq, hidden_dim, num_experts, 
                world_size, shmem, False
                )
        else:
            TorchA2A(
                rank, tokens, meta, batch,
                seq, hidden_dim, num_experts,
                world_size, False, shmem
                )
        shmem.barrier()

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        if opt:
            CustomA2A(
                rank, tokens, meta, batch, 
                seq, hidden_dim, num_experts, 
                world_size, shmem, False
                )
        else:
            TorchA2A(
                rank, tokens, meta, batch,
                seq, hidden_dim, num_experts,
                world_size, False, shmem
                )

    shmem.barrier()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'[rank: {rank}] time taken: {(end_time - start_time):.5f}')
    dist.destroy_process_group()

if __name__ == "__main__":
    ## Input parameters. ##
    world_size, batch, seq, hidden_dim = 8, 8, 2048, 1024
    num_experts = world_size * 2
    run_custom_a2a: bool = False 
    ## A custom test case for convenience. ##
    mp.spawn(callee, args=(batch, seq, hidden_dim, num_experts, world_size, run_custom_a2a), nprocs=world_size, join=True)
