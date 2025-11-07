import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import iris

# Device-side APIs
@triton.jit
def kernel(buffer, buffer_size: tl.constexpr, block_size: tl.constexpr, heap_bases_ptr, source_rank: tl.constexpr, target_rank: tl.constexpr, number: tl.constexpr):
    # Compute start index of this block
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    iris.store(buffer + offsets, number,
            source_rank, target_rank,
            heap_bases_ptr, mask=mask)

def _worker(rank, world_size):
    # Torch distributed initialization
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )

    # Iris initialization
    heap_size = 2**30   # 1GiB symmetric heap for inter-GPU communication
    iris_ctx = iris.iris(heap_size)
    cur_rank = iris_ctx.get_rank()

    # Iris tensor allocation
    buffer_size = 4096  # 4K elements buffer
    buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)

    # Launch the kernel on rank 0
    block_size = 1024
    grid = lambda meta: (triton.cdiv(buffer_size, meta["block_size"]),)
    print(f'buffer before: {buffer}')
    if cur_rank == 0:
        ## Print stuff out to see their internal types, for own self-use. ##
        print(f'heapbases: {iris_ctx.get_heap_bases()}, type: {type(iris_ctx.get_heap_bases())} iris_rank: {cur_rank}')

    kernel[grid](
        buffer,
        buffer_size,
        block_size,
        iris_ctx.get_heap_bases(),
        cur_rank, (cur_rank + 1) % world_size, cur_rank+17
    )

    # Synchronize all ranks
    iris_ctx.barrier()
    dist.destroy_process_group()
    print(f'buffer after: {buffer}')

if __name__ == "__main__":
    world_size = 8  # Using two ranks
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
