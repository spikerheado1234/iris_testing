from __future__ import annotations
import random
from typing import Tuple, List
import torch
import torch.distributed as dist
import os
from dataclasses import dataclass
import time
# in case some error of naming so I think the predicton here is necessary
try:
    import torch.distributed._symmetric_memory as symmem
    HAS_SYMMEM = True
except ImportError:
    symmem = None
    HAS_SYMMEM = False

import torch.nn.functional as F

########################
# SymmMem-backed buffers
########################

@dataclass
class ShmemBuffers:
    # Tensor objects (kept to prevent garbage collection)
    pca: torch.Tensor
    counts_ready: torch.Tensor
    token_buf: torch.Tensor
    token_sync: torch.Tensor
    tile_counter: torch.Tensor
    
    # Tensor containing base pointers for all ranks (passed to Triton kernels)
    pca_bases: torch.Tensor
    counts_ready_bases: torch.Tensor
    token_buf_bases: torch.Tensor
    token_sync_bases: torch.Tensor

    # new hanler based shmempart for pytorch 
    pca_hdl: object
    counts_ready_hdl: object
    token_buf_hdl: object
    token_sync_hdl: object

def _device_for_rank(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}")


def _as_cuda_i64_ptr_tensor(ptrs, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(ptrs), dtype=torch.int64, device=device)

### get_buffer_ptrs and alloc symmetric will no longer be utilized due to pytorch difference from iris

# new alloc with pytorch symmmem
def _alloc_and_rdzv(
        shape, 
        dtype: torch.dtype, 
        device: torch.device
):
    """
    Allocate a symmetric tensor and immediately rendezvous it.

    Returns:
        tensor, handle, base_ptrs_tensor[int64 on CUDA]
    """
    # only for testing
    if not HAS_SYMMEM:
        raise RuntimeError("PyTorch SymmMem is required but not available")

    t = symmem.empty(shape, dtype=dtype, device=device)
    hdl = symmem.rendezvous(t, group=dist.group.WORLD)

    # PyTorch SymmMem exposes peer base pointers via the handle.
    bases = _as_cuda_i64_ptr_tensor(hdl.buffer_ptrs, device)
    return t, hdl, bases


# updated shmem buufer alloc with handle support of pytorch ver
def alloc_shmem_buffers(
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
) -> ShmemBuffers:
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    """
    Allocate all buffers needed by custom_a2a.

    Layouts must stay compatible with all_to_all.py / kernels.py:
      pca          : [e_local, world_size]         int32
      counts_ready : [1]                           int32
      token_buf    : [e_local, world_size, cap, H] token_dtype
      token_sync   : [e_local]                     int32
      tile_counter : [e_local, world_size]         int32   (local scratch only)
    """
    rank = dist.get_rank()
    device = _device_for_rank(rank)

    # new const for tile logic
    TILE_SIZE = 64
    MAX_TILES = (capacity + TILE_SIZE - 1) // TILE_SIZE
    
    # SymmMem tensors + handles + peer base pointers
    pca, pca_hdl, pca_bases = _alloc_and_rdzv(
        (e_local, world_size), torch.int32, device
    )
    counts_ready, counts_ready_hdl, counts_ready_bases = _alloc_and_rdzv(
        (1,), torch.int32, device
    )
    token_buf, token_buf_hdl, token_buf_bases = _alloc_and_rdzv(
        (e_local, world_size, capacity, hidden_dim), token_dtype, device
    )
    token_sync, token_sync_hdl, token_sync_bases = _alloc_and_rdzv(
        (e_local, world_size, MAX_TILES), torch.int32, device
    ) #3D matrix now

    # local-only scratch 3dnow
    tile_counter = torch.zeros((e_local, world_size, MAX_TILES), dtype=torch.int32, device=device)

    print(f"[rank{rank}] pca_bases={pca_bases.tolist()}", flush=True)
    print(f"[rank{rank}] counts_ready_bases={counts_ready_bases.tolist()}", flush=True)
    print(f"[rank{rank}] token_buf_bases={token_buf_bases.tolist()}", flush=True)
    print(f"[rank{rank}] token_sync_bases={token_sync_bases.tolist()}", flush=True)

    return ShmemBuffers(
        pca=pca, 
        counts_ready=counts_ready, 
        token_buf=token_buf,
        token_sync=token_sync, 
        tile_counter=tile_counter,
        pca_bases=pca_bases, 
        counts_ready_bases=counts_ready_bases,
        token_buf_bases=token_buf_bases, 
        token_sync_bases=token_sync_bases,
        pca_hdl=pca_hdl, 
        counts_ready_hdl=counts_ready_hdl,
        token_buf_hdl=token_buf_hdl, 
        token_sync_hdl=token_sync_hdl,
    )


def reset_custom_iter_state(buffers, clear_token_buf: bool = False) -> None:
    
    """
    Per-iteration reset for custom path.

    For correctness/debugging, clear everything that may contain stale state.
    For pure perf runs, you can decide whether to skip some zeroing.
    """
    buffers.pca.zero_()
    buffers.counts_ready.zero_()
    buffers.token_sync.zero_()
    buffers.tile_counter.zero_()
    if clear_token_buf:
        buffers.token_buf.zero_()
################
# Small helpers
###############
  
def wait_token_sync_ready(
    token_sync: torch.Tensor,
    world_size: int,
    *,
    sleep_us: int = 0,
    max_spins: int = 1_000_000,
) -> None:
    """
    Host-side completion wait for scatter payload readiness.

    token_sync: [E_local] int32 CUDA tensor.
    We wait until every local expert has received one completion signal
    from every source rank, i.e. token_sync[e] == world_size for all e.

    This is intentionally a minimal benchmark-side consume, not a final
    device-side protocol implementation.
    """
    assert token_sync.is_cuda, "token_sync must be CUDA"
    assert token_sync.dtype == torch.int32, "token_sync must be int32"

    target = int(world_size)
    spins = 0
    while True:
        # .item() gives us a simple host-visible completion condition.
        ready = bool(torch.all(token_sync == target).item())
        if ready:
            return

        spins += 1
        if spins >= max_spins:
            cur_min = int(token_sync.min().item())
            cur_max = int(token_sync.max().item())
            raise RuntimeError(
                f"wait_token_sync_ready timeout: expected all == {target}, "
                f"but observed min={cur_min}, max={cur_max}"
            )

        if sleep_us > 0:
            time.sleep(sleep_us * 1e-6)
def nvtx_push(msg: str) -> None:
    torch.cuda.nvtx.range_push(msg)

def nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


# judging and checking cap
def _allreduce_max_i32(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def _allreduce_min_i32(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.MIN)
    return y


def _sync_and_check(ok_flag: torch.Tensor) -> int:
    ok_global = _allreduce_min_i32(ok_flag).item()
    if ok_global == 1:
        dist.barrier()
    return ok_global


def _assert_cuda_int32(x: torch.Tensor, name: str) -> None:
    assert x.is_cuda, f"{name} must be CUDA"
    assert x.dtype == torch.int32, f"{name} must be int32"



def _build_dst_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """
    dst_offsets[dst] = prefix sum of total tokens to earlier destinations.
    send_counts: [world, e_local] int32
    """
    send_dst_sizes = send_counts.sum(dim=1).to(torch.int32)
    dst_offsets = (torch.cumsum(send_dst_sizes, dim=0) - send_dst_sizes).to(torch.int32)
    return dst_offsets.contiguous()

def build_expert_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """
    Prefix offsets within each destination block in send_payload.

    send_counts: [world, E_local] int32
    returns: expert_offs[dst, e]
    """
    _assert_cuda_int32(send_counts, "send_counts")
    sc64 = send_counts.to(torch.int64)
    offs = (torch.cumsum(sc64, dim=1) - sc64).to(torch.int32)
    return offs.contiguous()



def _masked_stats(
    custom_buf: torch.Tensor,
    base_buf: torch.Tensor,
    counts_mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute max|diff| and sums over ONLY the valid (non-padding) region.

    custom_buf/base_buf: [E, world, CAP, H]
    counts_mat:           [E, world]  (counts_mat[e, src] = number of valid rows)

    Returns: (max_diff, sum_custom, sum_base) as 0-dim float32 tensors on GPU.
    """
    assert custom_buf.shape == base_buf.shape
    E, W, CAP, H = custom_buf.shape

    # mask[e, src, m] = (m < counts_mat[e, src])
    m = torch.arange(CAP, device=custom_buf.device, dtype=torch.int32)[None, None, :]
    mask = (m < counts_mat.to(torch.int32)[:, :, None]).unsqueeze(-1)  # [E, W, CAP, 1]

    diff = (custom_buf - base_buf).abs().to(torch.float32)
    diff_masked = diff * mask.to(torch.float32)

    max_diff = diff_masked.max()

    sum_custom = (custom_buf.to(torch.float32) * mask.to(torch.float32)).sum()
    sum_base = (base_buf.to(torch.float32) * mask.to(torch.float32)).sum()

    return max_diff, sum_custom, sum_base

def compute_capacity_from_pca(pca: torch.Tensor) -> int:
    cap = pca.max().to(torch.int32)
    dist.all_reduce(cap, op=dist.ReduceOp.MAX)
    return max(int(cap.item()), 1)




############################
# Data generation / routing
############################

def set_seed(base_seed: int, rank: int) -> None:
    #Use different seeds per-rank for token generation / routing decisions.
  
    seed = int(base_seed) + int(rank)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gen_local_tokens(
    batch: int,
    seq: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    base_seed: int,
    rank: int,
) -> torch.Tensor:
    #Generate local tokens [batch*seq, hidden_dim] deterministically per rank
    set_seed(base_seed, rank)
    return torch.rand(batch * seq, hidden_dim, dtype=dtype, device=device)


def gen_router(
    hidden_dim: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    base_seed: int,
) -> torch.Tensor:
    #Generate a (shared-across-ranks) router matrix [H, num_experts].
    # same router across ranks; only tokens differ per rank.
    random.seed(int(base_seed) + 777)
    torch.manual_seed(int(base_seed) + 777)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(base_seed) + 777)
        torch.cuda.manual_seed_all(int(base_seed) + 777)

    router = torch.zeros(hidden_dim, num_experts, dtype=dtype, device=device)
    router = router + torch.randn_like(router) * 0.001
    return router


def route_and_pack_padding_free(
    tokens: torch.Tensor,
    router: torch.Tensor,
    topk: int,
    world_size: int,
    num_experts_total: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padding-free routing -> (send_payload, send_counts, send_dst_offsets, send_dst_sizes).

    Layout of send_payload is DEST-major then EXPERT-local-major:
      [dst0][e0 tokens][e1 tokens]...[eE-1 tokens] [dst1] ...

    send_counts: int32 [world, E_local]
    send_dst_offsets: int32 [world] (row offsets into send_payload)
    send_dst_sizes: int32 [world] (rows per dst)

    """
    assert tokens.is_cuda, "Expect CUDA tokens"
    device = tokens.device
    dtype = tokens.dtype

    assert num_experts_total % world_size == 0, "num_experts_total must be divisible by world_size"
    e_local = num_experts_total // world_size

    # logits: [T, E]
    routed_values = F.softmax(torch.einsum('th,he->te', tokens, router), dim=-1)
    _, top_idxs = torch.topk(routed_values, topk, dim=-1)  # [T, topk]

    # Build per-(dst, expert_local) buckets (Python lists for determinism)
    # bucket[dst][e_local] is a Python list of 1D tensors [H]
    buckets: List[List[List[torch.Tensor]]] = [
        [[] for _ in range(e_local)] for _ in range(world_size)
    ]

    # Stable append order: increasing token index, then increasing top-k slot
    
    for ti in range(tokens.shape[0]):
        tok = tokens[ti]
        for kk in range(topk):
            ex = int(top_idxs[ti, kk].item())
            dst = ex // e_local
            el = ex - dst * e_local
            buckets[dst][el].append(tok)

    # Build counts and payload
    send_counts = torch.zeros((world_size, e_local), dtype=torch.int32, device=device)
    payload_chunks: List[torch.Tensor] = []
    dst_offsets = torch.zeros((world_size,), dtype=torch.int32, device=device)
    dst_sizes = torch.zeros((world_size,), dtype=torch.int32, device=device)

    running = 0
    for dst in range(world_size):
        dst_offsets[dst] = running
        dst_rows = 0
        for el in range(e_local):
            n = len(buckets[dst][el])
            send_counts[dst, el] = n
            if n > 0:
                payload_chunks.append(torch.stack(buckets[dst][el], dim=0))
            dst_rows += n
        dst_sizes[dst] = dst_rows
        running += dst_rows

    if running == 0:
        # Degenerate: no tokens routed (possible if batch*seq==0). Create empty payload
        send_payload = torch.empty((0, tokens.shape[1]), dtype=dtype, device=device)
    else:
        send_payload = torch.cat(payload_chunks, dim=0)

    return send_payload, send_counts, dst_offsets, dst_sizes

def gen_local_weights(
    e_local: int,
    hidden_dim: int,
    out_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    base_seed: int,
    rank: int,
) -> torch.Tensor:
    # Make weights rank-dependent but deterministic.
    set_seed(base_seed + 12345, rank)
    w = torch.empty((e_local, hidden_dim, out_dim), dtype=dtype, device=device)
    torch.nn.init.normal_(w, mean=0.0, std=0.02)
    return w

