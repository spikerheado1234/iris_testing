from __future__ import annotations
import math
import random
from typing import Tuple, List

import torch
import torch.nn.functional as F
import torch.distributed as dist

from dataclasses import dataclass


def nvtx_push(msg):
    torch.cuda.nvtx.range_push(msg)

def nvtx_pop():
    torch.cuda.nvtx.range_pop()

# judging and checking cap
def _allreduce_max_i32(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def _sync_and_check(ok_flag: torch.Tensor) -> int:
    ok_global = _allreduce_min_i32(ok_flag).item()
    if ok_global == 1:
        dist.barrier()
    return ok_global

def _allreduce_min_i32(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.MIN)
    return y

def _assert_cuda_int32(x: torch.Tensor, name: str) -> None:
    
    assert x.is_cuda, f"{name} must be CUDA"
    assert x.dtype == torch.int32, f"{name} must be int32"



def _build_dst_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """dst_offsets[dst] = prefix sum of total tokens to earlier destinations."""
    # send_counts: [world, E_local] int32
    send_dst_sizes = send_counts.sum(dim=1).to(torch.int32)
    dst_offsets = (torch.cumsum(send_dst_sizes, dim=0) - send_dst_sizes).to(torch.int32)
    return dst_offsets.contiguous()


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

# Step-1/2 run wrapper
def build_expert_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """
    Prefix offsets within each destination block in send_payload.

    send_counts: [world, E] int32
    returns: expert_offs[dst, e] (row offset within the dst block).
    """

    _assert_cuda_int32(send_counts, "send_counts")
    sc64 = send_counts.to(torch.int64)
    offs = (torch.cumsum(sc64, dim=1) - sc64).to(torch.int32)
    return offs.contiguous()

# NEW shmembuffers and alloc func for further timing
@dataclass
class ShmemBuffers:
    # Step-1 outputs / sync
    pca: torch.Tensor          # [E, world] int32 (symmetric)
    counts_ready: torch.Tensor # [1] int32 (symmetric)

    # Step-2 outputs / sync
    token_buf: torch.Tensor     # [E, world, CAP, H] (symmetric)
    token_sync: torch.Tensor    # [E] int32 (symmetric)
    tile_counter: torch.Tensor  # [E, world] int32 (symmetric)  
    
    # sender-side per-(dst, expert) timing: [world, E_local, 3]
    debug_time: torch.Tensor
    
    # receiver-side per-expert wait timing: [E_local]
    recv_wait_cycles: torch.Tensor

    # Cached heap bases
    heap_bases: torch.Tensor

def alloc_shmem_buffers(
    shmem,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
) -> ShmemBuffers:
    # Step-1
    pca = shmem.zeros((e_local, world_size), dtype=torch.int32, device="cuda")
    counts_ready = shmem.zeros((1,), dtype=torch.int32, device="cuda")

    # Step-2
    token_buf = shmem.zeros((e_local, world_size, capacity, hidden_dim), dtype=token_dtype, device="cuda")
    token_sync = shmem.zeros((e_local,), dtype=torch.int32, device="cuda")

    # NEW: per-(expert,src) tile counter on dst
    tile_counter = shmem.zeros((e_local, world_size), dtype=torch.int32, device="cuda")
    debug_time = torch.zeros((world_size, e_local, 3), dtype=torch.int64, device="cuda")
    
    # NEW: receiver wait cycles
    recv_wait_cycles = torch.zeros((e_local,), dtype=torch.int64, device="cuda")

    heap_bases = shmem.get_heap_bases()
    return ShmemBuffers(
        pca=pca,
        counts_ready=counts_ready,
        token_buf=token_buf,
        token_sync=token_sync,
        tile_counter=tile_counter,   
        debug_time=debug_time,
        recv_wait_cycles=recv_wait_cycles,
        heap_bases=heap_bases
    )
    
    


    




@dataclass
class CountsBuffers:
    pca: torch.Tensor
    counts_ready: torch.Tensor
    heap_bases: torch.Tensor

def reset_counts_sync(cb: CountsBuffers) -> None:
    cb.counts_ready.zero_()

def reset_token_sync(tb: TokenBuffers) -> None:
    tb.token_sync.zero_()

def reset_token_buf_debug(tb: TokenBuffers) -> None:
    tb.token_buf.zero_()

@dataclass
class TokenBuffers:
    token_buf: torch.Tensor
    token_sync: torch.Tensor
    tile_counter: torch.Tensor  # <-- NEW


def alloc_counts_buffers(shmem, world_size: int, e_local: int) -> CountsBuffers:
    pca = shmem.zeros((e_local, world_size), dtype=torch.int32, device="cuda")
    counts_ready = shmem.zeros((1,), dtype=torch.int32, device="cuda")
    heap_bases = shmem.get_heap_bases()
    return CountsBuffers(pca=pca, counts_ready=counts_ready, heap_bases=heap_bases)

def alloc_token_buffers(
    shmem,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
) -> TokenBuffers:
    token_buf = shmem.zeros((e_local, world_size, capacity, hidden_dim), dtype=token_dtype, device="cuda")
    token_sync = shmem.zeros((e_local,), dtype=torch.int32, device="cuda")
    tile_counter = shmem.zeros((e_local, world_size), dtype=torch.int32, device="cuda")  # <-- NEW
    return TokenBuffers(token_buf=token_buf, token_sync=token_sync, tile_counter=tile_counter)

def compute_capacity_from_pca(pca: torch.Tensor) -> int:
    cap = pca.max().to(torch.int32)
    dist.all_reduce(cap, op=dist.ReduceOp.MAX)
    return max(int(cap.item()), 1)



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
    
    t = tokens
    for ti in range(t.shape[0]):
        tok = t[ti]
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

def route_and_pack_fixed(
    tokens: torch.Tensor,
    router: torch.Tensor,
    topk: int,
    world_size: int,
    num_experts_total: int,
    fixed_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (Fixed-Block / Padded Routing) good for both version
    """
    assert tokens.is_cuda, "Expect CUDA tokens"
    device = tokens.device
    dtype = tokens.dtype

    assert num_experts_total % world_size == 0, "num_experts_total must be divisible by world_size"
    e_local = num_experts_total // world_size
    hidden_dim = tokens.shape[-1]

    # computing routing
    routed_values = F.softmax(torch.einsum('th,he->te', tokens, router), dim=-1)
    _, top_idxs = torch.topk(routed_values, topk, dim=-1)  # [T, topk]

    # get tokens
    buckets: List[List[List[torch.Tensor]]] = [
        [[] for _ in range(e_local)] for _ in range(world_size)
    ]
    
    for ti in range(tokens.shape[0]):
        tok = tokens[ti]
        for kk in range(topk):
            ex = int(top_idxs[ti, kk].item())
            dst = ex // e_local
            el = ex - dst * e_local
            buckets[dst][el].append(tok)

    # make fixed payload and counts
    total_rows = world_size * e_local * fixed_capacity
    send_payload = torch.zeros((total_rows, hidden_dim), dtype=dtype, device=device)
    # forced = fixed_capacity
    send_counts = torch.full((world_size, e_local), fixed_capacity, dtype=torch.int32, device=device)
    
    dst_offsets = torch.zeros((world_size,), dtype=torch.int32, device=device)
    dst_sizes = torch.full((world_size,), e_local * fixed_capacity, dtype=torch.int32, device=device)

    # into Payload (with Zero-padding)
    running = 0
    for dst in range(world_size):
        dst_offsets[dst] = running
        for el in range(e_local):
            n_actual = len(buckets[dst][el])
            n_copy = min(n_actual, fixed_capacity) # tuncate if more than cap
            if n_copy > 0:
                chunk = torch.stack(buckets[dst][el][:n_copy], dim=0)
                # calcculate start id for this block in the payload
                start_idx = running + el * fixed_capacity
                send_payload[start_idx : start_idx + n_copy] = chunk
        running += e_local * fixed_capacity

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

