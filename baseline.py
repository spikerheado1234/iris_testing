from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.distributed as dist

# Utilities
def _assert_cuda(x: torch.Tensor, name: str):
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor (got device={x.device}).")

def _assert_int32(x: torch.Tensor, name: str):
    if x.dtype != torch.int32:
        raise ValueError(f"{name} must be int32 (got {x.dtype}).")

def _cuda_event_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end

def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))



# Persistent buffers
@dataclass
class BaselineBuffers:
    # Counts buffers
    recv_counts: torch.Tensor            # [world_size, e_local] int32
    recv_counts_flat: torch.Tensor       # [world_size * e_local] int32
    out_splits: torch.Tensor             # [world_size] int32, sum over experts per src

    # Token buffers (flat payload)
    recv_payload_flat: torch.Tensor      # [total_recv, hidden_dim] dtype=token_dtype

    # Optional "final layout" buffer for correctness checks
    token_buf: Optional[torch.Tensor]    # [e_local, world_size, capacity, hidden_dim] or None


def init_baseline_buffers(
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
    device: torch.device,
    total_recv: int,
    allocate_token_buf: bool = False,
) -> BaselineBuffers:
    """
    Allocate persistent buffers once, outside the timing loop.
    total_recv = sum over src of (sum_e recv_counts[src, e]) for the current iteration.
    """
    recv_counts = torch.empty((world_size, e_local), device=device, dtype=torch.int32)
    recv_counts_flat = torch.empty((world_size * e_local,), device=device, dtype=torch.int32)
    out_splits = torch.empty((world_size,), device=device, dtype=torch.int32)

    recv_payload_flat = torch.empty((total_recv, hidden_dim), device=device, dtype=token_dtype)

    token_buf = None
    if allocate_token_buf:
        token_buf = torch.empty((e_local, world_size, capacity, hidden_dim), device=device, dtype=token_dtype)

    return BaselineBuffers(
        recv_counts=recv_counts,
        recv_counts_flat=recv_counts_flat,
        out_splits=out_splits,
        recv_payload_flat=recv_payload_flat,
        token_buf=token_buf,
    )



# Step-1 counts exchange (reference)
def exchange_counts_a2a(
    send_counts: torch.Tensor,
    buffers: BaselineBuffers,
    strict_capacity: bool,
    capacity: int,
) -> torch.Tensor:
    """
    send_counts: [world_size, e_local] int32
      send_counts[dst, e] = number of tokens this rank will send to dst's local expert e.

    returns:
      recv_counts: [world_size, e_local] int32
      recv_counts[src, e] = number of tokens src sends to *my* local expert e.
    """
    _assert_cuda(send_counts, "send_counts")
    _assert_int32(send_counts, "send_counts")

    world_size, e_local = send_counts.shape
    # Flatten as [dst-major, e-major] for a2a.
    send_flat = send_counts.contiguous().view(-1)  # [world_size * e_local]

    # Each rank sends exactly e_local int32s to each peer for counts exchange.
    in_splits_list  = [int(e_local)] * int(world_size)
    out_splits_list = [int(e_local)] * int(world_size)
    dist.all_to_all_single(
        buffers.recv_counts_flat,
        send_flat,
        output_split_sizes=out_splits_list,
        input_split_sizes=in_splits_list,
    )

    buffers.recv_counts.copy_(buffers.recv_counts_flat.view(world_size, e_local))

    if strict_capacity:
        mx = int(buffers.recv_counts.max().item())
        if mx > capacity:
            raise ValueError(f"[baseline_step1] recv_counts max={mx} exceeds capacity={capacity} (strict_capacity=True).")

    return buffers.recv_counts


# Step-2 token exchange (reference)
def exchange_payload_a2a(
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    buffers: BaselineBuffers,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    send_payload: [sum_dst sum_e send_counts[dst,e], hidden_dim], packed by dst then expert (must match your packer)
    send_counts:  [world_size, e_local] int32
    recv_counts:  [world_size, e_local] int32

    returns:
      recv_payload_flat: [total_recv, hidden_dim]
      in_splits:  [world_size] int32 (#rows sent to each dst)
      out_splits: [world_size] int32 (#rows received from each src)
    """
    _assert_cuda(send_payload, "send_payload")
    _assert_cuda(send_counts, "send_counts")
    _assert_cuda(recv_counts, "recv_counts")
    _assert_int32(send_counts, "send_counts")
    _assert_int32(recv_counts, "recv_counts")

    world_size, e_local = send_counts.shape

    # input split sizes: how many rows we send to each dst = sum_e send_counts[dst,e]
    in_splits = send_counts.sum(dim=1).to(torch.int32).contiguous()  # [world_size]
    total_send = int(in_splits.sum().item())
    if send_payload.shape[0] != total_send:
        raise ValueError(f"send_payload rows={send_payload.shape[0]} != total_send={total_send} derived from send_counts.")

    # output split sizes: how many rows we receive from each src = sum_e recv_counts[src,e]
    buffers.out_splits.copy_(recv_counts.sum(dim=1).to(torch.int32))
    out_splits = buffers.out_splits.contiguous()
    total_recv = int(out_splits.sum().item())

    # Ensure recv buffer capacity
    if buffers.recv_payload_flat.shape[0] != total_recv or buffers.recv_payload_flat.shape[1] != send_payload.shape[1]:
        # Reallocate ONLY when needed (outside timing ideally)
        buffers.recv_payload_flat = torch.empty((total_recv, send_payload.shape[1]),
                                               device=send_payload.device,
                                               dtype=send_payload.dtype)
    in_splits_list  = in_splits.to("cpu", non_blocking=False).to(torch.int64).tolist()
    out_splits_list = out_splits.to("cpu", non_blocking=False).to(torch.int64).tolist()


    dist.all_to_all_single(
        buffers.recv_payload_flat,
        send_payload,
        output_split_sizes=out_splits_list,
        input_split_sizes=in_splits_list,
    )

    return buffers.recv_payload_flat, in_splits, out_splits



# Optional reorder for correctness 
def reorder_flat_to_token_buf(
    recv_payload_flat: torch.Tensor,
    recv_counts: torch.Tensor,
    capacity: int,
    token_buf_out: torch.Tensor,
) -> torch.Tensor:
    """
    token_buf_out: [e_local, world, capacity, hidden_dim]
    Layout matches your IRIS buffers.token_buf contract.

    This is correctness scaffolding.
    It is intentionally written to be clear rather than fastest.
    Keep it OUTSIDE your comm-timing numbers.
    """
    _assert_cuda(recv_payload_flat, "recv_payload_flat")
    _assert_cuda(recv_counts, "recv_counts")
    _assert_int32(recv_counts, "recv_counts")

    world_size, e_local = recv_counts.shape
    hidden_dim = recv_payload_flat.shape[1]

    if token_buf_out.shape != (e_local, world_size, capacity, hidden_dim):
        raise ValueError(f"token_buf_out shape {tuple(token_buf_out.shape)} mismatch expected "
                         f"({e_local},{world_size},{capacity},{hidden_dim}).")

    # Fill with zeros for deterministic correctness
    token_buf_out.zero_()

    # recv_payload_flat is assumed packed by src rank, then by local expert (same as your kernel contract)
    # For each src:
    #   take blocks for expert 0..E-1 of sizes recv_counts[src,e]
    off = 0
    for src in range(world_size):
        for e in range(e_local):
            c = int(recv_counts[src, e].item())
            if c == 0:
                continue
            if c > capacity:
                raise ValueError(f"[reorder] recv_count src={src},e={e} is {c} > capacity={capacity}")
            token_buf_out[e, src, :c, :] = recv_payload_flat[off:off+c, :]
            off += c

    if off != recv_payload_flat.shape[0]:
        raise RuntimeError(f"[reorder] consumed rows={off} != total_recv={recv_payload_flat.shape[0]}")
    return token_buf_out



# One-call runner (timed segments)
def run_baseline_ref(
    rank: int,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    buffers: BaselineBuffers,
    do_reorder: bool = False,
    profile: bool = True,          # no longer use
    strict_capacity: bool = True,
    barrier: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Returns:
      token_buf (if do_reorder and buffers.token_buf is allocated) else None,
      timings dict: step1_ms, step2_ms, reorder_ms (if enabled)
      meta dict: recv_counts, in_splits, out_splits, total_send, total_recv
    """
    _assert_cuda(send_payload, "send_payload")
    _assert_cuda(send_counts, "send_counts")
    _assert_int32(send_counts, "send_counts")

    if send_counts.shape != (world_size, e_local):
        raise ValueError(f"send_counts shape {tuple(send_counts.shape)} != ({world_size},{e_local})")

    t: Dict[str, float] = {}
    meta: Dict[str, torch.Tensor] = {}

    # Step1 counts
    if barrier:
        dist.barrier()
    torch.cuda.synchronize()

    s1, e1 = _cuda_event_timer()
    s1.record()
    recv_counts = exchange_counts_a2a(send_counts, buffers, strict_capacity=strict_capacity, capacity=capacity)
    e1.record()
    t["step1_ms"] = _elapsed_ms(s1, e1)
    meta["recv_counts"] = recv_counts

    # Step2 tokens
    if barrier:
        dist.barrier()
    torch.cuda.synchronize()

    s2, e2 = _cuda_event_timer()
    s2.record()
    recv_payload_flat, in_splits, out_splits = exchange_payload_a2a(
        send_payload=send_payload,
        send_counts=send_counts,
        recv_counts=recv_counts,
        buffers=buffers,
    )
    e2.record()
    t["step2_ms"] = _elapsed_ms(s2, e2)

    meta["in_splits"] = in_splits
    meta["out_splits"] = out_splits
    meta["total_send"] = in_splits.sum()
    meta["total_recv"] = out_splits.sum()

    # Optional reorder (correctness scaffold)
    token_buf = None
    if do_reorder:
        if buffers.token_buf is None:
            raise ValueError("do_reorder=True but buffers.token_buf is None. Allocate with allocate_token_buf=True.")

        torch.cuda.synchronize()
        s3, e3 = _cuda_event_timer()
        s3.record()
        token_buf = reorder_flat_to_token_buf(recv_payload_flat, recv_counts, capacity, buffers.token_buf)
        e3.record()
        t["reorder_ms"] = _elapsed_ms(s3, e3)
    else:
        t["reorder_ms"] = 0.0

    return token_buf, t, meta
