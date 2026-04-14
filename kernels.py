from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

"""
world = DeviceCount E = E_local = T 
H = hidden_dimension 
CAP = max_{e,s} pca[e,s] 
Capcity in real moe 
"""



@triton.jit
def counts_exchange_kernel(
    send_counts_ptr,  
    pca_ptr,          
    counts_ready_ptr, 
    # New arguments: Base pointer lists
    pca_bases_ptr,         # [World] Ptr to PCA buffer start on each rank
    counts_ready_bases_ptr,# [World] Ptr to Ready buffer start on each rank
    # *,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    dst = tl.program_id(0)
    
    for e0 in tl.static_range(0, e_local, BLOCK_E):
        e = e0 + tl.arange(0, BLOCK_E)
        mask_e = e < e_local
        
        # 1. Load Local Counts
        vals = tl.load(send_counts_ptr + dst * e_local + e, mask=mask_e, other=0)
        
        # 2. Store Remote (Native Put)
        # Logical position in remote PCA: pca[e, src_rank]
        # Offset calculation: (e * world_size + src_rank)
        offset = (e * world_size + src_rank)
        
        # Load remote base address for the destination rank
        remote_base = tl.load(pca_bases_ptr + dst).to(tl.pointer_type(tl.int32))
        remote_ptr = remote_base + offset
        tl.store(remote_ptr, vals, mask=mask_e)
        
    tl.debug_barrier() 

    # 3. Remote ocmpletion signal
    remote_ready_base = tl.load(counts_ready_bases_ptr + dst).to(tl.pointer_type(tl.int32))
    ptr_vec = remote_ready_base + tl.zeros((BLOCK_E,), dtype=tl.int32)
    mask_0 = tl.arange(0, BLOCK_E) == 0
    tl.atomic_add(ptr_vec, 1, mask=mask_0, sem="release")


@triton.jit
def tokens_exchange_kernel(
    send_ptr,            
    send_counts_ptr,     
    dst_offsets_ptr,     
    expert_offs_ptr,     
    token_buf_ptr,       
    token_sync_ptr,      
    tile_counter_ptr,    
    # New arguments: Base pointer lists
    token_buf_bases_ptr,  # [World]
    token_sync_bases_ptr, # [World]
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    CAP: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,   # new tile size
    MAX_TILES: tl.constexpr, # new max tile num
):

    dst    = tl.program_id(0)
    expert = tl.program_id(1)
    tile_id = tl.program_id(2) # RECV tile_id now

    n = tl.load(send_counts_ptr + dst * e_local + expert).to(tl.int32)
    n_eff = tl.minimum(n, tl.full((), CAP, tl.int32))

    # Signal 0 tokens case
    if tile_id == 0:
        if n_eff == 0:
            remote_sync_base = tl.load(token_sync_bases_ptr + dst).to(tl.pointer_type(tl.int32))
            remote_sync_offset = (expert * world_size * MAX_TILES) + (src_rank * MAX_TILES) + 0
            ptr_vec = remote_sync_base + remote_sync_offset + tl.zeros((BLOCK_K,), dtype=tl.int32)
            mask_0 = tl.arange(0, BLOCK_K) == 0
            tl.atomic_xchg(ptr_vec, 1, mask=mask_0, sem="release")
            return

    tile_start_tid = tile_id * TILE_SIZE
    if tile_start_tid >= n_eff:
        return

    tokens_in_this_tile = tl.minimum(TILE_SIZE, n_eff - tile_start_tid)

    dst_base  = tl.load(dst_offsets_ptr + dst).to(tl.int32)
    e_off     = tl.load(expert_offs_ptr + dst * e_local + expert).to(tl.int32)
    remote_token_base = tl.load(token_buf_bases_ptr + dst).to(tl.pointer_type(tl.bfloat16))

    # 64 tokens in a row

    for offset in range(TILE_SIZE):
        # no break support so use if here
        if offset < tokens_in_this_tile:
            tid = tile_start_tid + offset
            send_row = dst_base + e_off + tid
            row_offset = (expert.to(tl.int64) * world_size + src_rank) * CAP * hidden_dim + tid * hidden_dim
            
            for k0 in tl.static_range(0, hidden_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < hidden_dim
                vals = tl.load(send_ptr + send_row * hidden_dim + offs_k, mask=k_mask)
                remote_ptr = remote_token_base + row_offset + offs_k
                tl.store(remote_ptr, vals, mask=k_mask)

    my_counter_offset = (expert * world_size * MAX_TILES) + (dst * MAX_TILES) + tile_id
    counter_ptr_vec = tile_counter_ptr + my_counter_offset + tl.zeros((BLOCK_K,), dtype=tl.int32)
    
    # 2. Local completion accounting: each thread +1 with its old value
    # use acq_rel to ensure the last thread can see the store results from other threads
    old_val = tl.atomic_add(counter_ptr_vec, 1, sem="acq_rel")
    
    # 3. Final completion signal: he who gets BLOCK_K - 1, is the last one to finish
    mask_last = (old_val == BLOCK_K - 1)
    
    remote_sync_base = tl.load(token_sync_bases_ptr + dst).to(tl.pointer_type(tl.int32))
    remote_sync_offset = (expert * world_size * MAX_TILES) + (src_rank * MAX_TILES) + tile_id
    remote_ptr_vec = remote_sync_base + remote_sync_offset + tl.zeros((BLOCK_K,), dtype=tl.int32)
    
    # only the last thread will emit the final remote unlock signal, releasing all threads at once!
    tl.atomic_xchg(remote_ptr_vec, 1, mask=mask_last, sem="release")

@triton.jit
def token_shuffle(
    pca_cumsum_ptr, pca_ptr, 
    token_buffer_ptr, 
    output_buffer_ptr, 
    token_sync_ptr,           # now to the 3D matrix [E, W, MAX_TILES]
    E: tl.constexpr, world_size: tl.constexpr,
    mxa: tl.constexpr, hidden_dim: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TILE_SIZE: tl.constexpr,  # new size of tile 64
    MAX_TILES: tl.constexpr   # new max tile para
):
    """
    #Triton kernel that reshuffles data post all-to-all (prior to expert compute) 
    #to eliminate zero-padding.

    #Args:
    #    pca_cumsum_ptr (Tensor): [E, world_size]-sized physical counts array. 
    #        pca_cumsum_ptr[i, j] = x means x tokens 
    #    pca_ptr (Tensor): [E, world_size]-sized physical counts array.
    #        pca_ptr[i, j] = x represents that x tokens are routed from device j
    #        to expert i on the current rank.
    #    token_buffer_ptr (Tensor): [E, world_size, capacity, hidden_dim]-sized tensor.
    #        the output buffer that the all-to-all writes to.
    #    token_sync_ptr (Tensor): [E]-sized tensor. These are synchronization variables
    #        set by the prior all-to-all to ensure correctness.
    #    E (int): number of *local* experts.
    #    world_size (int): number of participating ranks.
    #    mxa (int): maximum capaicty (2nd dimension of the token_buffer_ptr array).
    #    hidden_dim (int): token hidden-dimensions.
"""
   
    expert = tl.program_id(0)
    dev_id = tl.program_id(1)
    token_id = tl.program_id(2)

    num_tokens = tl.load(pca_ptr + (expert * world_size + dev_id).to(tl.int64))

    if num_tokens <= token_id:
        return 

    # 3D super fast Spin-lock
   
    tile_id = token_id // TILE_SIZE
    sync_offset = (expert * world_size * MAX_TILES) + (dev_id * MAX_TILES) + tile_id
   
    
    
    offs = tl.arange(0, BLOCK_X)
    inp_ptrs = expert * world_size * mxa * hidden_dim + dev_id * mxa * hidden_dim + token_id * hidden_dim + offs
    cum_summed_prev = tl.load(pca_cumsum_ptr + (expert * world_size + dev_id).to(tl.int64))
    
    # with + offs，make it same as tkns 
    packed_ptrs = (cum_summed_prev + token_id) * hidden_dim + offs
    
    for i in tl.range(0, tl.cdiv(hidden_dim, BLOCK_X)):
        # safer mask
        mask = (i * BLOCK_X + offs) < hidden_dim
        tkns = tl.load(token_buffer_ptr + inp_ptrs, mask=mask)
        tl.store(output_buffer_ptr + packed_ptrs, tkns, mask=mask)
        
        packed_ptrs += BLOCK_X
        inp_ptrs += BLOCK_X


@triton.jit
def counts_exchange_pull(
    ############################################
    ### These two buffers are for READING ONLY. 
    ############################################
    cnts_bases,  # [world_size] (symmetric memory, addressable by all devices) int64
    offsets_bases,   # [world_size] (symmetric memory, addressable by all devices) int64
    ################################################
    ### These three buffers are for Writing ONLY. 
    ################################################
    local_expert_cnts, # [world_size, expert_pack] (local memory) int64
    local_expert_offset_idxs, # [world_size] (local memory) int64
    cnt_exchange_sync, # [1] (unit-sized array in local memory)  int32 
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    This is the metadata exchange kernel. At the end of this kernel,
    two important tensors are populated. 

    1. local_expert_cnts -> [world_size, expert_pack] sized. local_expert_cnts[i, j] = x
    indicates that device i will x tokens to this rank's local expert j.
    2. local_expert_offset_idxs -> [world_size] sized. local_expert_offset_idxs[i] = x indicates
    that device i's first token to be routed to this rank resides at index x.

    Invariant: prior to launching this kernel, cnts and offsets must be populated by each device.
    Moreover, the tokens should be reordered to be contiguous (with respect to experts). This means
    that the tokens on each device should look like the following:
    [
    ---- token routed to expert 0 ----,
    ---- token routed to expert 0 ----,
    ---- token routed to expert 0 ----,
    ---- token routed to expert 1 ----,
    ---- token routed to expert 1 ----,
    ---- token routed to expert 1 ----,
    ...
    ---- token routed to expert k ----
    ]
    """
    dev_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK_M)
    mask = offs < e_local

    remote_cnts_base_i64 = tl.load(cnts_bases + dev_id)
    remote_cnts_base = tl.cast(remote_cnts_base_i64, tl.pointer_type(tl.int64))
    remote_cnts_ptrs = remote_cnts_base + src_rank * e_local + offs
    vals = tl.load(remote_cnts_ptrs, mask=mask, other=0)
    tl.store(local_expert_cnts + dev_id * e_local + offs, vals, mask=mask)

    remote_offsets_base_i64 = tl.load(offsets_bases + dev_id)
    remote_offsets_base = tl.cast(remote_offsets_base_i64, tl.pointer_type(tl.int64))
    off_val = tl.load(remote_offsets_base + src_rank)
    tl.store(local_expert_offset_idxs + dev_id, off_val)
    # maybe we dont need the lock here
    #tl.atomic_add(cnt_exchange_sync, 1, sem="release", scope="sys")

    #ws = tl.full((), world_size, tl.int32)
    #v = tl.atomic_cas(cnt_exchange_sync, ws, ws, sem="acquire", scope="sys")
    #while v != ws:
    #   v = tl.atomic_cas(cnt_exchange_sync, ws, ws, sem="acquire", scope="sys")

## TODO(ahangupta): if this consumes too much runtime, we can migrate this into
##                      a triton kernel as well.
@torch.compile
def roll_cum_sum(cnts, local_expert_offset_idxs):
    """
    This produces read/write metadata that we can use to index token tensors.
    This will be called in between the metadata-kernel and the physical token 
    exchange kernel.

    Inputs:
        1. cnts -> [world_size, expert_pack]. cnts[i, j] = x on device d indicates that
                        device i will route x tokens to expert j that resides on d.
        2. local_expert_offset_idxs -> [world_size]. local_expert_offset_idxs[i] = x on device d
                                            indicates that the first token to be routed to d from device
                                            i resides at index x.

    Outputs:
        1. read_meta -> [world_size, expert_pack]. read_meta[i, j] = x indicates that in device
                            i's token buffer, the first token routed to expert j on this device lies at index x.
        2. write_meta -> [world_size, expert_pack]. write_meta[i, j] = x indicates that in this device's 
                            gathered_token buffer, the first token routed to expert j from device i should be written to 
                            at index x.

    """
    ## This produces the write_meta array. ##
    cnts_transpose = torch.transpose(cnts, 0, 1)
    prev_shape = cnts_transpose.shape
    write_meta = torch.transpose(torch.roll(torch.cumsum(cnts_transpose.reshape(-1), dim=0), 1).reshape(prev_shape), 0, 1).contiguous()
    write_meta[0, 0] = 0  ## Since we start with 0. 

    ## This produces the read_meta array. ##
    read_meta = torch.roll(torch.cumsum(cnts, dim=-1) + local_expert_offset_idxs[:, None], 1, dims=-1)
    read_meta[:, 0] = local_expert_offset_idxs

    return read_meta, write_meta

#@triton.autotune(
#    configs=[
#        triton.Config({'BLOCK_M': 32},  num_warps=2),
#        triton.Config({'BLOCK_M': 32},  num_warps=4),
#        triton.Config({'BLOCK_M': 32},  num_warps=8),
#        triton.Config({'BLOCK_M': 64},  num_warps=2),
#        triton.Config({'BLOCK_M': 64},  num_warps=4),
#        triton.Config({'BLOCK_M': 64},  num_warps=8),
#        triton.Config({'BLOCK_M': 128}, num_warps=2),
#        triton.Config({'BLOCK_M': 128}, num_warps=4),
#        triton.Config({'BLOCK_M': 128}, num_warps=8),
#        triton.Config({'BLOCK_M': 256}, num_warps=2),
#        triton.Config({'BLOCK_M': 256}, num_warps=4),
#        triton.Config({'BLOCK_M': 256}, num_warps=8),
#    ],
#    key=['hidden_dim'],
#)
@triton.jit
def token_exchange_pull(
    ########################################
    ### These buffers are for READING ONLY. 
    ########################################
    tokens_bases,    # [world_size] -> symmetric memory. int64
    read_meta, # [world_size, expert_pack] -> local memory. int64
    write_meta, # [world_size, expert_pack] -> local memory. int64
    local_expert_cnts, # [world_size, expert_pack] -> local memory. int64
    ########################################
    ### These buffers are for WRITING ONLY. 
    ########################################
    gathered_tokens, # [T_recv, hidden_dim] -> local memory.
    token_sync, # [world_size, expert_pack] -> local memory. int32
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    This kernel physically exchanges tokens after the metadata exchange stage.

    It implements a "pull" a2a kernel, where device i gathers all its relevant tokens
    from the other devices (rather than scattering its own tokens).
    """
    dev_id = tl.program_id(0)
    expert_id = tl.program_id(1)
    
    ## This determines which token should move in shmem. ##
    offset = tl.program_id(2)

    if offset >= tl.load(local_expert_cnts + dev_id * e_local + expert_id):
        ## Selectively predicate all blocks that are not required and
        ##    over-provisioned.
        return 

    read_row = tl.load(read_meta + dev_id * e_local + expert_id) + offset
    write_row = tl.load(write_meta + dev_id * e_local + expert_id) + offset

    remote_tokens_base_i64 = tl.load(tokens_bases + dev_id)
    remote_tokens_base = tl.cast(remote_tokens_base_i64, tl.pointer_type(tl.bfloat16))

    for k0 in tl.static_range(0, hidden_dim, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask = k < hidden_dim

        src_ptrs = remote_tokens_base + read_row * hidden_dim + k
        dst_ptrs = gathered_tokens + write_row * hidden_dim + k

        vals = tl.load(src_ptrs, mask=mask, other=0)
        tl.store(dst_ptrs, vals, mask=mask)

   
    ## Now, we ring a bell to indicate completion. ##
    #tl.atomic_add(token_sync + dev_id * e_local + expert_id, 1, sem='release', scope='sys')

@triton.jit
def grouped_gemm(
    token_ptrs,            # [S, H]
    expert_weights_ptr,    # [E, H, N]
    output_ptrs,           # [S, N]
    expert_tkn_cnt_ptr,    # [E]
    expert_offsets_ptr,    # [E]
    pid_to_expert_ptr,     # [total_tiles]
    pid_to_m_ptr,          # [total_tiles]
    pid_to_n_ptr,          # [total_tiles]
    NUM_SM: tl.constexpr,  # unused for now
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # O(1) schedule look up
    g = tl.load(pid_to_expert_ptr + pid).to(tl.int32)
    tile_m_idx = tl.load(pid_to_m_ptr + pid).to(tl.int32)
    tile_n_idx = tl.load(pid_to_n_ptr + pid).to(tl.int32)

    gm = tl.load(expert_tkn_cnt_ptr + g).to(tl.int64)
    token_base = tl.load(expert_offsets_ptr + g).to(tl.int64)

    offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, hidden_dim, BLOCK_SIZE_K):
        k_idx = k0 + offs_k

        a_ptrs = token_ptrs + (token_base + offs_m)[:, None] * hidden_dim + k_idx[None, :]
        b_ptrs = (
            expert_weights_ptr
            + g * hidden_dim * expert_hidden_dim
            + k_idx[:, None] * expert_hidden_dim
            + offs_n[None, :]
        )

        a_mask = (offs_m[:, None] < gm) & (k_idx[None, :] < hidden_dim)
        b_mask = (k_idx[:, None] < hidden_dim) & (offs_n[None, :] < expert_hidden_dim)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c = acc.to(output_ptrs.dtype.element_ty)

    c_ptrs = output_ptrs + (token_base + offs_m)[:, None] * expert_hidden_dim + offs_n[None, :]
    c_mask = (offs_m[:, None] < gm) & (offs_n[None, :] < expert_hidden_dim)
    tl.store(c_ptrs, c, mask=c_mask)