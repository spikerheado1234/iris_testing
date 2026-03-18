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

    # Signal Completion
    # Offset = 0 (Scalar)
    remote_ready_base = tl.load(counts_ready_bases_ptr + dst).to(tl.pointer_type(tl.int32))
    tl.atomic_add(remote_ready_base, 1, sem="release")



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
    # *,
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
    tid    = tl.program_id(2)

    n = tl.load(send_counts_ptr + dst * e_local + expert).to(tl.int32)
    n_eff = tl.minimum(n, tl.full((), CAP, tl.int32))

    # Signal 0 tokens case
    if tid == 0:
        if n_eff == 0:
            remote_sync_base = tl.load(token_sync_bases_ptr + dst).to(tl.pointer_type(tl.int32))
            # new 3d offset
            remote_sync_offset = (expert * world_size * MAX_TILES) + (src_rank * MAX_TILES) + 0
            tl.atomic_add(remote_sync_base + remote_sync_offset, 1, sem="release")
            return

    if tid >= n_eff:
        return

    dst_base  = tl.load(dst_offsets_ptr + dst).to(tl.int32)
    e_off     = tl.load(expert_offs_ptr + dst * e_local + expert).to(tl.int32)
    send_row  = dst_base + e_off + tid

    # Calculate Remote Offset (relative to Token Buffer start)
    # Layout: [E, W, CAP, H]
    # Formula: expert * (W*CAP*H) + src_rank * (CAP*H) + tid * H
    # Cast to int64 to prevent overflow
    row_offset = (expert.to(tl.int64) * world_size + src_rank) * CAP * hidden_dim + tid * hidden_dim
    
    # Load remote base address
    remote_token_base = tl.load(token_buf_bases_ptr + dst).to(tl.pointer_type(tl.bfloat16)) # Assuming bf16

    for k0 in tl.static_range(0, hidden_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_dim

        # Load Local Data
        vals = tl.load(send_ptr + send_row * hidden_dim + offs_k, mask=k_mask)
        
        # Store to Remote
        remote_ptr = remote_token_base + row_offset + offs_k
        tl.store(remote_ptr, vals, mask=k_mask)

    # calc where tile belong
    tile_id = tid // TILE_SIZE

    # calc the actual token number inside
    tile_start_tid = tile_id * TILE_SIZE
    tokens_in_this_tile = tl.minimum(TILE_SIZE, n_eff - tile_start_tid)

    # find the local conuter of the tile
    ctr_offset = (expert * world_size * MAX_TILES) + (dst * MAX_TILES) + tile_id
    ctr_ptr = tile_counter_ptr + ctr_offset
  
    # incremnt counter inside
    old_val = tl.atomic_add(ctr_ptr, 1, sem="release")

    if old_val == tokens_in_this_tile - 1:
        
        # this one is optional,make sure all the store from the block are visible
        tl.debug_barrier() 

        remote_sync_base = tl.load(token_sync_bases_ptr + dst).to(tl.pointer_type(tl.int32))
        
        # to：expert，sender src_rank，which tile_id 
        remote_sync_offset = (expert * world_size * MAX_TILES) + (src_rank * MAX_TILES) + tile_id
        
        tl.atomic_add(remote_sync_base + remote_sync_offset, 1, sem="release")
"""
@triton.jit
def token_shuffle(
    pca_cumsum_ptr, pca_ptr, ## Both of size: [E, world_size]
    token_buffer_ptr, # Size: [E, world_size, mxa, hidden_dim]
    output_buffer_ptr, # Size: [S, hidden_dim]
    token_sync_ptr, # Size: [E] int32 
    E: tl.constexpr, world_size: tl.constexpr,
    mxa: tl.constexpr, hidden_dim: tl.constexpr
    BLOCK_X: tl.contexpr  ## We have 1-d blocks only.
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

    ## We predicate some blocks off immediately. ##
    num_tokens = tl.load(pca_ptr + (expert * world_size + dev_id).to(tl.int64))

    if num_tokens < token_id:
        return ## Immediately terminate.

    ## Else, we have a non-zero token to shift to the output buffer. ##

    ## We have to wait on the prior shmem puts to finish successfuly. 
    while tl.load(token_sync_ptr + expert.to(tl.int64), volatile=True) < world_size:
        pass 
    
    ## We loop over the hidden dimension and shift a token over to the output buffer. 
    inp_ptrs = expert * world_size * mxa * hidden_dim + dev_id *  mxa * hidden_dim + token_id * hidden_dim + tl.arange(BLOCK_X)
    cum_summed_prev = tl.load(pca_cumsum_ptr + (expert * world_size + dev_id).to(tl.int64))
    packed_ptrs = (cum_summed_prev + token_id) * hidden_dim
    for _ in tl.range(tl.cdiv(hidden_dim, BLOCK_X)):

        tkns = tl.load(token_buffer_ptr + inp_ptrs, mask=0)
        tl.store(output_buffer_ptr + packed_ptrs, tkns)

        packed_ptrs += BLOCK_X
        inp_ptrs += BLOCK_X

@triton.jit
def grouped_gemm(
    # device tensor of matrices pointers
    token_ptrs, # Shape: [S, hidden_dim].
    expert_weights, # Shape: [hidden_dim, expert_hidden_dim]
    output_ptrs, # Shape: [S, expert_hidden_dim]
    expert_tkn_cnt_ptr, # Shape: [E], token count per expert.
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # number of gemms -> equivalent to local expert count.
    expert_cnt: tl.contexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.contexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    #This kernel implements a grouped-gemm on the input token-buffers.

    #Args:
    #    token_ptrs (Tensor): [S, hidden_dim]-sized array. S is the packed
    #        number of tokens (no zero-padding) post data-shuffling.
    #    expert_weights (Tensor): [hidden_dim, expert_hidden_dim]-sized array. 
    #        This represents each experts' weights.
    #    output_ptrs (Tensor): [S, expert_hidden_dim]-sized array. Buffer to store
    #        the results of processing the input tokens with expert weights.
    #    expert_tkn_cnt_ptr (Tensor): [E]-sized array representing the tokens routed 
    #        to expert i on the current rank. 
        
    #    Rest of the arguments are self-explanatory.
"""
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(expert_cnt):
        # get the gemm size of the current problem
        gm = tl.load(expert_tkn_cnt_ptr + g)
        gn = expert_hidden_dim
        gk = hidden_dim
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = token_ptrs + offs_am[:, None] * hidden_dim + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * expert_hidden_dim + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(output_ptrs.dtype.element_ty)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + expert_hidden_dim * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles
    """

