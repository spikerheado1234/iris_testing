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
):
    dst    = tl.program_id(0)
    expert = tl.program_id(1)
    tid    = tl.program_id(2)

    n = tl.load(send_counts_ptr + dst * e_local + expert).to(tl.int32)
    n_eff = tl.minimum(n, tl.full((), CAP, tl.int32))

    # Signal 0 tokens case
    if tid == 0:
        if n_eff == 0:
            # Native Atomic Add to Remote
            remote_sync_base = tl.load(token_sync_bases_ptr + dst).to(tl.pointer_type(tl.int32))
            # Offset = expert index
            tl.atomic_add(remote_sync_base + expert, 1, sem="release")
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

    # Local Counter Logic (Unchanged, local scratchpad)
    ctr_ptr = tile_counter_ptr + expert * world_size + dst
    tl.atomic_add(ctr_ptr, 1, sem="release")

    if tid == 0:
        # Spin Wait (Local atomic CAS)
        v = tl.atomic_cas(ctr_ptr, n_eff, n_eff, sem='acquire')
        while v != n_eff:
            v = tl.atomic_cas(ctr_ptr, n_eff, n_eff, sem='acquire')

        # Remote Signal
        remote_sync_base = tl.load(token_sync_bases_ptr + dst).to(tl.pointer_type(tl.int32))
        tl.atomic_add(remote_sync_base + expert, 1, sem="release")
  

@triton.jit
def token_shuffle_kernel(
    pca_offsets_ptr, pca_ptr, 
    token_buffer_ptr, 
    output_buffer_ptr, 
    E: tl.constexpr, world_size: tl.constexpr,
    mxa: tl.constexpr, hidden_dim: tl.constexpr,
    BLOCK_X: tl.constexpr
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

    # 1D Flatten index for pca arrays
    idx = expert * world_size + dev_id
    num_tokens = tl.load(pca_ptr + idx)

    # Predicate off immediately if this thread handles empty padding
    if token_id >= num_tokens:
        return 

    # Fetch global starting offset directly from the exclusive cumsum passed from CPU
    base_out_row = tl.load(pca_offsets_ptr + idx)
    target_row = base_out_row + token_id

    # Calculate input and output pointer offsets
    inp_row_offset = (expert * world_size * mxa + dev_id * mxa + token_id) * hidden_dim
    out_row_offset = target_row * hidden_dim

    for k0 in tl.static_range(0, hidden_dim, BLOCK_X):
        offs = k0 + tl.arange(0, BLOCK_X)
        mask = offs < hidden_dim

        tkns = tl.load(token_buffer_ptr + inp_row_offset + offs, mask=mask)
        tl.store(output_buffer_ptr + out_row_offset + offs, tkns, mask=mask)

@triton.jit
def grouped_gemm(
    token_ptrs,          # Shape: [S, hidden_dim]
    expert_weights,      # Shape: [E_local, hidden_dim, expert_hidden_dim]
    output_ptrs,         # Shape: [S, expert_hidden_dim]
    expert_tkn_cnt_ptr,  # Shape: [E_local]
    NUM_SM: tl.constexpr,
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
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
        gm = tl.load(expert_tkn_cnt_ptr + g)
        gn = expert_hidden_dim
        gk = hidden_dim
        
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        
        while (tile_idx >= last_problem_end) and (tile_idx < last_problem_end + num_tiles):
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            
            # Fix: Handle 3D addressing for expert_weights (incorporating expert dimension g)
            weight_base_offset = g * hidden_dim * expert_hidden_dim

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            for k_step in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                k_offset = k_step * BLOCK_SIZE_K
                current_offs_k = k_offset + offs_k

                # Mask handling to prevent out-of-bounds memory access
                mask_a = (offs_am[:, None] < gm) & (current_offs_k[None, :] < gk)
                mask_b = (current_offs_k[:, None] < gk) & (offs_bn[None, :] < gn)

                a_ptrs = token_ptrs + offs_am[:, None] * hidden_dim + current_offs_k[None, :]
                b_ptrs = expert_weights + weight_base_offset + current_offs_k[:, None] * expert_hidden_dim + offs_bn[None, :]

                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                
                accumulator += tl.dot(a, b)

            c = accumulator.to(output_ptrs.dtype.element_ty)
            
            # Compute the global starting offset for tokens (needs accumulation of previous experts' token counts).
            # Since looping accumulation in Triton is cumbersome, passing prefix sums from CPU is recommended; 
            # simplifying here assuming a compact layout block.
            # This part might need fine-tuning based on your exact token routing logic.
            
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            mask_c = (offs_cm[:, None] < gm) & (offs_bn[None, :] < gn)
            c_ptrs = output_ptrs + offs_cm[:, None] * expert_hidden_dim + offs_bn[None, :]
            
            tl.store(c_ptrs, c, mask=mask_c)

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles
