from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

import iris

"""
world = DeviceCount E = E_local = T 
H = hidden_dimension 
CAP = max_{e,s} pca[e,s] 
Capcity in real moe 
"""
@triton.jit
# inline_asm for time measurements of all the atmoic instructions
def get_amdgpu_clock():
    # use s_memrealtime to get realtime AMD GPU (64-bit)
    # s_waitcnt lgkmcnt(0) to ennsure all memory operations are end before read
    return tl.inline_asm_elementwise(
        "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)",
        "=s", # output scalar register
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1
    )

################################################################
## These Kernels implement a scatter approach (pushing the data.)
###############################################################

# Step-1 kernel: counts exchange
@triton.jit
def counts_exchange_kernel(
    send_counts_ptr,  # [world, E] int32 (local)
    pca_ptr,  # [E, world] int32 (symmetric on dst)
    counts_ready_ptr,  # [1] int32 (symmetric on dst)
    heap_bases,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Write counts to each dst's PCA[:, src_rank], then signal counts_ready++ on dst."""

    dst = tl.program_id(0)  # one program per destination rank
    
    

    # Write the E counts for this destination.
    for e0 in tl.static_range(0, e_local, BLOCK_E):
        e = e0 + tl.arange(0, BLOCK_E)
        mask_e = e < e_local

        # Remote write: pca[e, src_rank] on destination.
        src_ptr = send_counts_ptr + dst * e_local + e  # new pointer 
        remote_ptr = pca_ptr + e * world_size + src_rank 
             
        iris.put( 
            src_ptr,            # from_ptr: pointer
            remote_ptr,         # to_ptr: pointer
            from_rank=src_rank,
            to_rank=dst,
            heap_bases=heap_bases,
            mask=mask_e,        
        )


    # Signal completion to destination (release semantics).
    iris.atomic_add( 
        counts_ready_ptr,
        1,
        src_rank,
        dst,
        heap_bases,
        sem="release",
        scope="sys",
    )
    # spin wait on elocal variables that waits

# Step-2 kernel: token exchange with the original logic
@triton.jit
def tokens_exchange_kernel(
    send_ptr,            # [sum_send, H]
    send_counts_ptr,     # [W, E] int32 local
    dst_offsets_ptr,     # [W] int32 local
    expert_offs_ptr,     # [W, E] int32 local (prefix within dst block)
    token_buf_ptr,       # [E, W, CAP, H] symmetric on dst
    token_sync_ptr,      # [E] int32 symmetric on dst
    tile_counter_ptr,    # [E, W] int32 LOCAL scratch on src (we repurpose it!)
    debug_time_ptr,      # [NEW] pointer used to write back timing data [W, E]
    heap_bases,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    CAP: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grid = (dst, expert, token_id)
    dst    = tl.program_id(0)
    expert = tl.program_id(1)
    tid    = tl.program_id(2)  # token row id within this (dst, expert)

  

    # how many rows to send to (dst, expert)
    n = tl.load(send_counts_ptr + dst * e_local + expert).to(tl.int32)
    n_eff = tl.minimum(n, tl.full((), CAP, tl.int32))
    
    # If no tokens: ONE program (tid==0) sends completion signal.
    if tid == 0:
        if n_eff == 0:
            iris.atomic_add(
                token_sync_ptr + expert, 1,
                src_rank, dst, heap_bases,
                sem="release", scope="sys",
            )
            return

    # Only valid token-ids participate
    if tid >= n_eff:
        return

    # [NEW] Probes 0: record all 
    t0 = get_amdgpu_clock()

    # send row index in packed send payload
    dst_base  = tl.load(dst_offsets_ptr + dst).to(tl.int32)
    e_off     = tl.load(expert_offs_ptr + dst * e_local + expert).to(tl.int32)
    send_row  = dst_base + e_off + tid

    # remote base for this (expert, src_rank) slice on destination
    # [FIX] Cast to int64 to avoid overflow when CAP*H is large (>2GB)
    remote_base = (expert.to(tl.int64) * world_size + src_rank) * CAP * hidden_dim
    remote_row  = tid  # place at row=tid inside [CAP, H]

    # copy one token row, BLOCK_K across hidden dim
    for k0 in tl.static_range(0, hidden_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_dim

        send_ptrs   = send_ptr + send_row * hidden_dim + offs_k
        remote_ptrs = token_buf_ptr + remote_base + remote_row * hidden_dim + offs_k

        iris.put(
            send_ptrs,
            remote_ptrs,
            from_rank=src_rank,
            to_rank=dst,
            heap_bases=heap_bases,
            mask=k_mask,
        )

    # local completion accounting (ON SRC GPU)
    # tile_counter_ptr used as LOCAL scratch: tile_counter[expert, dst] counts completed tokens
    ctr_ptr = tile_counter_ptr + expert * world_size + dst
    tl.atomic_add(ctr_ptr, 1, sem="release", scope="sys")

    # ONE program spins until all tokens for this (dst, expert) done, then signals dst.token_sync[expert] += 1
    if tid == 0:
        # [NEW] probe1. record while the spin lock begin
        t1 = get_amdgpu_clock()

        v = tl.atomic_cas(ctr_ptr, n_eff, n_eff, sem='acquire', scope='sys')
        tl.debug_barrier()            
        while v != n_eff:
            v = tl.atomic_cas(ctr_ptr, n_eff, n_eff, sem='acquire', scope='sys')

        # [NEW] probe 2: all threads ends
        t2 = get_amdgpu_clock()

        iris.atomic_add(
            token_sync_ptr + expert, 1,
            src_rank, dst, heap_bases,
            sem="release", scope="sys",
        )
        
        # [NEW] probe 3: cross gpu signal sent
        t3 = get_amdgpu_clock()

       # [NEW] calculated the time cost and into the share mem
        debug_idx = (dst * e_local + expert) * 3  # *3 here record all 3 data
        tl.store(debug_time_ptr + debug_idx + 0, t1 - t0) # Phase 1: iris.put 
        tl.store(debug_time_ptr + debug_idx + 1, t2 - t1) # Phase 2: Spin-lock time
        tl.store(debug_time_ptr + debug_idx + 2, t3 - t2) # Phase 3: Remote Atomic time

## THIS KERNEL make timing realizble
@triton.jit
def receiver_wait_kernel(
    token_sync_ptr,        # [E_local] int32 symmetric/local visible
    recv_wait_cycles_ptr,  # [E_local] int64 local
    world_size: tl.constexpr,
):
    expert = tl.program_id(0)

    t0 = get_amdgpu_clock()

    ws = tl.full((), world_size, tl.int32)
    v = tl.atomic_cas(token_sync_ptr + expert, ws, ws, sem="acquire", scope="sys")
    tl.debug_barrier()
    while v != ws:
        v = tl.atomic_cas(token_sync_ptr + expert, ws, ws, sem="acquire", scope="sys")

    t1 = get_amdgpu_clock()
    tl.store(recv_wait_cycles_ptr + expert, t1 - t0)
 
################################################################
## These Kernels implement a gather approach (pulling the data.)
###############################################################

#@triton.autotune(
#    configs=[
#        triton.Config({'BLOCK_M': 16}, num_warps=2),
#        triton.Config({'BLOCK_M': 16}, num_warps=4),
#        triton.Config({'BLOCK_M': 32}, num_warps=2),
#        triton.Config({'BLOCK_M': 32}, num_warps=4),
#        triton.Config({'BLOCK_M': 64}, num_warps=2),
#        triton.Config({'BLOCK_M': 64}, num_warps=4),
#    ],
#    key=['e_local'],
#)
@triton.jit
def counts_exchange_pull(
    ############################################
    ### These two buffers are for READING ONLY. 
    ############################################
    cnts,  # [world_size, expert_pack] (symmetric memory, addressable by all devices)
    offsets, # [world_size] (symmetric memory, addressable by all devices)
    ################################################
    ### These three buffers are for Writing ONLY. 
    ################################################
    local_expert_cnts, # [world_size, expert_pack] (local memory)
    local_expert_offset_idxs, # [world_size] (local memory)
    cnt_exchange_sync, # [1] (unit-sized array in local memory)
    heap_bases,
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

    iris.get(
        cnts + src_rank * e_local + tl.arange(0, BLOCK_M),
        local_expert_cnts + dev_id * e_local + tl.arange(0, BLOCK_M),
        from_rank=src_rank,
        to_rank=dev_id,
        heap_bases=heap_bases,
        mask=tl.arange(0, BLOCK_M) < e_local
    )

    iris.get(
        offsets + src_rank,
        local_expert_offset_idxs + dev_id,
        from_rank=src_rank,
        to_rank=dev_id,
        heap_bases=heap_bases,
        mask=None
    )

    tl.atomic_add(cnt_exchange_sync, 1, sem="release", scope='sys')

    ## Spin-wait till completion, required for correctness. ##
    ws = tl.full([], world_size, dtype=tl.int32)
    v = tl.atomic_cas(cnt_exchange_sync, ws, ws, sem='acquire', scope='sys')
    tl.debug_barrier()
    while v != ws:
        v = tl.atomic_cas(cnt_exchange_sync, ws, ws, sem='acquire', scope='sys')

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
    tokens, # [C, hidden_dim] -> symmetric memory.
    read_meta, # [world_size, expert_pack] -> local memory.
    write_meta, # [world_size, expert_pack] -> local memory.
    local_expert_cnts, # [world_size, expert_pack] -> symmetric memory.
    ########################################
    ### These buffers are for WRITING ONLY. 
    ########################################
    gathered_tokens, # [C', hidden_dim] -> local memory.
    token_sync, # [world_size, expert_pack] -> local memory.
    ## Extraneous bits of information.
    heap_bases,
    src_rank,
    world_size,
    e_local,
    hidden_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
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

    read_meta_ptrs = tl.load(read_meta + dev_id * e_local + expert_id) 
    write_meta_ptrs = tl.load(write_meta + dev_id * e_local + expert_id)

    from_ptrs = read_meta_ptrs * hidden_dim + offset * hidden_dim + tl.arange(0, BLOCK_M)
    to_ptrs = write_meta_ptrs * hidden_dim + offset * hidden_dim + tl.arange(0, BLOCK_M)

    for k in tl.range(tl.cdiv(hidden_dim, BLOCK_M)):
        iris.get(
            tokens + from_ptrs,
            gathered_tokens + to_ptrs,
            from_rank=src_rank,
            to_rank=dev_id,
            heap_bases=heap_bases,
            mask=tl.arange(0, BLOCK_M)+k*BLOCK_M < hidden_dim
        )

        from_ptrs += BLOCK_M
        to_ptrs += BLOCK_M

    ## Now, we ring a bell to indicate completion. ##
    tl.atomic_add(token_sync + dev_id * e_local + expert_id, 1, sem='release', scope='sys')

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

