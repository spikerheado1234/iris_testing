from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def grouped_gemm_dx(
    d_out_ptr,           # [S, N]
    weights_ptr,         # [E, H, N]
    d_tokens_ptr,        # [S, H]
    expert_tkn_cnt,      # [E]
    expert_offsets,      # [E]
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    max_m_tiles: tl.constexpr,
    num_h_tiles: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    tiles_per_expert = max_m_tiles * num_h_tiles
    e = pid // tiles_per_expert
    rem = pid % tiles_per_expert
    tile_m_idx = rem // num_h_tiles
    tile_h_idx = rem % num_h_tiles

    if e >= expert_cnt:
        return

    e64 = e.to(tl.int64)
    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)
    row0 = tile_m_idx * BLOCK_SIZE_M
    if row0 >= cnt:
        return

    token_base = tl.load(expert_offsets + e).to(tl.int64)
    offs_m_local = row0 + tl.arange(0, BLOCK_SIZE_M)
    offs_m = (token_base + offs_m_local).to(tl.int64)
    offs_h = tile_h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_mask = offs_m_local < cnt
    h_mask = offs_h < hidden_dim

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    n0 = 0
    while n0 < expert_hidden_dim:
        n = n0 + offs_k
        n_mask = n < expert_hidden_dim

        a_ptrs = d_out_ptr + offs_m[:, None] * expert_hidden_dim + n[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

        # W remains physically [E, H, N]. Load it as [N_tile, H_tile]
        # for dX = dY @ W.T without materializing W.T.
        b_ptrs = (
            weights_ptr
            + e64 * hidden_dim * expert_hidden_dim
            + n[:, None]
            + offs_h[None, :] * expert_hidden_dim
        )
        b = tl.load(b_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0)

        acc += tl.dot(a, b)
        n0 += BLOCK_SIZE_K

    out_ptrs = d_tokens_ptr + offs_m[:, None] * hidden_dim + offs_h[None, :]
    out_mask = m_mask[:, None] & h_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def grouped_gemm_dx_tiled(
    d_out_ptr,           # [S, N]
    weights_ptr,         # [E, H, N]
    d_tokens_ptr,        # [S, H]
    expert_tkn_cnt,      # [E]
    expert_offsets,      # [E]
    dx_counter_ptr,      # [E, max_m_tiles] int32
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    max_m_tiles: tl.constexpr,
    num_h_tiles: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    tiles_per_expert = max_m_tiles * num_h_tiles
    e = pid // tiles_per_expert
    rem = pid % tiles_per_expert
    tile_m_idx = rem // num_h_tiles
    tile_h_idx = rem % num_h_tiles

    if e >= expert_cnt:
        return

    e64 = e.to(tl.int64)
    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)
    row0 = tile_m_idx * BLOCK_SIZE_M
    if row0 >= cnt:
        return

    token_base = tl.load(expert_offsets + e).to(tl.int64)
    offs_m_local = row0 + tl.arange(0, BLOCK_SIZE_M)
    offs_m = (token_base + offs_m_local).to(tl.int64)
    offs_h = tile_h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_mask = offs_m_local < cnt
    h_mask = offs_h < hidden_dim

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    n0 = 0
    while n0 < expert_hidden_dim:
        n = n0 + offs_k
        n_mask = n < expert_hidden_dim

        a_ptrs = d_out_ptr + offs_m[:, None] * expert_hidden_dim + n[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

        b_ptrs = (
            weights_ptr
            + e64 * hidden_dim * expert_hidden_dim
            + n[:, None]
            + offs_h[None, :] * expert_hidden_dim
        )
        b = tl.load(b_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0)

        acc += tl.dot(a, b)
        n0 += BLOCK_SIZE_K

    out_ptrs = d_tokens_ptr + offs_m[:, None] * hidden_dim + offs_h[None, :]
    out_mask = m_mask[:, None] & h_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    tl.atomic_add(dx_counter_ptr + e * max_m_tiles + tile_m_idx, 1)


@triton.jit
def grouped_gemm_dw_partial(
    tokens_ptr,          # [S, H]
    d_out_ptr,           # [S, N]
    partial_ptr,         # [E, M_TILE_BATCH, HN_TILE_BATCH, BLOCK_H, BLOCK_N]
    expert_tkn_cnt,      # [E]
    expert_offsets,      # [E]
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    total_tokens: tl.constexpr,
    m_tile_start: tl.constexpr,
    m_tile_batch: tl.constexpr,
    hn_tile_start: tl.constexpr,
    hn_tile_batch: tl.constexpr,
    num_h_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    debug_expert_id: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    e = tl.program_id(0)
    hn_local = tl.program_id(1)
    m_local = tl.program_id(2)

    if e >= expert_cnt:
        return
    if debug_expert_id >= 0 and e != debug_expert_id:
        return
    if hn_local >= hn_tile_batch:
        return
    if m_local >= m_tile_batch:
        return

    global_hn_tile = hn_tile_start + hn_local
    if global_hn_tile >= num_h_tiles * num_n_tiles:
        return

    h_tile = global_hn_tile // num_n_tiles
    eh_tile = global_hn_tile % num_n_tiles

    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)
    token_base = tl.load(expert_offsets + e).to(tl.int64)
    global_m_tile = m_tile_start + m_local
    row0 = global_m_tile * BLOCK_SIZE_M
    if row0 >= cnt:
        return

    offs_h = h_tile * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_n = eh_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = row0 + tl.arange(0, BLOCK_SIZE_M)

    h_mask = offs_h < hidden_dim
    n_mask = offs_n < expert_hidden_dim

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)

    m_mask = offs_m < cnt
    rows = (token_base + offs_m).to(tl.int64)
    row_mask = m_mask & (rows < total_tokens)

    a_ptrs = tokens_ptr + rows[:, None] * hidden_dim + offs_h[None, :]
    a = tl.load(a_ptrs, mask=row_mask[:, None] & h_mask[None, :], other=0.0)

    expert_hidden_dim_i64 = tl.full((1,), expert_hidden_dim, dtype=tl.int64)
    b_ptrs = d_out_ptr + rows[:, None] * expert_hidden_dim_i64 + offs_n[None, :]
    b = tl.load(b_ptrs, mask=row_mask[:, None] & n_mask[None, :], other=0.0)

    acc += tl.dot(tl.trans(a), b)

    out_ptrs = (
        partial_ptr
        + ((e * m_tile_batch + m_local) * hn_tile_batch + hn_local)
        * BLOCK_SIZE_H
        * BLOCK_SIZE_N
        + tl.arange(0, BLOCK_SIZE_H)[:, None] * BLOCK_SIZE_N
        + tl.arange(0, BLOCK_SIZE_N)[None, :]
    )
    out_mask = h_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def grouped_gemm_dw_reduce(
    partial_ptr,         # [E, M_TILE_BATCH, HN_TILE_BATCH, BLOCK_H, BLOCK_N]
    d_weights_ptr,       # [E, H, N]
    expert_tkn_cnt,      # [E]
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    m_tile_start: tl.constexpr,
    m_tile_batch: tl.constexpr,
    hn_tile_start: tl.constexpr,
    hn_tile_batch: tl.constexpr,
    num_h_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    debug_expert_id: tl.constexpr,
    debug_hn_id: tl.constexpr,
    no_store: tl.constexpr,
    m_loop_limit: tl.constexpr,
    STORE_BLOCK: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    e = tl.program_id(0)
    hn_local = tl.program_id(1)
    store_chunk = tl.program_id(2)

    if e >= expert_cnt:
        return
    if debug_expert_id >= 0 and e != debug_expert_id:
        return
    if hn_local >= hn_tile_batch:
        return
    if debug_hn_id >= 0 and hn_local != debug_hn_id:
        return

    e64 = e.to(tl.int64)
    global_hn_tile = hn_tile_start + hn_local
    if global_hn_tile >= num_h_tiles * num_n_tiles:
        return

    h_tile = global_hn_tile // num_n_tiles
    eh_tile = global_hn_tile % num_n_tiles

    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)

    tile_elems = BLOCK_SIZE_H * BLOCK_SIZE_N
    elem = store_chunk * STORE_BLOCK + tl.arange(0, STORE_BLOCK)
    bh = elem // BLOCK_SIZE_N
    bn = elem % BLOCK_SIZE_N
    h_idx = h_tile * BLOCK_SIZE_H + bh
    n_idx = eh_tile * BLOCK_SIZE_N + bn
    elem_mask = (elem < tile_elems) & (h_idx < hidden_dim) & (n_idx < expert_hidden_dim)

    out_ptrs = (
        d_weights_ptr
        + e64 * hidden_dim * expert_hidden_dim
        + h_idx * expert_hidden_dim
        + n_idx
    )
    acc = tl.zeros((STORE_BLOCK,), dtype=tl.float32)
    if not no_store and m_tile_start > 0:
        acc = tl.load(out_ptrs, mask=elem_mask, other=0.0).to(tl.float32)

    rel_m_tile = 0
    while rel_m_tile < m_loop_limit:
        row0 = (m_tile_start + rel_m_tile) * BLOCK_SIZE_M
        rel_valid = row0 < cnt
        partial_ptrs = (
            partial_ptr
            + ((e * m_tile_batch + rel_m_tile) * hn_tile_batch + hn_local)
            * BLOCK_SIZE_H
            * BLOCK_SIZE_N
            + elem
        )
        partial = tl.load(partial_ptrs, mask=elem_mask & rel_valid, other=0.0).to(tl.float32)
        acc += partial
        rel_m_tile += 1

    if not no_store:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=elem_mask)


@triton.jit
def grouped_gemm_dw_fused(
    tokens_ptr,          # [S, H]
    d_out_ptr,           # [S, N]
    d_weights_ptr,       # [E, H, N]
    expert_tkn_cnt,      # [E]
    expert_offsets,      # [E]
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    total_tokens: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    e = tl.program_id(0)
    tile_h_idx = tl.program_id(1)
    tile_n_idx = tl.program_id(2)

    if e >= expert_cnt:
        return

    e64 = e.to(tl.int64)
    hidden_dim_i64 = tl.full((1,), hidden_dim, dtype=tl.int64)
    expert_hidden_dim_i64 = tl.full((1,), expert_hidden_dim, dtype=tl.int64)

    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)
    token_base = tl.load(expert_offsets + e).to(tl.int64)

    offs_h = tile_h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    h_mask = offs_h < hidden_dim
    n_mask = offs_n < expert_hidden_dim

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)

    m0 = 0
    while m0 < cnt:
        m = m0 + offs_m
        m_mask = m < cnt
        rows = (token_base + m).to(tl.int64)
        row_mask = m_mask & (rows < total_tokens)

        a_ptrs = tokens_ptr + rows[None, :] * hidden_dim_i64 + offs_h[:, None]
        a = tl.load(a_ptrs, mask=h_mask[:, None] & row_mask[None, :], other=0.0)

        b_ptrs = d_out_ptr + rows[:, None] * expert_hidden_dim_i64 + offs_n[None, :]
        b = tl.load(b_ptrs, mask=row_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)
        m0 += BLOCK_SIZE_M

    out_ptrs = (
        d_weights_ptr
        + e64 * hidden_dim_i64 * expert_hidden_dim_i64
        + offs_h[:, None] * expert_hidden_dim_i64
        + offs_n[None, :]
    )
    out_mask = h_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def grouped_gemm_dw_overlap(
    tokens_ptr,          # [S, H]
    d_out_ptr,           # [S, N]
    d_weights_ptr,       # [E, H, N]
    expert_tkn_cnt,      # [E]
    expert_offsets,      # [E]
    dx_counter_ptr,      # [E, max_m_tiles] int32
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    total_tokens: tl.constexpr,
    max_m_tiles: tl.constexpr,
    num_h_tiles: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    e = tl.program_id(0)
    tile_h_idx = tl.program_id(1)
    tile_n_idx = tl.program_id(2)

    if e >= expert_cnt:
        return

    e64 = e.to(tl.int64)
    hidden_dim_i64 = tl.full((1,), hidden_dim, dtype=tl.int64)
    expert_hidden_dim_i64 = tl.full((1,), expert_hidden_dim, dtype=tl.int64)

    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)
    token_base = tl.load(expert_offsets + e).to(tl.int64)

    offs_h = tile_h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    h_mask = offs_h < hidden_dim
    n_mask = offs_n < expert_hidden_dim

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)

    m0 = 0
    while m0 < cnt:
        tile_m = m0 // BLOCK_SIZE_M
        counter_ptr = dx_counter_ptr + e * max_m_tiles + tile_m
        flag = tl.load(counter_ptr)
        while flag < num_h_tiles:
            flag = tl.load(counter_ptr)

        m = m0 + offs_m
        m_mask = m < cnt
        rows = (token_base + m).to(tl.int64)
        row_mask = m_mask & (rows < total_tokens)

        a_ptrs = tokens_ptr + rows[None, :] * hidden_dim_i64 + offs_h[:, None]
        a = tl.load(a_ptrs, mask=h_mask[:, None] & row_mask[None, :], other=0.0)

        b_ptrs = d_out_ptr + rows[:, None] * expert_hidden_dim_i64 + offs_n[None, :]
        b = tl.load(b_ptrs, mask=row_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)
        m0 += BLOCK_SIZE_M

    out_ptrs = (
        d_weights_ptr
        + e64 * hidden_dim_i64 * expert_hidden_dim_i64
        + offs_h[:, None] * expert_hidden_dim_i64
        + offs_n[None, :]
    )
    out_mask = h_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def grouped_gemm_dw(
    tokens_ptr,          # [S, H]
    d_out_ptr,           # [S, N]
    d_weights_ptr,       # [E, H, N]
    expert_tkn_cnt,      # [E]
    expert_offsets,      # [E]
    expert_cnt: tl.constexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.constexpr,
    capacity: tl.constexpr,
    total_tokens: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Legacy single-kernel path kept for comparison/debugging. Expert.backward
    # uses the two-stage partial/reduce path for large-count stability.
    e = tl.program_id(0)
    tile_h_idx = tl.program_id(1)
    tile_n_idx = tl.program_id(2)

    if e >= expert_cnt:
        return

    e64 = e.to(tl.int64)
    cnt = tl.minimum(tl.load(expert_tkn_cnt + e).to(tl.int64), capacity).to(tl.int64)
    token_base = tl.load(expert_offsets + e).to(tl.int64)

    offs_h = tile_h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    h_mask = offs_h < hidden_dim
    n_mask = offs_n < expert_hidden_dim

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)

    m0 = 0
    while m0 < cnt:
        m = m0 + offs_m
        m_mask = m < cnt
        rows = (token_base + m).to(tl.int64)
        row_mask = m_mask & (rows < total_tokens)

        a_ptrs = tokens_ptr + rows[None, :] * hidden_dim + offs_h[:, None]
        a = tl.load(a_ptrs, mask=h_mask[:, None] & row_mask[None, :], other=0.0)

        expert_hidden_dim_i64 = tl.full((1,), expert_hidden_dim, dtype=tl.int64)
        b_ptrs = d_out_ptr + rows[:, None] * expert_hidden_dim_i64 + offs_n[None, :]
        b = tl.load(b_ptrs, mask=row_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)
        m0 += BLOCK_SIZE_M

    out_ptrs = (
        d_weights_ptr
        + e64 * hidden_dim * expert_hidden_dim
        + offs_h[:, None] * expert_hidden_dim
        + offs_n[None, :]
    )
    out_mask = h_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def token_exchange_push_back(
    d_tokens_bases,       # [W] int64 symmetric base ptrs for remote d_tokens
    read_meta,            # [W, E]
    write_meta,           # [W, E]
    local_expert_cnts,    # [W, E]
    d_gathered_tokens,    # [S_recv, H]
    src_rank,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    dst = tl.program_id(0)
    e = tl.program_id(1)
    tm = tl.program_id(2)

    cnt = tl.load(local_expert_cnts + dst * e_local + e)
    row0 = tm * BLOCK_SIZE_M
    if row0 >= cnt:
        return

    read_base = tl.load(read_meta + dst * e_local + e)
    write_base = tl.load(write_meta + dst * e_local + e)

    offs_m = row0 + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < cnt

    remote_d_tokens_base = tl.load(d_tokens_bases + dst).to(tl.pointer_type(tl.bfloat16))

    k0 = 0
    while k0 < hidden_dim:
        offs_k = k0 + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < hidden_dim

        src_ptrs = d_gathered_tokens + (write_base + offs_m)[:, None] * hidden_dim + offs_k[None, :]
        dst_ptrs = remote_d_tokens_base + (read_base + offs_m)[:, None] * hidden_dim + offs_k[None, :]

        vals = tl.load(src_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        tl.store(dst_ptrs, vals, mask=m_mask[:, None] & k_mask[None, :])

        k0 += BLOCK_SIZE_K
