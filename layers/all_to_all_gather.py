import torch
import torch.distributed as dist
import os

from kernels import counts_exchange_pull, roll_cum_sum, token_exchange_pull,  token_exchange_pull_tiles
from backward_kernels import token_exchange_push_back
from utils import nvtx_push, nvtx_pop
from utils import HAS_SYMMEM, symmem, _as_cuda_i64_ptr_tensor

from dataclasses import dataclass
import triton


_GATHER_BACKWARD_CACHE = {}


def _get_backward_tokens_workspace(tokens, world_size):
    if not HAS_SYMMEM:
        raise RuntimeError("PyTorch SymmMem is required for AllToAllGatherOp backward")

    device = tokens.device
    key = (
        world_size,
        device.index,
        tokens.dtype,
        tuple(tokens.shape),
    )
    cached = _GATHER_BACKWARD_CACHE.get(key)
    if cached is not None:
        return cached

    d_tokens = symmem.empty(tuple(tokens.shape), dtype=tokens.dtype, device=device)
    d_tokens_hdl = symmem.rendezvous(d_tokens, group=dist.group.WORLD)
    d_tokens_bases = _as_cuda_i64_ptr_tensor(d_tokens_hdl.buffer_ptrs, device)

    cached = {
        "d_tokens": d_tokens,
        "d_tokens_bases": d_tokens_bases,
        "d_tokens_hdl": d_tokens_hdl,
    }
    _GATHER_BACKWARD_CACHE[key] = cached
    return cached


class AllToAllGatherOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        cnts,              # local symm tensor [W, E]
        offsets,           # local symm tensor [W]
        tokens,            # local symm tensor [T, H]
        cnts_bases,        # [W] int64 on cuda
        offsets_bases,     # [W] int64 on cuda
        tokens_bases,      # [W] int64 on cuda
        e_local,
        
    ):
        """
        Pull-style all-to-all gather implemented on top of Pytorch symmetric memory.
        Fllowed by the orginal gather design

        On rank d, the operation proceeds in three stages:

        1. Metadata pull:
           The first kernel pulls from every peer the routing metadata relevant to rank d:
             - local_expert_cnts[i, j]: number of tokens that peer i sends to expert j on rank d
             - local_expert_offset_idxs[i]: starting row in peer i's packed send buffer for tokens routed to rank d

        2. Local metadata preparation:
           `roll_cum_sum` converts the pulled metadata into:
             - read_meta[i, j]: starting row in peer i's send buffer for expert j destined for rank d
             - write_meta[i, j]: starting row in rank d's packed receive buffer for tokens pulled
               from peer i for expert j

        3. Token pull:
           The second kernel uses the metadata above to pull token rows directly from each peer's
           symmetric token buffer into a packed local receive buffer (`gathered_tokens`).

        Such may keep the gather path padding-free after metadata preparation 
        """
        
        ## New sort of initilization $$
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = tokens.device
        backward_ws = _get_backward_tokens_workspace(tokens, world_size)

        #print(f"[rank{rank}] gather: enter forward", flush=True)

        local_expert_cnts = torch.zeros((world_size, e_local), dtype=torch.int64, device=device)
        local_expert_offset_idxs = torch.zeros((world_size,), dtype=torch.int64, device=device)
        cnt_exchange_sync = torch.zeros((1,), dtype=torch.int32, device=device)

        BLOCK_M = 64

        #print(f"[rank{rank}] gather: before stage1 launch", flush=True)
        counts_exchange_pull[(world_size,)](
            cnts_bases=cnts_bases,
            offsets_bases=offsets_bases,
            local_expert_cnts=local_expert_cnts,
            local_expert_offset_idxs=local_expert_offset_idxs,
            cnt_exchange_sync=cnt_exchange_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_M=BLOCK_M,
        )
        #print(f"[rank{rank}] gather: after stage1 launch", flush=True)
        #print(f"[rank{rank}] gather: before stage1 cuda synchronize", flush=True)
              
        #print(f"[rank{rank}] gather: after stage1 cuda synchronize", flush=True)
        #print(f"[rank{rank}] gather: before roll_cum_sum", flush=True)
        read_meta, write_meta = roll_cum_sum(local_expert_cnts, local_expert_offset_idxs)
        #print(f"[rank{rank}] gather: after roll_cum_sum", flush=True)
        total_recv = int(local_expert_cnts.sum().item())
        max_cnt = max(int(local_expert_cnts.max().item()), 1)
        if os.environ.get("DEBUG_GATHER_FWD", "0") == "1":
            print(
                f"[GATHER_FWD][rank{rank}] total_recv={total_recv} max_cnt={max_cnt} "
                f"local_expert_cnts={local_expert_cnts.tolist()}",
                flush=True,
            )
        gathered_tokens = torch.empty((total_recv, tokens.shape[-1]), dtype=tokens.dtype, device=device)
        token_sync = torch.zeros((world_size, e_local), dtype=torch.int32, device=device)

        #print(f"[rank{rank}] gather: after roll_cum_sum total_recv={total_recv} max_cnt={max_cnt}", flush=True)

        token_exchange_pull[(world_size, e_local, max_cnt)](
            tokens_bases=tokens_bases,
            read_meta=read_meta,
            write_meta=write_meta,
            local_expert_cnts=local_expert_cnts,
            gathered_tokens=gathered_tokens,
            token_sync=token_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            hidden_dim=tokens.shape[-1],
            BLOCK_K=256,
            num_warps=4,
        )
        #print(f"[rank{rank}] gather: after stage2 launch", flush=True)

        #print(f"[rank{rank}] gather: before final cuda synchronize", flush=True)
        #torch.cuda.synchronize(device)
        #print(f"[rank{rank}] gather: after final cuda synchronize", flush=True)

        ctx.save_for_backward(read_meta, write_meta, local_expert_cnts)
        ctx.tokens_bases = tokens_bases
        ctx.d_tokens_bases = backward_ws["d_tokens_bases"]
        ctx.d_tokens = backward_ws["d_tokens"]
        ctx.world_size = world_size
        ctx.e_local = e_local
        ctx.tokens_shape = tokens.shape
        ctx.tokens_dtype = tokens.dtype

        return gathered_tokens

    @staticmethod
    def backward(ctx, d_gathered_tokens):
        read_meta, write_meta, local_expert_cnts = ctx.saved_tensors

        if ctx.d_tokens is None or ctx.d_tokens_bases is None:
            return None, None, None, None, None, None, None

        d_gathered_tokens = d_gathered_tokens.contiguous()
        block_m = 64
        max_cnt = max(int(local_expert_cnts.max().item()), 1)
        max_tiles = triton.cdiv(max_cnt, block_m)

        if os.environ.get("DEBUG_SKIP_PUSH_BACK", "0") != "1":
            token_exchange_push_back[(ctx.world_size, ctx.e_local, max_tiles)](
                d_tokens_bases=ctx.d_tokens_bases,
                read_meta=read_meta,
                write_meta=write_meta,
                local_expert_cnts=local_expert_cnts,
                d_gathered_tokens=d_gathered_tokens,
                src_rank=dist.get_rank(),
                world_size=ctx.world_size,
                e_local=ctx.e_local,
                hidden_dim=ctx.tokens_shape[-1],
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_K=256,
                num_warps=4,
            )
        torch.cuda.current_stream().synchronize()

        return None, None, ctx.d_tokens.clone(), None, None, None, None

# new stuff for overlap test
@dataclass
class GatherOverlapState:
    gathered_tokens: torch.Tensor      # [S, H]
    local_expert_cnts: torch.Tensor    # [W, E]
    read_meta: torch.Tensor            # [W, E]
    write_meta: torch.Tensor           # [W, E]
    ready: torch.Tensor                # [E, W, MAX_TILES]
    world_size: int
    e_local: int
    hidden_dim: int
    block_m: int
    spin_iters: torch.Tensor | None = None
    wait_cycles: torch.Tensor | None = None
    prog_pid: torch.Tensor | None = None
    prog_cnt: torch.Tensor | None = None
    probe_total_tiles: int = 0
    probe_num_n_tiles: int = 0

@dataclass
class GatherOverlapWorkspace:
    local_expert_cnts: torch.Tensor         # [W, E] int64
    local_expert_offset_idxs: torch.Tensor  # [W] int64
    cnt_exchange_sync: torch.Tensor         # [1] int32

    gathered_tokens: torch.Tensor           # [S, H]
    ready: torch.Tensor                     # [E, W, MAX_TILES] int32

    world_size: int
    e_local: int
    hidden_dim: int
    max_tiles: int
    block_m: int


def alloc_gather_overlap_workspace(
    *,
    world_size: int,
    e_local: int,
    total_recv: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
    device: torch.device,
    max_tiles: int,
    block_m: int = 64,
) -> GatherOverlapWorkspace:
    return GatherOverlapWorkspace(
        local_expert_cnts=torch.empty(
            (world_size, e_local),
            dtype=torch.int64,
            device=device,
        ),
        local_expert_offset_idxs=torch.empty(
            (world_size,),
            dtype=torch.int64,
            device=device,
        ),
        cnt_exchange_sync=torch.empty(
            (1,),
            dtype=torch.int32,
            device=device,
        ),
        gathered_tokens=torch.empty(
            (total_recv, hidden_dim),
            dtype=token_dtype,
            device=device,
        ),
        ready=torch.zeros(
            (e_local, world_size, max_tiles),
            dtype=torch.int32,
            device=device,
        ),
        world_size=world_size,
        e_local=e_local,
        hidden_dim=hidden_dim,
        max_tiles=max_tiles,
        block_m=block_m,
    )


def reset_gather_overlap_workspace(ws: GatherOverlapWorkspace) -> None:
    # Only reset stateful metadata / synchronization buffers.
    # Do not zero gathered_tokens; it is fully overwritten by producer kernels.
    ws.local_expert_cnts.zero_()
    ws.local_expert_offset_idxs.zero_()
    ws.cnt_exchange_sync.zero_()
    ws.ready.zero_()


def gather_a2a_prepare_metadata_overlap(
    cnts,
    offsets,
    tokens,
    cnts_bases,
    offsets_bases,
    e_local: int,
    workspace: GatherOverlapWorkspace,
    comm_stream: torch.cuda.Stream | None = None,
) -> GatherOverlapState:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(
        f"[rank{rank}] ENTER gather_a2a_prepare_metadata_overlap",
        flush=True,
    )

    assert workspace.world_size == world_size
    assert workspace.e_local == e_local
    assert workspace.hidden_dim == tokens.shape[-1]

    def _launch_stage1():
        counts_exchange_pull[(world_size,)](
            cnts_bases=cnts_bases,
            offsets_bases=offsets_bases,
            local_expert_cnts=workspace.local_expert_cnts,
            local_expert_offset_idxs=workspace.local_expert_offset_idxs,
            cnt_exchange_sync=workspace.cnt_exchange_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_M=workspace.block_m,
        )

    if comm_stream is None:
        _launch_stage1()
        read_meta, write_meta = roll_cum_sum(
            workspace.local_expert_cnts,
            workspace.local_expert_offset_idxs,
        )
    else:
        with torch.cuda.stream(comm_stream):
            _launch_stage1()
            read_meta, write_meta = roll_cum_sum(
                workspace.local_expert_cnts,
                workspace.local_expert_offset_idxs,
            )

    state = GatherOverlapState(
        gathered_tokens=workspace.gathered_tokens,
        local_expert_cnts=workspace.local_expert_cnts,
        read_meta=read_meta,
        write_meta=write_meta,
        ready=workspace.ready,
        world_size=world_size,
        e_local=e_local,
        hidden_dim=tokens.shape[-1],
        block_m=workspace.block_m,
        spin_iters=None,
        wait_cycles=None,
        prog_pid=None,
        prog_cnt=None,
        probe_total_tiles=0,
        probe_num_n_tiles=0,
    )
    print(
        f"[rank{rank}] EXIT gather_a2a_prepare_metadata_overlap",
        flush=True,
    )
    return state


def gather_a2a_launch_tiles_overlap(
    tokens_bases,
    state: GatherOverlapState,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    rank = dist.get_rank()
    world_size = state.world_size
    e_local = state.e_local
    hidden_dim = state.hidden_dim
    max_tiles = state.ready.shape[-1]
    block_m = state.block_m

    def _launch_stage2():
        token_exchange_pull_tiles[(world_size, e_local, max_tiles)](
            tokens_bases=tokens_bases,
            read_meta=state.read_meta,
            write_meta=state.write_meta,
            local_expert_cnts=state.local_expert_cnts,
            gathered_tokens=state.gathered_tokens,
            ready=state.ready,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            hidden_dim=hidden_dim,
            max_tiles=max_tiles,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=256,
            num_warps=4,
        )

    if comm_stream is None:
        _launch_stage2()
    else:
        with torch.cuda.stream(comm_stream):
            _launch_stage2()

@dataclass
class GatherOverlapWorkspace:
    local_expert_cnts: torch.Tensor         # [W, E] int64
    local_expert_offset_idxs: torch.Tensor  # [W]    int64
    cnt_exchange_sync: torch.Tensor         # [1]    int32

    gathered_tokens: torch.Tensor           # [S, H]
    ready: torch.Tensor                     # [E, W, MAX_TILES] int32

    world_size: int
    e_local: int
    hidden_dim: int
    max_tiles: int
    block_m: int

def gather_a2a_launch_tiles_overlap(
    tokens_bases,      # [W] int64 on cuda
    state: GatherOverlapState,
    comm_stream: torch.cuda.Stream | None = None,
):
    rank = dist.get_rank()
    print(f"[rank{rank}] ENTER gather_a2a_launch_tiles_overlap", flush=True)
    world_size = state.world_size
    e_local = state.e_local
    hidden_dim = state.hidden_dim
    max_tiles = state.ready.shape[-1]
    block_m = state.block_m

    def _launch_stage2():
        token_exchange_pull_tiles[(world_size, e_local, max_tiles)](
            tokens_bases=tokens_bases,
            read_meta=state.read_meta,
            write_meta=state.write_meta,
            local_expert_cnts=state.local_expert_cnts,
            gathered_tokens=state.gathered_tokens,
            ready=state.ready,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            hidden_dim=hidden_dim,
            max_tiles=max_tiles,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=256,
            num_warps=4,
        )

    if comm_stream is None:
        _launch_stage2()
    else:
        with torch.cuda.stream(comm_stream):
            _launch_stage2()
    print(f"[rank{rank}] EXIT gather_a2a_launch_tiles_overlap", flush=True)
            
def gather_a2a_prepare_overlap(
    cnts,              # local symm tensor [W, E]
    offsets,           # local symm tensor [W]
    tokens,            # local symm tensor [T, H]
    cnts_bases,        # [W] int64 on cuda
    offsets_bases,     # [W] int64 on cuda
    tokens_bases,      # [W] int64 on cuda
    e_local: int,
    comm_stream: torch.cuda.Stream | None = None,
    block_m: int = 64,
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = tokens.device
    hidden_dim = tokens.shape[-1]

    # ----- Stage 1: metadata pull -----
    local_expert_cnts = torch.zeros((world_size, e_local), dtype=torch.int64, device=device)
    local_expert_offset_idxs = torch.zeros((world_size,), dtype=torch.int64, device=device)
    cnt_exchange_sync = torch.zeros((1,), dtype=torch.int32, device=device)

    def _launch_stage1():
        counts_exchange_pull[(world_size,)](
            cnts_bases=cnts_bases,
            offsets_bases=offsets_bases,
            local_expert_cnts=local_expert_cnts,
            local_expert_offset_idxs=local_expert_offset_idxs,
            cnt_exchange_sync=cnt_exchange_sync,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_M=block_m,
        )

    if comm_stream is None:
        _launch_stage1()
    else:
        with torch.cuda.stream(comm_stream):
            _launch_stage1()

    # This still relies on sequential launches on the same stream.
    read_meta, write_meta = roll_cum_sum(local_expert_cnts, local_expert_offset_idxs)

    # This first version still makes these shape decisions on the host.
    total_recv = int(local_expert_cnts.sum().item())
    max_cnt = max(int(local_expert_cnts.max().item()), 1)
    max_tiles = triton.cdiv(max_cnt, block_m)

    gathered_tokens = torch.empty((total_recv, hidden_dim), dtype=tokens.dtype, device=device)
    ready = torch.zeros((e_local, world_size, max_tiles), dtype=torch.int32, device=device)

    # ----- Stage 2: tile producer -----
    def _launch_stage2():
        token_exchange_pull_tiles[(world_size, e_local, max_tiles)](
            tokens_bases=tokens_bases,
            read_meta=read_meta,
            write_meta=write_meta,
            local_expert_cnts=local_expert_cnts,
            gathered_tokens=gathered_tokens,
            ready=ready,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            hidden_dim=hidden_dim,
            max_tiles=max_tiles,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=256,
            num_warps=4,
        )

    if comm_stream is None:
        _launch_stage2()
    else:
        with torch.cuda.stream(comm_stream):
            _launch_stage2()

    return GatherOverlapState(
        gathered_tokens=gathered_tokens,
        local_expert_cnts=local_expert_cnts,
        read_meta=read_meta,
        write_meta=write_meta,
        ready=ready,
        world_size=world_size,
        e_local=e_local,
        hidden_dim=hidden_dim,
        block_m=block_m,
        spin_iters=None,
        wait_cycles=None,
        prog_pid=None,
        prog_cnt=None,
        probe_total_tiles=0,
        probe_num_n_tiles=0,
    )

gather_a2a = AllToAllGatherOp.apply
