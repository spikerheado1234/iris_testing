import os
import time
import argparse
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._symmetric_memory as symmem

from layers.all_to_all_gather import gather_a2a
from layers.expert import expert
from utils import _alloc_symm_gather_buffers


# =========================================================
# Part 0. Common helpers
# =========================================================

def _require_cuda():
    assert torch.cuda.is_available(), "CUDA/HIP device is required."


def _device():
    return torch.device("cuda")


def _dtype():
    return torch.bfloat16


def _rand_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_dist_tcp(rank: int, world_size: int, master_port: str = "29500") -> None:
    cache_dir = f"/tmp/triton_cache_{os.getuid()}_{os.getpid()}_rank{rank}"
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )


def _destroy_dist():
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _require_dist():
    assert dist.is_initialized(), "torch.distributed must be initialized."
    assert dist.get_world_size() > 1, "Need WORLD_SIZE > 1."
    assert torch.cuda.is_available(), "CUDA/HIP required."


def _make_expert_offsets(local_expert_cnt: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros_like(local_expert_cnt, dtype=torch.int32, device=local_expert_cnt.device)
    if local_expert_cnt.numel() > 1:
        offsets[1:] = local_expert_cnt.cumsum(dim=0)[:-1].to(torch.int32)
    return offsets


def ref_expert(tokens: torch.Tensor, weights: torch.Tensor, local_expert_cnt: torch.Tensor) -> torch.Tensor:
    """
    CPU reference for packed expert-major tokens.
    tokens: [S, H]
    weights: [E, H, N]
    local_expert_cnt: [E]
    """
    if tokens.numel() == 0:
        return torch.empty((0, weights.size(-1)), dtype=torch.float32)

    tokens_cpu = tokens.float().cpu()
    weights_cpu = weights.float().cpu()
    cnt_cpu = local_expert_cnt.cpu()

    out_chunks = []
    offset = 0
    for i in range(weights_cpu.size(0)):
        cnt = int(cnt_cpu[i].item())
        if cnt > 0:
            out_chunks.append(torch.matmul(tokens_cpu[offset:offset + cnt], weights_cpu[i]))
            offset += cnt

    if len(out_chunks) == 0:
        return torch.empty((0, weights.size(-1)), dtype=torch.float32)

    return torch.cat(out_chunks, dim=0).float()


# =========================================================
# Part 1. Single-GPU GEMM correctness tests
# =========================================================

def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim, counts=None, seed=42):
    _require_cuda()
    _rand_seed(seed)

    device = _device()
    dtype = _dtype()

    if counts is None:
        expert_token_cnt = torch.randint(
            low=0,
            high=33,
            size=(num_local_experts,),
            device=device,
            dtype=torch.int32,
        )
    else:
        expert_token_cnt = torch.tensor(counts, device=device, dtype=torch.int32)
        assert expert_token_cnt.numel() == num_local_experts

    total_tokens = int(expert_token_cnt.sum().item())
    tokens = torch.randn((total_tokens, token_hid_dim), device=device, dtype=dtype)
    weights = torch.randn((num_local_experts, token_hid_dim, expert_hid_dim), device=device, dtype=dtype)
    return tokens, weights, expert_token_cnt


def run_one_gemm_case(
    num_local_experts,
    token_hid_dim,
    expert_hid_dim,
    counts=None,
    seed=42,
    atol=2e-2,
    rtol=2e-2,
):
    tokens, weights, local_expert_cnt = gen_gemm_input(
        num_local_experts=num_local_experts,
        token_hid_dim=token_hid_dim,
        expert_hid_dim=expert_hid_dim,
        counts=counts,
        seed=seed,
    )

    expert_offsets = _make_expert_offsets(local_expert_cnt)
    assert int(local_expert_cnt.sum().item()) == tokens.size(0)
    for i in range(local_expert_cnt.numel()):
        cnt = int(local_expert_cnt[i].item())
        off = int(expert_offsets[i].item())
        assert off >= 0
        assert cnt >= 0
        assert off + cnt <= tokens.size(0)

    out = expert(tokens, weights, local_expert_cnt, num_local_experts)
    torch.cuda.synchronize()

    ref = ref_expert(tokens, weights, local_expert_cnt).float()
    out_cpu = out.float().cpu()

    assert out_cpu.shape == ref.shape, f"shape mismatch: {tuple(out_cpu.shape)} vs {tuple(ref.shape)}"

    max_diff = (out_cpu - ref).abs().max().item() if out_cpu.numel() > 0 else 0.0
    print(
        f"[GEMM] E={num_local_experts} H={token_hid_dim} N={expert_hid_dim} "
        f"counts={local_expert_cnt.tolist()} max_diff={max_diff:.6f}",
        flush=True,
    )
    assert torch.allclose(out_cpu, ref, atol=atol, rtol=rtol), f"GEMM mismatch, max_diff={max_diff}"


def test_gemm_small_suite():
    print("=== Part 1: GEMM small correctness suite ===", flush=True)
    run_one_gemm_case(2, 64, 64, counts=[0, 0], seed=1)
    run_one_gemm_case(2, 64, 64, counts=[1, 0], seed=2)
    run_one_gemm_case(2, 64, 64, counts=[1, 3], seed=3)
    run_one_gemm_case(4, 128, 64, counts=[0, 1, 2, 7], seed=4)
    run_one_gemm_case(4, 128, 128, counts=[5, 0, 9, 1], seed=5)
    run_one_gemm_case(4, 256, 128, counts=[17, 3, 0, 11], seed=6)


# =========================================================
# Part 2. Single-GPU GEMM stress tests
# =========================================================

def test_gemm_random_stress(num_cases=20):
    print("=== Part 2: GEMM random stress suite ===", flush=True)
    _require_cuda()
    cases = [
        (2, 64, 64),
        (4, 128, 64),
        (4, 128, 128),
        (4, 256, 128),
        (8, 256, 256),
    ]

    for i in range(num_cases):
        E, H, N = cases[i % len(cases)]
        run_one_gemm_case(E, H, N, counts=None, seed=100 + i)


# =========================================================
# Part 3. Multi-GPU gather_a2a correctness
# =========================================================

def test_gather_a2a(e_local: int = 2, hidden_dim: int = 128, cap: int = 32, threshold: float = 1e-2):
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda")

    # deterministic metadata: every src sends cap rows to every (dst, local expert)
    send_counts = torch.full((world_size, e_local), cap, device=device, dtype=torch.int32)
    dst_offsets = (torch.arange(world_size, device=device, dtype=torch.int32) * (e_local * cap)).contiguous()

    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device=device, dtype=torch.bfloat16)

    # encode source/dst/expert/token into value for exact checking
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    symm = _alloc_symm_gather_buffers(
        world_size=world_size,
        e_local=e_local,
        total_rows=total_rows,
        hidden_dim=hidden_dim,
        token_dtype=tokens.dtype,
        send_counts=send_counts,
        dst_offsets=dst_offsets,
        tokens=tokens,
        device=device,
    )

    dist.barrier()

    gathered = gather_a2a(
        symm["cnts"],
        symm["offsets"],
        symm["tokens"],
        symm["cnts_bases"],
        symm["offsets_bases"],
        symm["tokens_bases"],
        e_local,
    )

    torch.cuda.synchronize()
    dist.barrier()

    expected_rows = world_size * e_local * cap
    assert gathered.shape == (expected_rows, hidden_dim), (
        f"bad shape: got {gathered.shape}, expected {(expected_rows, hidden_dim)}"
    )

    # expected packed output: expert-major, then src-major, then token order
    expected = torch.empty_like(gathered)
    out_row = 0
    for e in range(e_local):
        for src in range(world_size):
            for t in range(cap):
                val = float(src * 1_000_000 + rank * 10_000 + e * 100 + t)
                expected[out_row].fill_(val)
                out_row += 1

    assert torch.allclose(gathered.float(), expected.float(), rtol=threshold, atol=threshold), (
        f"gather mismatch on rank {rank}"
    )

    if rank == 0:
        print(f"[GATHER] PASS e_local={e_local} H={hidden_dim} cap={cap}", flush=True)


# =========================================================
# Part 4. Multi-GPU gather_a2a -> expert smoke test
# =========================================================

def test_gather_e2e(
    e_local: int = 2,
    hidden_dim: int = 128,
    out_dim: int = 64,
    cap: int = 16,
    threshold: float = 2e-2,
):
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda")

    send_counts = torch.full((world_size, e_local), cap, device=device, dtype=torch.int32)
    dst_offsets = (torch.arange(world_size, device=device, dtype=torch.int32) * (e_local * cap)).contiguous()

    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device=device, dtype=torch.bfloat16)

    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    symm = _alloc_symm_gather_buffers(
        world_size=world_size,
        e_local=e_local,
        total_rows=total_rows,
        hidden_dim=hidden_dim,
        token_dtype=tokens.dtype,
        send_counts=send_counts,
        dst_offsets=dst_offsets,
        tokens=tokens,
        device=device,
    )

    dist.barrier()

    gathered = gather_a2a(
        symm["cnts"],
        symm["offsets"],
        symm["tokens"],
        symm["cnts_bases"],
        symm["offsets_bases"],
        symm["tokens_bases"],
        e_local,
    )

    # deterministic packed counts:
    # each local expert gets cap tokens from every src
    local_expert_cnt = torch.full(
        (e_local,),
        world_size * cap,
        device=device,
        dtype=torch.int32,
    )

    # fixed shared weights
    _rand_seed(1234)
    weights = torch.randn((e_local, hidden_dim, out_dim), device=device, dtype=torch.bfloat16)

    out_custom = expert(gathered, weights, local_expert_cnt, world_size * e_local)
    torch.cuda.synchronize()
    dist.barrier()

    # exact packed reference gathered output
    expected_gathered = torch.empty_like(gathered)
    out_row = 0
    for e in range(e_local):
        for src in range(world_size):
            for t in range(cap):
                val = float(src * 1_000_000 + rank * 10_000 + e * 100 + t)
                expected_gathered[out_row].fill_(val)
                out_row += 1

    assert torch.allclose(
        gathered.float(),
        expected_gathered.float(),
        rtol=threshold,
        atol=threshold,
    ), f"gathered token mismatch on rank {rank}"

    ref_out = ref_expert(expected_gathered, weights, local_expert_cnt)
    out_custom_cpu = out_custom.float().cpu()

    max_diff = (out_custom_cpu - ref_out).abs().max().item() if out_custom_cpu.numel() > 0 else 0.0
    assert out_custom_cpu.shape == ref_out.shape, (
        f"shape mismatch on rank {rank}: got {tuple(out_custom_cpu.shape)} vs {tuple(ref_out.shape)}"
    )
    assert torch.allclose(out_custom_cpu, ref_out, atol=threshold, rtol=threshold), (
        f"gather e2e mismatch on rank {rank}, max_diff={max_diff}"
    )

    if rank == 0:
        print(
            f"[GATHER-E2E] PASS e_local={e_local} H={hidden_dim} out_dim={out_dim} cap={cap} max_diff={max_diff:.6f}",
            flush=True,
        )


# =========================================================
# Dist worker
# =========================================================

def _worker_gather(local_rank: int, world_size: int, mode: str, master_port: str):
    _init_dist_tcp(local_rank, world_size, master_port=master_port)
    symmem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    print(f"[rank{local_rank}] init ok, cuda={torch.cuda.current_device()}", flush=True)

    try:
        if mode in ("gather", "all"):
            test_gather_a2a(e_local=2, hidden_dim=128, cap=32)
            test_gather_a2a(e_local=4, hidden_dim=256, cap=16)

        if mode in ("e2e", "all"):
            test_gather_e2e(e_local=2, hidden_dim=128, out_dim=64, cap=16)
            test_gather_e2e(e_local=4, hidden_dim=256, out_dim=128, cap=8)

        if local_rank == 0:
            print("[DIST] gather tests finished", flush=True)

    finally:
        _destroy_dist()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["gemm", "gather", "e2e", "all"],
        help="gemm = single-GPU only; gather/e2e/all need multi-GPU",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--stress-cases", type=int, default=20)
    args = parser.parse_args()

    print("RUNNING unit_tests_gather from:", __file__, flush=True)

    if args.mode == "gemm":
        _require_cuda()
        torch.cuda.set_device(0)
        test_gemm_small_suite()
        test_gemm_random_stress(num_cases=args.stress_cases)
        print("[GEMM] PASS", flush=True)
        return

    _require_cuda()
    assert torch.cuda.device_count() >= args.world_size, (
        f"Need at least {args.world_size} GPUs, got {torch.cuda.device_count()}"
    )

    mp.spawn(
        _worker_gather,
        args=(args.world_size, args.mode, args.master_port),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()