import torch
import torch.nn.functional as F

from kernels import grouped_gemm


import triton

import os
import torch.distributed as dist

class Expert(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tokens,              # [S, H]
        weights,             # [E, H, N]
        local_expert_cnt,    # [E]
        aggregrate_exp_cnt
    ):
        if tokens.numel() == 0:
            return torch.empty(
                (0, weights.size(-1)),
                dtype=tokens.dtype,
                device=tokens.device,
            )

        device = tokens.device
        E = weights.size(0)
        H = weights.size(1)
        N = weights.size(2)
        S = tokens.size(0)

        # prefix offsets for packed tokens
        expert_offsets = torch.zeros_like(local_expert_cnt, dtype=torch.int32, device=device)
        if E > 1:
            expert_offsets[1:] = local_expert_cnt.cumsum(dim=0)[:-1].to(torch.int32)

        out = torch.empty((S, N), dtype=tokens.dtype, device=device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # total tile count across all experts
        
        cnt_host = local_expert_cnt.tolist()
        pid_to_expert_host = []
        pid_to_m_host = []
        pid_to_n_host = []

        for g, cnt in enumerate(cnt_host):
            if cnt <= 0:
                continue
            m_tiles = triton.cdiv(cnt, BLOCK_M)
            n_tiles = triton.cdiv(N, BLOCK_N)
            for tm in range(m_tiles):
                for tn in range(n_tiles):
                    pid_to_expert_host.append(g)
                    pid_to_m_host.append(tm)
                    pid_to_n_host.append(tn)

        total_tiles = len(pid_to_expert_host)
        pid_to_expert = torch.tensor(pid_to_expert_host, device=device, dtype=torch.int32)
        pid_to_m = torch.tensor(pid_to_m_host, device=device, dtype=torch.int32)
        pid_to_n = torch.tensor(pid_to_n_host, device=device, dtype=torch.int32)


        if len(pid_to_expert_host) == 0:
            return out

        total_tiles = len(pid_to_expert_host)
        pid_to_expert = torch.tensor(pid_to_expert_host, device=device, dtype=torch.int32)
        pid_to_m = torch.tensor(pid_to_m_host, device=device, dtype=torch.int32)
        pid_to_n = torch.tensor(pid_to_n_host, device=device, dtype=torch.int32)

        grid = (total_tiles,)
        ### debugging info
        if os.environ.get("DEBUG_GGEMM", "0") == "1":
            rank = dist.get_rank() if dist.is_initialized() else -1
            print(f"[GGEMM][rank{rank}] S={S}, E={E}, H={H}, N={N}", flush=True)
            print(f"[GGEMM][rank{rank}] local_expert_cnt={local_expert_cnt.tolist()}", flush=True)
            print(f"[GGEMM][rank{rank}] expert_offsets={expert_offsets.tolist()}", flush=True)
            print(f"[GGEMM][rank{rank}] total_tiles={total_tiles}, grid={grid}", flush=True)
        assert int(local_expert_cnt.sum().item()) == S, f"sum(local_expert_cnt)={int(local_expert_cnt.sum().item())}, S={S}"

        for i in range(E):
            cnt = int(local_expert_cnt[i].item())
            off = int(expert_offsets[i].item())
            assert cnt >= 0
            assert off >= 0
            assert off + cnt <= S, f"expert {i}: off={off}, cnt={cnt}, S={S}"
        ###

        grouped_gemm[grid](
            tokens,
            weights,
            out,
            local_expert_cnt,
            expert_offsets,
            pid_to_expert,
            pid_to_m,
            pid_to_n,
            NUM_SM=1,  # keep simple first
            expert_cnt=E,
            hidden_dim=H,
            expert_hidden_dim=N,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
        )

        return out

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not implemented yet.")


expert = Expert.apply