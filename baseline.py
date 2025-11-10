"""
Baseline a2a implementation to compare perf against.
"""

import torch
import torch.distributed as dist

def run(
    rank: int, tokens: torch.tensor, meta: torch.tensor, 
    batch: int, seq: int, hidden_dim: int, 
    num_experts: int, world_size: int, 
    general_a2a: bool, shmem
):
    if general_a2a:
        pass ## Not implemented yet. ##
    else:
        assert tokens.shape[0] % world_size == 0, 'Incorrect sizes passed in.'
        transmit_size = tokens.shape[0] // world_size
        tokens_resh = list(torch.tensor_split(tokens, world_size, dim=0))
        tokens_recv = [torch.zeros(transmit_size, tokens.shape[-1], dtype=tokens.dtype).to("cuda" if torch.cuda.is_available() else "cpu") for _ in range(world_size)]
        dist.all_to_all(tokens_recv, tokens_resh)

        torch.cuda.synchronize()
        #print(f'[rank: {rank}], summed token buffer: {sum([i.sum() for i in tokens_recv])}')