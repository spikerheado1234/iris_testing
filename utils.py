import torch

def gen_tensor(
    batch: int, seq: int, hidden_dim: int, 
    world_size: int, num_experts: int, 
    rank: int) -> tuple[torch.tensor, torch.tensor]:
    torch.manual_seed(rank)
    tokens = torch.randn(batch*seq, hidden_dim, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

    ## We first do a common load tensor. ##
    assert (batch*seq) % num_experts == 0, 'must be evenly divisible'
    meta = torch.tensor([(batch*seq) // num_experts for _ in range(num_experts)]).to("cuda" if torch.cuda.is_available() else "cpu")

    return tokens, meta