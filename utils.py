import torch
import torch.nn.functional as F

def gen_tensor(
    batch: int, seq: int, hidden_dim: int, 
    world_size: int, num_experts: int, 
    rank: int, topk: int) -> tuple[torch.tensor, torch.tensor]:
    torch.manual_seed(rank)

    tokens = torch.rand(batch*seq, hidden_dim, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    router = torch.randn(hidden_dim, num_experts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

    routed_values = F.softmax(torch.einsum('bh, he -> be', tokens, router), dim=-1)
    top_vals, top_idxs = torch.topk(routed_values, topk)

    expert_tokens = []
    for ex in range(num_experts):
        mask = (top_idxs == ex).any(dim=-1)
        tkns = tokens[mask] # (num tokens routed to an expert, hidden dimension).
        if tkns.numel() > 0:
            expert_tokens.append(tkns)
        else:
            expert_tokens.append(torch.zeros(1, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu"))

    ## Next, we pad to the largest element-sized tensor.
    max_tkn_cnt = max([i.shape[0] for i in expert_tokens])

    expert_tokens = [F.pad(i, (0, max_tkn_cnt - i.shape[0]), "constant", 0) for i in expert_tokens]

    return expert_tokens

    ## Then, we send over.
    ## We first do a common load tensor. ##
    #assert (batch*seq) % num_experts == 0, 'must be evenly divisible'
    #meta = torch.tensor([(batch*seq) // num_experts for _ in range(num_experts)]).to("cuda" if torch.cuda.is_available() else "cpu")

    return expert_tokens 