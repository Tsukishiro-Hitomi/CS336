import torch
from einops import rearrange

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: (batch, context_length, vocab_size)
    # targets: (batch, context_length)
    logits = inputs.clone()
    logits = rearrange(logits, "... v -> (...) v")
    targets = rearrange(targets, "... -> (...)")
    batch = logits.shape[0]

    # numeric stablity
    logits -= torch.max(logits, dim=-1, keepdim=True).values

    loss = .0

    for i in range(batch):
        loss -= logits[i, targets[i]]

    logits_sum = torch.sum(torch.exp(logits), dim=-1)
    log_logits_sum = torch.log(logits_sum)

    loss += torch.sum(log_logits_sum)
    return loss / batch
