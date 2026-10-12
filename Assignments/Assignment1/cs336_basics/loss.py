import torch

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: (batch, vocab_size)
    # targets: (batch, )
    batch, vocab_size = inputs.shape
    print(f"batch = {batch}, vocab_size = {vocab_size}")

    logits = inputs.clone()

    # numeric stablity
    logits -= torch.max(logits, dim=-1, keepdim=True).values

    loss = .0
    for i in range(batch):
        loss -= logits[i, targets[i]]

    logits_sum = torch.sum(torch.exp(logits), dim=-1)
    log_logits_sum = torch.log(logits_sum)

    loss += torch.sum(log_logits_sum)
    return loss / batch
