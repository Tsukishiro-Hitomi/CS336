import numpy.typing as npt
import torch

def get_batch(x: npt.NDArray,
              batch_size: int, 
              context_length: int, 
              device: str
              ) -> tuple[torch.Tensor, torch.Tensor]:
    assert device in ("cpu", "cuda:0"), "device must be either 'cpu' or 'cuda:0'"
    total_length = x.shape[0]
    assert total_length >= context_length + 1, "the length of dataset must be greater than context_length"
    pre = torch.zeros((batch_size, context_length), device=device, dtype=torch.long)
    next = torch.zeros((batch_size, context_length), device=device, dtype=torch.long)
    for i in range(batch_size):
        # pre: [0, total_length - context_length) - [context_length - 1, total_length - 1)
        # cur: [1, total_length - context_length + 1) - [context_length, total_length)
        start = torch.randint(0, total_length - context_length, (1,)).item()
        pre[i] = torch.Tensor(x[start : start + context_length], device=device)
        next[i] = torch.Tensor(x[start + 1: start + context_length + 1], device=device)

    return (pre, next)