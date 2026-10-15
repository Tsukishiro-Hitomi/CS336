import math
import torch
from collections.abc import Iterable

def learning_rate_schedule(t: int,
                           alpha_max: float,
                           alpha_min: float,
                           t_w: int,
                           t_c: int) -> float:
    assert t_w <= t_c, f"t_w = {t_w} is the lower bound, t_c = {t_c} is the upper bound"
    if t < t_w:
        return t / t_w * alpha_max
    if t >= t_w and t <= t_c:
        return alpha_min + 0.5 * (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (alpha_max - alpha_min)
    return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      M: float,
                      eps = 1e-6):
    sum = .0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.data
        sum += torch.sum(torch.square(grad))

    grad_norm = math.sqrt(sum)
    if grad_norm >= M:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.mul_(M / (grad_norm + eps))

            
