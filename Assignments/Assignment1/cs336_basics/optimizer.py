from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]    # get the learning rate
            for p in group ["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]   # get state associated with p
                t = state.get("t", 0)   # get iteration number from the state, or 0
                grad = p.grad.data      # get the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad      # update weight tensor in-place
                state["t"] = t + 1      # increment iteration number

        return loss

# toy example
"""
for lr in [1, 1e1, 1e2, 1e3]:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10))) 
    opt = SGD([weights], lr=lr)
    print(f"lr = {lr}")
    for t in range(10):  
        opt.zero_grad() # Reset the gradients for all learnable parameters. 
        loss = (weights**2).mean() # Compute a scalar loss value. 
        print(loss.cpu().item()) 
        loss.backward() # Run backward pass, which computes gradients. 
        opt.step() # Run optimizer step.
"""

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr: float, 
                 weight_decay: float,
                 betas: tuple[float, float],
                 eps: float,
                 ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:  
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                grad = p.grad.data

                p.data -= lr * weight_decay * p.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss