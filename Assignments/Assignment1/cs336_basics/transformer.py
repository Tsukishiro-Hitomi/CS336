import torch
import torch.nn as nn
from cs336_basics.layers import Linear, RMSNorm
from cs336_basics.attention import multihead_self_attention, RoPE

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_ff: int | None = None):
        
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            # compute d_ff
            d_ff = int(d_model * 8 / 3)
            if d_ff % 64 != 0:
                d_ff -= d_ff % 64
        self.d_ff = d_ff

        self.linear1 = Linear(self.d_model, self.d_ff)
        self.linear2 = Linear(self.d_ff, self.d_model)
        self.linear3 = Linear(self.d_model, self.d_ff)
    
    def forward(self, x) -> torch.Tensor:
        # W1x: (... d_model) -> (... d_ff)
        out = self.linear1.forward(x)

        # SiLU (W1x)
        out = silu(out)

        # SiLU (W1x) * W3x
        out = out * self.linear3.forward(x)

        # W2 x SiLU (W1x) * W3x
        out = self.linear2.forward(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int, 
                 d_ff: int,
                 rope: RoPE,
                 token_positions: torch.Tensor):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff)
        self.attention = multihead_self_attention(d_model, num_heads)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.rope = rope
        self.token_positions = token_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attention.forward(self.rmsnorm1.forward(x), self.rope, self.token_positions)
        return y + self.swiglu(self.rmsnorm2.forward(y))