import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self,
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None):
        
        super().__init__()

        assert d_k % 2 == 0
        self.d_k = d_k

        # (k // 2, )
        k = torch.arange(d_k // 2, device=device)

        # (L, )
        positions = torch.arange(max_seq_len, device=device)

        # (d // 2, )
        inv_freq = 1.0 / (theta ** ((2 * k) / d_k))

        # (L, d // 2)
        theta_ik = positions[: , None] * inv_freq[None, :]

        sin_ik =  torch.sin(theta_ik)
        self.register_buffer("sin_ik", sin_ik, persistent=False)
        cos_ik = torch.cos(theta_ik)
        self.register_buffer("cos_ik", cos_ik, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., L, d)
        # token_positions: (..., L)

        assert x.shape[-1] == self.d_k
        assert x.shape[-2] == token_positions.shape[-1]

        # (..., L, d // 2)
        sin = self.sin_ik[token_positions].to(dtype=x.dtype)
        cos = self.cos_ik[token_positions].to(dtype=x.dtype)

        # (..., L, d // 2, 1)
        sin = sin.unsqueeze(-1)
        cos = cos.unsqueeze(-1)

        x_shape = x.shape

        # (..., L, d // 2, 2)
        x = x.reshape(*x.shape[:-1], self.d_k // 2, 2)

        # (..., L, d // 2, 1)
        first = x[..., 0]
        second = x[..., 1]

        rotated_first = first * cos[..., 0] - second * sin[..., 0]
        rotated_second = first * sin[..., 0] + second * cos[..., 0]
        rotated = torch.stack((rotated_first, rotated_second), dim=-1)

        # (..., L, d // 2, 2) -> (..., L, d)
        return rotated.reshape(x_shape)
        
def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    logits = x.clone()
    logits -= torch.max(logits, dim=i, keepdim=True).values
    exp_logits = torch.exp(logits)
    prob = exp_logits / torch.sum(exp_logits, dim=i, keepdim=True)
    return prob

