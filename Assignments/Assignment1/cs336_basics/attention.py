import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.layers import Linear
import math

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

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Q: (..., querys, d_k)
    # K: (..., keys, d_k)
    # V: (..., keys, d_v)
    # mask: (..., querys, keys)
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... querys d_k, ... keys d_k -> ... querys keys") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = softmax(scores, -1)
    return einsum(attn, V, "... querys keys, ... keys d_v -> ... querys d_v")

class multihead_self_attention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q = Linear(d_model, d_model, device, dtype)
        self.k = Linear(d_model, d_model, device, dtype)
        self.v = Linear(d_model, d_model, device, dtype)
        self.o = Linear(d_model, d_model, device, dtype)

    def forward(self, features: torch.Tensor, rope: RoPE, token_positions: torch.Tensor) -> torch.Tensor:
        # features: (..., L, D)
        sequence_length = features.shape[-2]
        features_shape = features.shape

        # (..., L, D) -> (..., L, H, D // H)
        q = self.q.forward(features)
        k = self.k.forward(features)
        v = self.v.forward(features)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.d_model // self.num_heads)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.d_model // self.num_heads)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.d_model // self.num_heads)

        # q, k, v: (..., H, L, D // H)
        q = einsum(q, "... L H D_h -> ... H L D_h")
        k = einsum(k, "... L H D_h -> ... H L D_h")
        v = einsum(v, "... L H D_h -> ... H L D_h")

        # apply rope to q and k, not v
        if rope is not None:
            q = rope.forward(q, token_positions)
            k = rope.forward(k, token_positions)

        # (..., H, L, D // H)
        mask = torch.ones(sequence_length, sequence_length, device=features.device, dtype=torch.bool).tril()
        value = scaled_dot_product_attention(q, k, v, mask)
        # (..., L, D, D // H)
        value = einsum(value, "... H L D_h -> ... L H D_h")
        # (..., L, D)
        value = value.reshape(features_shape)

        return self.o.forward(value)

