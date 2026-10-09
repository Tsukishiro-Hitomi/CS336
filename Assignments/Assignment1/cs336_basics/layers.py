import torch
import math
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        
        super().__init__()

        # (out_features, in_features)
        self.W = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )

        # initialization
        std = math.sqrt(2 / (out_features + in_features))
        nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., in_features) -> (..., out_features)
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int,
                 embedding_dim: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()

        # (num_embeddings, embedding_dim)
        self.embedding = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        # initialization
        nn.init.trunc_normal_(self.embedding, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()

        # initialization
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # check x's shape: (..., d_model)
        assert x.shape[-1] == self.d_model, f"the last dimension of x must be equal to {self.d_model}"
        in_dtype = x.dtype

        # UPCAST input to torch.float32 to prevent overflow when squaring
        x = x.to(torch.float32)

        # (..., 1)
        rms_x = torch.sqrt(torch.mean(torch.square(x), axis=-1, keepdim=True) + self.eps)
        result = einsum(x, self.g, "... d_model, d_model -> ... d_model") / rms_x
        return result.to(in_dtype)

