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