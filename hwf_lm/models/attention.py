"""Sparse Attention: local sliding window + global tokens.

PyTorch SDPA boolean convention: True = ALLOWED to attend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    """
    Causal sparse attention:
    - Every token attends to a local window of the past (local_window)
    - First `n_global` tokens are global (attended by everyone, causally)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        local_window: int = 128,
        n_global: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.n_global = n_global
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def _build_sparse_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Boolean mask for SDPA: True = ALLOWED to attend.
        """
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)

        local_ok = (diff >= 0) & (diff <= self.local_window)

        is_global_key = idx < self.n_global
        global_ok = is_global_key.unsqueeze(0).expand(N, N) & (diff >= 0)

        allowed = local_ok | global_ok
        return allowed.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is None:
            mask = self._build_sparse_mask(N, x.device)

        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class AttnRegion(nn.Module):
    """Stack of sparse attention blocks with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_heads: int,
        local_window: int = 128,
        n_global: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SparseAttention(d_model, num_heads, local_window, n_global, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for norm, layer, drop in zip(self.norms, self.layers, self.dropouts):
            x = x + drop(layer(norm(x), mask))
        return x
