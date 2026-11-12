"""Neural Cellular Automata (NCA) style local iterative layers — causal version."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCAOnlyLayer(nn.Module):
    """
    Causal NCA layer: T steps of local neighborhood update using only past + present.
    Window of size `window` means current token + `window` previous tokens.
    """

    def __init__(self, d_model: int, window: int = 3, T: int = 3):
        super().__init__()
        self.window = window  # number of past tokens
        self.T = T
        self.local_proj = nn.Linear(d_model * (window + 1), d_model)
        nn.init.xavier_uniform_(self.local_proj.weight)
        nn.init.zeros_(self.local_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D). Causal: only [t-window ... t]."""
        for _ in range(self.T):
            B, N, D = x.shape
            x_pad = F.pad(x, (0, 0, self.window, 0), mode="replicate")
            x_unf = x_pad.unfold(dimension=1, size=self.window + 1, step=1)
            x_loc = x_unf.reshape(B, N, -1)
            update = F.gelu(self.local_proj(x_loc))
            x = x + update
        return x


class NCARegion(nn.Module):
    """Stack of causal NCA layers with pre-norm."""

    def __init__(self, d_model: int, n_layers: int, window: int, T: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([NCAOnlyLayer(d_model, window, T) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer, drop in zip(self.norms, self.layers, self.dropouts):
            x = x + drop(layer(norm(x)))
        return x
