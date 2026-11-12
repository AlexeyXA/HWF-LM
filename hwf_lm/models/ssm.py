"""Simplified Mamba-style SSM block.

For production speed install `mamba-ssm` and replace the sequential scan.
This implementation is correct but O(L) sequential and slow on long sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Minimal selective SSM (Mamba-style) with sequential scan."""

    def __init__(self, d_model: int, state_size: int = 64, expand: int = 2, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.state_size = state_size

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=conv_kernel,
            groups=self.d_inner, padding=conv_kernel - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, 2 * state_size + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner)

        A = torch.arange(1, state_size + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [1, self.state_size, self.state_size], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

        # Sequential selective scan (causal)
        h = torch.zeros(B, self.d_inner, self.state_size, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t]
            y = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + self.D * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)


class SSMRegion(nn.Module):
    """Stack of Mamba blocks with pre-norm."""

    def __init__(self, d_model: int, n_layers: int, state_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model, state_size) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer, drop in zip(self.norms, self.layers, self.dropouts):
            x = x + drop(layer(norm(x)))
        return x
