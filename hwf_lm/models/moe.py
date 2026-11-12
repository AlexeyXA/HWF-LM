"""Sparse Mixture-of-Experts FFN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 4):
        super().__init__()
        hidden = d_model * ffn_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SparseMoE_FFN(nn.Module):
    """Top-k sparse MoE feed-forward."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, ffn_mult) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        returns: (out, router_logits)
        """
        B, N, D = x.shape
        logits = self.gate(x)  # (B, N, E)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_idx[..., k]  # (B, N)
            weight = topk_scores[..., k].unsqueeze(-1)  # (B, N, 1)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    selected = x[mask]
                    expert_out = self.experts[e](selected)
                    out[mask] = out[mask] + weight[mask] * expert_out

        return self.dropout(out), logits


def load_balancing_loss(router_logits: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Switch Transformer style load-balancing auxiliary loss."""
    scores = F.softmax(router_logits, dim=-1)
    density = scores.mean(dim=(0, 1))
    # fraction of tokens routed to each expert (approx via top-1)
    top1 = scores.argmax(dim=-1)
    counts = torch.zeros(num_experts, device=router_logits.device)
    for e in range(num_experts):
        counts[e] = (top1 == e).float().mean()
    return num_experts * (density * counts).sum()
