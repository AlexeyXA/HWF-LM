"""HWF-LM: Hierarchical Water-Filling Language Model."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from hwf_lm.config import HWFConfig
from hwf_lm.models.ssm import SSMRegion
from hwf_lm.models.nca import NCARegion
from hwf_lm.models.attention import AttnRegion
from hwf_lm.models.moe import SparseMoE_FFN, load_balancing_loss


class HWF_LM(nn.Module):
    """
    Three-region hybrid LM:
      Region 1: SSM (local / short range)
      Region 2: NCA (mid range)
      Region 3: Sparse Attention (long range)
    Each region followed by sparse MoE FFN.
    """

    def __init__(self, config: HWFConfig):
        super().__init__()
        self.config = config
        n_ssm, n_nca, n_attn = config.n_layers_per_region

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)

        self.ssm = SSMRegion(config.d_model, n_ssm, config.ssm_state, config.dropout)
        self.nca = NCARegion(config.d_model, n_nca, config.nca_k, config.nca_T, config.dropout)
        self.attn = AttnRegion(
            config.d_model, n_attn, config.n_heads,
            local_window=config.local_window,
            n_global=config.n_global_tokens,
            dropout=config.dropout,
        )

        self.moe_ssm = SparseMoE_FFN(
            config.d_model, config.num_experts, config.moe_top_k, config.ffn_mult, config.dropout
        )
        self.moe_nca = SparseMoE_FFN(
            config.d_model, config.num_experts, config.moe_top_k, config.ffn_mult, config.dropout
        )
        self.moe_attn = SparseMoE_FFN(
            config.d_model, config.num_experts, config.moe_top_k, config.ffn_mult, config.dropout
        )

        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.emb.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, return_aux: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        B, N = input_ids.shape
        assert N <= self.config.max_len

        pos = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h = self.emb(input_ids) + self.pos_emb(pos)
        aux = torch.tensor(0.0, device=h.device)

        h = self.ssm(h)
        out, logits = self.moe_ssm(h)
        h = h + out
        if return_aux:
            aux = aux + load_balancing_loss(logits, self.config.num_experts, self.config.moe_top_k)

        h = self.nca(h)
        out, logits = self.moe_nca(h)
        h = h + out
        if return_aux:
            aux = aux + load_balancing_loss(logits, self.config.num_experts, self.config.moe_top_k)

        h = self.attn(h)
        out, logits = self.moe_attn(h)
        h = h + out
        if return_aux:
            aux = aux + load_balancing_loss(logits, self.config.num_experts, self.config.moe_top_k)

        logits_out = self.lm_head(self.norm_f(h))
        if return_aux:
            return logits_out, aux
        return logits_out

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.weight.numel()
        return n


def build_model_from_config(config: HWFConfig) -> HWF_LM:
    return HWF_LM(config)
