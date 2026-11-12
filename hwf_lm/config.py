"""Configuration dataclasses for HWF-LM."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional
import json
import yaml


@dataclass
class RhoResult:
    """Result of Stage 1 spectrum measurement."""
    rho_0: float
    alpha: float
    H_0: float
    I_inf: float
    max_rel_residual: float

    def is_applicable(self, max_residual: float = 0.05, alpha_range: Tuple[float, float] = (1.05, 2.5)) -> bool:
        return (
            self.max_rel_residual <= max_residual
            and alpha_range[0] <= self.alpha <= alpha_range[1]
            and self.rho_0 > 0
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "RhoResult":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class HWFConfig:
    """Architecture configuration produced by Stage 2."""
    ell_1: int
    ell_2: int
    ssm_state: int
    nca_T: int
    nca_k: int
    n_layers_per_region: Tuple[int, int, int]
    d_model: int = 256
    n_heads: int = 8
    vocab_size: int = 50257
    max_len: int = 2048
    num_experts: int = 8
    moe_top_k: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    local_window: int = 128
    n_global_tokens: int = 4

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = asdict(self)
        data["n_layers_per_region"] = list(self.n_layers_per_region)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HWFConfig":
        with open(path) as f:
            data = json.load(f)
        data["n_layers_per_region"] = tuple(data["n_layers_per_region"])
        return cls(**data)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    seq_len: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 5000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    aux_loss_coef: float = 0.01
    eval_every: int = 200
    save_every: int = 500
    num_stories: int = 5000
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float32"
    output_dir: str = "outputs"
    resume: Optional[str] = None


@dataclass
class MeasureConfig:
    """Stage 1 measurement settings."""
    num_stories: int = 5000
    seq_len: int = 512
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_window: int = 128
    token_budget: int = 2_000_000
    batch_size: int = 16
    lr: float = 3e-4
    windows: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    device: str = "cuda"
    output_dir: str = "outputs/stage1"


def save_yaml(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj, f, default_flow_style=False)
