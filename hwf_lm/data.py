"""Data loading utilities for HWF-LM (TinyStories by default)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import tiktoken
from typing import Tuple


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def load_tinystories_tokens(
    num_stories: int = 5000,
    seq_len: int = 512,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    print(f"Loading TinyStories ({num_stories} stories)...")
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{num_stories}]")
    enc = get_tokenizer()

    all_ids = []
    for story in ds["text"]:
        all_ids.extend(enc.encode_ordinary(story))

    print(f"Total tokens: {len(all_ids):,}")
    all_ids = np.array(all_ids, dtype=np.int64)

    n_win = len(all_ids) // (seq_len + 1)
    windows = all_ids[: n_win * (seq_len + 1)].reshape(n_win, seq_len + 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_win)
    windows = windows[perm]

    n_val = max(1, int(n_win * val_split))
    print(f"Windows: train={n_win - n_val}, val={n_val}")
    return windows[n_val:], windows[:n_val], enc.n_vocab


def make_dataloader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    x = torch.from_numpy(windows[:, :-1].copy())
    y = torch.from_numpy(windows[:, 1:].copy())
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_dataloaders(
    num_stories: int = 5000,
    seq_len: int = 256,
    batch_size: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    train_w, val_w, vocab_size = load_tinystories_tokens(
        num_stories=num_stories, seq_len=seq_len, val_split=val_split, seed=seed
    )
    train_loader = make_dataloader(train_w, batch_size, shuffle=True)
    val_loader = make_dataloader(val_w, batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, vocab_size
