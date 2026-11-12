"""Data loading utilities for HWF-LM (TinyStories by default).

Stories are split into train/val *before* windowing to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import tiktoken
from typing import Tuple, List


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def _stories_to_windows(story_ids: List[List[int]], seq_len: int) -> np.ndarray:
    """Concatenate tokens within each story only, then cut windows. No cross-story windows."""
    windows = []
    for ids in story_ids:
        if len(ids) < seq_len + 1:
            continue
        n = len(ids) // (seq_len + 1)
        if n == 0:
            continue
        arr = np.array(ids[: n * (seq_len + 1)], dtype=np.int64)
        windows.append(arr.reshape(n, seq_len + 1))
    if not windows:
        raise RuntimeError("No windows produced — increase num_stories or decrease seq_len")
    return np.concatenate(windows, axis=0)


def load_tinystories_tokens(
    num_stories: int = 5000,
    seq_len: int = 512,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split by *stories* first, then window independently."""
    print(f"Loading TinyStories ({num_stories} stories)...")
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{num_stories}]")
    enc = get_tokenizer()

    rng = np.random.default_rng(seed)
    n = len(ds)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = set(perm[:n_val].tolist())

    train_stories, val_stories = [], []
    for i, text in enumerate(ds["text"]):
        ids = enc.encode_ordinary(text)
        if i in val_idx:
            val_stories.append(ids)
        else:
            train_stories.append(ids)

    train_w = _stories_to_windows(train_stories, seq_len)
    val_w = _stories_to_windows(val_stories, seq_len)
    train_w = train_w[rng.permutation(len(train_w))]
    val_w = val_w[rng.permutation(len(val_w))]

    print(f"Stories: train={len(train_stories)}, val={len(val_stories)}")
    print(f"Windows: train={len(train_w)}, val={len(val_w)}")
    return train_w, val_w, enc.n_vocab


def make_dataloader(windows, batch_size, shuffle=True, drop_last=True):
    x = torch.from_numpy(windows[:, :-1].copy())
    y = torch.from_numpy(windows[:, 1:].copy())
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_dataloaders(num_stories=5000, seq_len=256, batch_size=16, val_split=0.1, seed=42):
    train_w, val_w, vocab_size = load_tinystories_tokens(num_stories, seq_len, val_split, seed)
    return (
        make_dataloader(train_w, batch_size, shuffle=True),
        make_dataloader(val_w, batch_size, shuffle=False, drop_last=False),
        vocab_size,
    )
