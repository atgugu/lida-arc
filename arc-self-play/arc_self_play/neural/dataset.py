"""PyTorch datasets built from self-play puzzles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .encoding import encode_grid, encode_grid_pair
from .labels import (
    TASK_CATEGORIES,
    extract_task_label,
    extract_task_index,
    extract_primitive_labels,
    extract_composition_sequences,
)


class SelfPlayDataset(Dataset):
    """One example per puzzle: grid pair + task label + primitive labels.

    Args:
        puzzles_path: Path to ``generated_puzzles.json``.
        all_primitives: Ordered list of primitive names (defines label indices).
    """

    def __init__(self, puzzles_path: str, all_primitives: List[str]) -> None:
        self.all_primitives = all_primitives
        self.prim_to_idx = {n: i for i, n in enumerate(all_primitives)}

        with open(puzzles_path) as f:
            raw = json.load(f)

        self.examples: List[Dict[str, Any]] = []
        for p in raw:
            train = p.get("train", [])
            if not train:
                continue
            inp = train[0]["input"] if isinstance(train[0], dict) else train[0][0]
            out = train[0]["output"] if isinstance(train[0], dict) else train[0][1]

            transform = []
            for t in p.get("transformation", []):
                if isinstance(t, dict):
                    transform.append((t["op"], t.get("params", {})))
                else:
                    transform.append(tuple(t))

            self.examples.append({
                "input_grid": inp,
                "output_grid": out,
                "transformation": transform,
                "task_index": extract_task_index(transform),
                "primitive_labels": extract_primitive_labels(transform, all_primitives),
                "puzzle_id": p.get("puzzle_id", ""),
                "difficulty": p.get("difficulty", ""),
                "train": [(
                    d["input"] if isinstance(d, dict) else d[0],
                    d["output"] if isinstance(d, dict) else d[1],
                ) for d in train],
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "grid_pair": encode_grid_pair(ex["input_grid"], ex["output_grid"]),
            "task_index": torch.tensor(ex["task_index"], dtype=torch.long),
            "primitive_labels": torch.from_numpy(ex["primitive_labels"]),
            "puzzle_id": ex["puzzle_id"],
            "difficulty": ex["difficulty"],
        }

    def get_composition_examples(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ex in self.examples:
            p = {
                "train": ex["train"],
                "transformation": ex["transformation"],
            }
            out.extend(extract_composition_sequences(p, self.prim_to_idx))
        return out

    def print_stats(self) -> None:
        tasks = {c: 0 for c in TASK_CATEGORIES}
        diffs: Dict[str, int] = {}
        prim_usage = np.zeros(len(self.all_primitives))

        for ex in self.examples:
            tasks[extract_task_label(ex["transformation"])] += 1
            d = ex["difficulty"]
            diffs[d] = diffs.get(d, 0) + 1
            prim_usage += ex["primitive_labels"]

        n = len(self.examples)
        print(f"\nDataset: {n} puzzles, {len(self.all_primitives)} primitives")
        print("\nTask distribution:")
        for t, c in tasks.items():
            print(f"  {t:15s} {c:4d} ({c/n:.1%})")
        print("\nDifficulty distribution:")
        for d in ("easy", "medium", "hard"):
            c = diffs.get(d, 0)
            print(f"  {d:8s} {c:4d} ({c/n:.1%})")


class CompositionDataset(Dataset):
    """One example per step in a composition sequence."""

    def __init__(self, base: SelfPlayDataset, max_seq: int = 5) -> None:
        self.max_seq = max_seq
        self.examples = base.get_composition_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        seq = ex["primitive_sequence"]
        slen = min(len(seq), self.max_seq)
        padded = (seq + [0] * self.max_seq)[:self.max_seq]
        mask = [1.0] * slen + [0.0] * (self.max_seq - slen)

        return {
            "input_grid": encode_grid(ex["input_grid"]),
            "primitive_sequence": torch.tensor(padded, dtype=torch.long),
            "sequence_mask": torch.tensor(mask),
            "next_primitive": torch.tensor(ex["next_primitive"], dtype=torch.long),
            "sequence_length": torch.tensor(slen, dtype=torch.long),
        }


def create_dataloaders(
    puzzles_path: str,
    all_primitives: List[str],
    batch_size: int = 32,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SelfPlayDataset]:
    """Build train / val / test loaders from a puzzle JSON file."""
    ds = SelfPlayDataset(puzzles_path, all_primitives)

    n = len(ds)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    n_test = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    kw: Dict[str, Any] = {"num_workers": 0}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(val_ds,   batch_size=batch_size, **kw),
        DataLoader(test_ds,  batch_size=batch_size, **kw),
        ds,
    )
