"""Load EventX data from Hugging Face or local files.

Supports two local directory layouts:
  1. Prepared HF layout: data/t1/train.jsonl, data/t1/test.jsonl, ...
  2. Original raw layout: task1/groundtruth/..., task4/t4_labels.jsonl, ...

The loader auto-detects which layout is present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

# Default HF repo -- update after publishing
HF_REPO = "mlsys-io/EventXBench"

VALID_TASKS = {"t1", "t2", "t3", "t4", "t5", "t6", "t7"}


def load_task(
    task: str,
    repo: str = HF_REPO,
    local_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load a task dataset.

    Args:
        task: Task name (t1-t7).
        repo: Hugging Face dataset repo ID.
        local_dir: If set, load from local directory instead of HF.
        split: If set, return only this split ("train", "validation"/"val", or "test").
               If None, returns (train, test) tuple for tasks with both splits,
               or just test DataFrame for tasks with only test.

    Returns:
        DataFrame or (train_df, test_df) tuple.
    """
    task = task.lower().strip()
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid: {sorted(VALID_TASKS)}")

    if local_dir:
        return _load_local(task, Path(local_dir), split)
    return _load_hf(task, repo, split)


def _load_hf(task: str, repo: str, split: Optional[str]):
    from datasets import load_dataset

    ds = load_dataset(repo, task, trust_remote_code=True)

    if split:
        split_name = _normalize_split_name(split, ds.keys())
        return ds[split_name].to_pandas()

    splits = list(ds.keys())
    if "train" in splits and "test" in splits:
        return ds["train"].to_pandas(), ds["test"].to_pandas()
    elif "test" in splits:
        return ds["test"].to_pandas()
    elif "train" in splits:
        # e.g. T3: {train, gold} - no "test" split exists at all.
        return ds["train"].to_pandas()
    else:
        return ds[splits[0]].to_pandas()


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# Prepared HF layout: data/t1/train.jsonl, data/t1/test.jsonl
_HF_LAYOUT = {
    "t1": {
        "train": "t1/train.jsonl",
        "val": "t1/val.jsonl",
        "validation": "t1/validation.jsonl",
        "test": "t1/test.jsonl",
    },
    "t2": {
        "train": "t2/t2_train.jsonl",
        "val": "t2/t2_val.jsonl",
        "validation": "t2/t2_val.jsonl",
        "test": "t2/t2_test.jsonl",
    },
    # "train" = full silver-labeled export (`final_grade`, label_source auto/llm).
    # Named "train" (not "test") because baselines self-split it 70/30 by
    # condition_id at runtime rather than treating it as a held-out set.
    # "gold" = the separate, rare-grade-enriched, human-adjudicated audit pool
    # (`gold_grade`) - the actual held-out ground truth, see T3_Reproducible_Package.
    "t3": {"train": "t3/train.jsonl", "gold": "t3/gold.jsonl"},
    "t4": {
        "train": "t4/train.jsonl",
        "val": "t4/validation.jsonl",
        "validation": "t4/validation.jsonl",
        "test": "t4/test.jsonl",
    },
    "t5": {
        "train": "t5/train.jsonl",
        "val": "t5/validation.jsonl",
        "validation": "t5/validation.jsonl",
        "test": "t5/test.jsonl",
    },
    "t6": {
        "full": "t6/t6_full_with_split.jsonl",
        "train": "t6/train.jsonl",
        "val": "t6/val.jsonl",
        "validation": "t6/validation.jsonl",
        "test": "t6/test.jsonl",
    },
    "t7": {"train": "t7/train.jsonl", "test": "t7/test.jsonl"},
}

# Original raw layout from EventX/ directory
_RAW_LAYOUT = {
    "t1": {
        "train": "task1/groundtruth/t1_market_level_train_premarket_only_new.jsonl",
        "test": "task1/groundtruth/t1_market_level_test_premarket_only_new.jsonl",
    },
    "t2": {"test": "task2/t2_groundtruth.jsonl"},
    "t3": {"train": "task3/t3_final_graded.json", "gold": "task3/t3_gold_pool.json"},
    "t4": {"full": "task4/t4_labels.jsonl"},
    "t5": {"full": "task5+7/t5(7)_label.jsonl"},
    "t6": {
        "full": (
            "task6/t6_full_with_split.jsonl",
            "task6/t6_labels_with_split.jsonl",
            "task6/task6_labels_v2_tuned_t35confound_full.jsonl",
        )
    },
    "t7": {"full": "task5+7/t5(7)_label.jsonl"},
}


def _detect_layout(data_dir: Path, task: str) -> str:
    """Auto-detect which directory layout is present."""
    if any((data_dir / rel_path).exists() for rel_path in _HF_LAYOUT[task].values()):
        return "hf"
    # The versioned KDD builders expose their own directory as ``--local-dir``
    # (for example ``KDD/data/t1_kdd_v1`` or ``t2_kdd_v1``), so split files
    # are immediately inside data_dir rather than under a second task directory.
    if task in ("t1", "t2", "t4", "t5", "t6") and any(
        (data_dir / Path(rel_path).name).exists()
        for rel_path in _HF_LAYOUT[task].values()
    ):
        return "hf"
    return "raw"


def _is_direct_layout(data_dir: Path, task: str) -> bool:
    if task not in ("t1", "t2", "t4", "t5", "t6"):
        return False
    nested_exists = any(
        (data_dir / rel_path).exists() for rel_path in _HF_LAYOUT[task].values()
    )
    direct_exists = any(
        (data_dir / Path(rel_path).name).exists()
        for rel_path in _HF_LAYOUT[task].values()
    )
    return direct_exists and not nested_exists


def _resolve_prepared_path(data_dir: Path, task: str, relative_path: str) -> Path:
    """Resolve nested prepared paths plus the direct versioned task directory."""
    if _is_direct_layout(data_dir, task):
        return data_dir / Path(relative_path).name
    return data_dir / relative_path


def _resolve_layout_path(data_dir: Path, relative_path):
    if isinstance(relative_path, (tuple, list)):
        for candidate in relative_path:
            path = data_dir / candidate
            if path.exists():
                return path
        return data_dir / relative_path[0]
    return data_dir / relative_path


def _normalize_split_name(split: str, available) -> str:
    available_set = set(available)
    if split in available_set:
        return split
    aliases = {
        "val": "validation",
        "validation": "val",
    }
    alias = aliases.get(split)
    if alias in available_set:
        return alias
    return split


def _load_local(task: str, data_dir: Path, split: Optional[str]):
    layout = _detect_layout(data_dir, task)

    if layout == "hf":
        return _load_hf_layout(task, data_dir, split)
    return _load_raw_layout(task, data_dir, split)


def _load_hf_layout(task: str, data_dir: Path, split: Optional[str]):
    """Load from prepared HF directory structure."""
    files = _HF_LAYOUT[task]

    if "full" in files and _resolve_prepared_path(
        data_dir, task, files["full"]
    ).exists():
        df = _load_jsonl(_resolve_prepared_path(data_dir, task, files["full"]))
        if split:
            split_value = "val" if split == "validation" else split
            return df[df["split"] == split_value].reset_index(drop=True)
        return (
            df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True),
        )

    if split:
        # Some prepared datasets call the development split ``val`` and some
        # call it ``validation``.  Resolve aliases against files that actually
        # exist, rather than merely against the declared layout: both names are
        # declared above so that either on-disk convention is supported.
        available_files = {
            key: _resolve_prepared_path(data_dir, task, relative_path)
            for key, relative_path in files.items()
            if key != "full"
            and _resolve_prepared_path(data_dir, task, relative_path).exists()
        }
        split_key = _normalize_split_name(split, available_files.keys())
        if split_key not in available_files:
            available = sorted(available_files.keys())
            raise ValueError(f"Task {task} has no '{split}' split. Available: {available}")
        return _load_jsonl(available_files[split_key])

    if "train" in files and "test" in files:
        return (
            _load_jsonl(_resolve_prepared_path(data_dir, task, files["train"])),
            _load_jsonl(_resolve_prepared_path(data_dir, task, files["test"])),
        )
    if "test" in files:
        return _load_jsonl(_resolve_prepared_path(data_dir, task, files["test"]))
    if "train" in files:
        # e.g. T3: {train, gold} - no "test" split exists at all.
        return _load_jsonl(_resolve_prepared_path(data_dir, task, files["train"]))

    first_key = next(iter(files))
    return _load_jsonl(_resolve_prepared_path(data_dir, task, files[first_key]))


def _load_raw_layout(task: str, data_dir: Path, split: Optional[str]):
    """Load from original EventX/ raw data directory."""
    files = _RAW_LAYOUT[task]

    if "full" in files:
        path = _resolve_layout_path(data_dir, files["full"])
        if path.suffix == ".json":
            df = pd.read_json(path)
        else:
            df = _load_jsonl(path)

        if "split" in df.columns:
            if split:
                split_value = "val" if split == "validation" else split
                return df[df["split"] == split_value].reset_index(drop=True)
            if {"train", "test"}.issubset(set(df["split"].dropna().unique())):
                train_df = df[df["split"] == "train"].reset_index(drop=True)
                test_df = df[df["split"] == "test"].reset_index(drop=True)
                return train_df, test_df

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        if split == "train":
            return train_df
        elif split == "test":
            return test_df
        return train_df, test_df

    if split:
        if split not in files:
            available = sorted(files.keys())
            raise ValueError(f"Task {task} has no '{split}' split. Available: {available}")
        path = data_dir / files[split]
        if path.suffix == ".json":
            return pd.read_json(path)
        return _load_jsonl(path)

    if "train" in files and "test" in files:
        return _load_jsonl(data_dir / files["train"]), _load_jsonl(data_dir / files["test"])

    if "test" in files:
        target_key = "test"
    elif "train" in files:
        # e.g. T3: {train, gold} - no "test" split exists at all.
        target_key = "train"
    else:
        target_key = next(iter(files))

    target_path = data_dir / files[target_key]
    if target_path.suffix == ".json":
        return pd.read_json(target_path)
    return _load_jsonl(target_path)


def load_markets(repo: str = HF_REPO, local_path: Optional[str] = None) -> pd.DataFrame:
    """Load market metadata."""
    if local_path:
        return pd.read_json(local_path)
    from datasets import load_dataset
    ds = load_dataset(repo, "markets")
    return ds[list(ds.keys())[0]].to_pandas()


def load_ohlcv(repo: str = HF_REPO, local_path: Optional[str] = None) -> pd.DataFrame:
    """Load market OHLCV time series."""
    if local_path:
        return pd.read_json(local_path)
    from datasets import load_dataset
    ds = load_dataset(repo, "ohlcv")
    return ds[list(ds.keys())[0]].to_pandas()
