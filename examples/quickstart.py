#!/usr/bin/env python3
"""
EventXBench quickstart example.

Demonstrates how to:
  1. Load a task from HuggingFace
  2. Inspect the data
  3. Run a simple majority-class baseline
  4. Evaluate predictions with the EventXBench evaluation module
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

# ------------------------------------------------------------------
# 0.  Make sure the repo root is importable
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------
# 1.  Load Task 1 data from HuggingFace
# ------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading Task 1 (Conditional Market Volume Prediction)")
print("=" * 60)

try:
    from eventxbench import load_task

    train_data = load_task("t1", split="train")
    test_data = load_task("t1", split="test")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test  samples: {len(test_data)}")
except ImportError:
    print("  eventxbench package not found; using dummy data for demo.\n")
    # Dummy data so the rest of the script still runs
    train_data = [
        {"condition_id": f"c{i}", "label": ["high_interest", "moderate_interest", "low_interest"][i % 3]}
        for i in range(90)
    ]
    test_data = [
        {"condition_id": f"t{i}", "label": ["high_interest", "moderate_interest", "low_interest"][i % 3]}
        for i in range(30)
    ]
    print(f"  (dummy) Train samples: {len(train_data)}")
    print(f"  (dummy) Test  samples: {len(test_data)}")

# ------------------------------------------------------------------
# 2.  Inspect the data
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Inspect the data")
print("=" * 60)

train_labels = train_data["label"].tolist()
label_counts = Counter(train_labels)
print("  Training label distribution:")
for label, count in sorted(label_counts.items()):
    print(f"    {label}: {count}  ({count / len(train_labels):.1%})")

print(f"\n  Sample record: {json.dumps(train_data.iloc[0].to_dict(), default=str)}")

# ------------------------------------------------------------------
# 3.  Majority-class baseline
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Majority-class baseline")
print("=" * 60)

majority_label = label_counts.most_common(1)[0][0]
print(f"  Majority label: {majority_label}")

predictions = [
    {"condition_id": row["condition_id"], "label": majority_label}
    for _, row in test_data.iterrows()
]
print(f"  Generated {len(predictions)} predictions")

# ------------------------------------------------------------------
# 4.  Evaluate
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Evaluate with EventXBench metrics")
print("=" * 60)

from evaluation.metrics import accuracy, macro_f1

T1_LABELS = ["high_interest", "moderate_interest", "low_interest"]

y_true = test_data["label"].tolist()
y_pred = [majority_label] * len(test_data)

f1 = macro_f1(y_true, y_pred, labels=T1_LABELS)
acc = accuracy(y_true, y_pred)

print(f"  Macro-F1:  {f1:.4f}")
print(f"  Accuracy:  {acc:.4f}")

# Optionally, use the CLI evaluator by writing temp files
print("\n  (You can also evaluate via CLI:)")
print("    python evaluation/evaluate.py --task t1 --predictions preds.jsonl --gold gold.jsonl")

# ------------------------------------------------------------------
# 5.  Write predictions to a temp file for reference
# ------------------------------------------------------------------
tmp = Path(tempfile.gettempdir()) / "eventxbench_quickstart_preds.jsonl"
with open(tmp, "w") as fh:
    for p in predictions:
        fh.write(json.dumps(p) + "\n")
print(f"\n  Predictions written to: {tmp}")

print("\n" + "=" * 60)
print("Done! See evaluation/README.md for full prediction format details.")
print("=" * 60)
