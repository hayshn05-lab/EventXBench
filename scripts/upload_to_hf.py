#!/usr/bin/env python3
"""Upload EventX data to Hugging Face Hub.

Two-step workflow:
  1. First run prepare_hf_data.py to create clean data/t1..t6 directories
  2. Then run this script to push to HF

Usage:
    # Step 1: Prepare data
    python scripts/prepare_hf_data.py --source-dir ../EventX --output-dir data/

    # Step 2: Upload to HF
    export HF_TOKEN=hf_...
    python scripts/upload_to_hf.py --repo mlsys-io/EventXBench

    # Upload specific tasks only
    python scripts/upload_to_hf.py --repo mlsys-io/EventXBench --tasks t1 t4

    # Also upload large raw files (posts, markets, OHLCV)
    python scripts/upload_to_hf.py --repo mlsys-io/EventXBench --include-large-files --raw-dir ../EventX

    # Dry run
    python scripts/upload_to_hf.py --repo mlsys-io/EventXBench --dry-run
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


ALL_TASKS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def main():
    parser = argparse.ArgumentParser(description="Upload EventX to Hugging Face")
    parser.add_argument("--repo", required=True, help="HF repo ID, e.g. mlsys-io/EventXBench")
    parser.add_argument("--data-dir", default="data", help="Prepared data directory (default: data/)")
    parser.add_argument("--tasks", nargs="*", default=ALL_TASKS, help="Tasks to upload")
    parser.add_argument("--include-large-files", action="store_true",
                        help="Upload posts, market metadata, and OHLCV")
    parser.add_argument("--raw-dir", default="../EventX",
                        help="Path to raw EventX/ dir (for large files)")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("Set HF_TOKEN environment variable")

    api = HfApi(token=token)

    # Create repo
    if not args.dry_run:
        api.create_repo(repo_id=args.repo, repo_type="dataset",
                        private=args.private, exist_ok=True)
        print(f"Repository: https://huggingface.co/datasets/{args.repo}")

    # Upload the dataset loading script
    loader_script = Path(__file__).resolve().parent.parent / "EventXBench.py"
    if loader_script.exists():
        print(f"Uploading loading script: {loader_script.name}")
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(loader_script),
                path_in_repo="EventXBench.py",
                repo_id=args.repo, repo_type="dataset",
            )

    # Upload dataset card
    card_path = data_dir / "README.md"
    if not card_path.exists():
        card_path = Path(__file__).resolve().parent.parent / "data" / "README.md"
    if card_path.exists():
        print(f"Uploading dataset card")
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=args.repo, repo_type="dataset",
            )

    # Upload per-task data files
    for task in args.tasks:
        task_dir = data_dir / task
        if not task_dir.exists():
            print(f"SKIP {task}: {task_dir} not found (run prepare_hf_data.py first)")
            continue

        for jsonl_file in sorted(task_dir.glob("*.jsonl")):
            repo_path = f"data/{task}/{jsonl_file.name}"
            n_lines = sum(1 for _ in open(jsonl_file))
            print(f"  {repo_path} ({n_lines} rows)")
            if not args.dry_run:
                api.upload_file(
                    path_or_fileobj=str(jsonl_file),
                    path_in_repo=repo_path,
                    repo_id=args.repo, repo_type="dataset",
                )

    # Upload large raw files
    if args.include_large_files:
        raw_dir = Path(args.raw_dir).resolve()
        large_files = [
            ("posts_no_text.jsonl", "raw/posts_no_text.jsonl"),
            ("market_foundamental.json", "raw/market_fundamental.json"),
            ("market_ohlcv.json", "raw/market_ohlcv.json"),
        ]
        print("\nLarge files:")
        for local_name, repo_path in large_files:
            local_path = raw_dir / local_name
            if local_path.exists():
                size_gb = local_path.stat().st_size / (1024**3)
                print(f"  {repo_path} ({size_gb:.2f} GB)")
                if not args.dry_run:
                    api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=repo_path,
                        repo_id=args.repo, repo_type="dataset",
                    )
            else:
                print(f"  SKIP {local_name} (not found at {local_path})")

    print("\nDone!")
    if not args.dry_run:
        print(f"View: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
