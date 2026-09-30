#!/usr/bin/env python3
"""
Download standard benchmark datasets for MathAgent evaluation.

Benchmarks:
  - miniF2F (488 problems, Lean 4 + NL)
  - PutnamBench (672 Lean 4 problems, 9 domain categories)
  - MathOlympiadBench (360 Olympiad problems, Lean 4)

Usage:
    python -m src.benchmarks.download [--benchmark minif2f|putnambench|matholympiadbench|all]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "benchmarks"


def download_matholympiadbench():
    """Download MathOlympiadBench from HuggingFace."""
    output_path = DATA_DIR / "matholympiadbench.json"
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    print("  Downloading MathOlympiadBench from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Goedel-LM/MathOlympiadBench", split="train")
        entries = []
        for row in ds:
            entries.append({
                "problem_id": row.get("problem_id") or row.get("name", ""),
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "tags": row.get("tags", []),
                "formal_statement": row.get("formal_statement", ""),
                "informal_prefix": row.get("informal_prefix", ""),
                "lean4_code": row.get("lean4_code", ""),
                "header": row.get("header", ""),
                "solved": row.get("solved", False),
            })
        with open(output_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"  Saved {len(entries)} problems to {output_path}")

        # Stats
        cats = {}
        for e in entries:
            c = e.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        print(f"  Categories: {cats}")
        return output_path
    except ImportError:
        print("  ERROR: `pip install datasets` required")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def download_minif2f():
    """Download miniF2F-lean4 from HuggingFace."""
    output_path = DATA_DIR / "minif2f.json"
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    print("  Downloading miniF2F from HuggingFace...")
    try:
        from datasets import load_dataset
        entries = []
        for split_name in ["validation", "test"]:
            ds = load_dataset("cat-searcher/minif2f-lean4", split=split_name)
            for row in ds:
                entries.append({
                    "problem_id": row.get("id", ""),
                    "split": split_name,
                    "formal_statement": row.get("formal_statement", ""),
                    "header": row.get("header", ""),
                    "informal_stmt": row.get("informal_stmt", ""),
                    "informal_proof": row.get("informal_proof", ""),
                })
        with open(output_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"  Saved {len(entries)} problems to {output_path}")

        # Infer domains from problem IDs
        domains = {}
        for e in entries:
            pid = e["problem_id"]
            if "numbertheory" in pid.lower():
                d = "Number Theory"
            elif "algebra" in pid.lower():
                d = "Algebra"
            elif "amc" in pid.lower() or "aime" in pid.lower():
                d = "Competition"
            elif "imo" in pid.lower():
                d = "IMO"
            else:
                d = "Other"
            domains[d] = domains.get(d, 0) + 1
        print(f"  Domains (inferred): {domains}")
        return output_path
    except ImportError:
        print("  ERROR: `pip install datasets` required")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def download_putnambench():
    """Download PutnamBench by cloning the GitHub repo."""
    output_dir = DATA_DIR / "putnambench"
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"  Already exists: {output_dir}")
        return output_dir

    print("  Cloning PutnamBench from GitHub...")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/trishullab/PutnamBench.git",
             str(output_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  ERROR: git clone failed: {result.stderr[:200]}")
            return None

        # Count Lean 4 files
        lean_files = list(output_dir.rglob("*.lean"))
        print(f"  Cloned. Found {len(lean_files)} .lean files")

        # Try to find category info
        readme = output_dir / "README.md"
        if readme.exists():
            print(f"  README exists at {readme}")
        return output_dir
    except subprocess.TimeoutExpired:
        print("  ERROR: git clone timed out")
        return None
    except FileNotFoundError:
        print("  ERROR: git not found")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--benchmark",
        choices=["minif2f", "putnambench", "matholympiadbench", "all"],
        default="all",
        help="Which benchmark to download (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("BENCHMARK DATASET DOWNLOAD")
    print("=" * 60)

    if args.benchmark in ("matholympiadbench", "all"):
        print("\n[1/3] MathOlympiadBench (360 problems)")
        download_matholympiadbench()

    if args.benchmark in ("minif2f", "all"):
        print("\n[2/3] miniF2F (488 problems)")
        download_minif2f()

    if args.benchmark in ("putnambench", "all"):
        print("\n[3/3] PutnamBench (672 Lean 4 problems)")
        download_putnambench()

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file() and f.suffix in (".json", ".lean"):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(DATA_DIR)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
