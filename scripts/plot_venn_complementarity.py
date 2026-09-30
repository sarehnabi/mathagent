#!/usr/bin/env python3
"""
Generate paper_iclr/figures/plot_venn_complementarity.pdf.

Three Venn diagrams (miniF2F, PutnamBench, MathOlympiadBench) for Claude Sonnet 4.6:
each region counts problems solved by that combination of augmented modes but NOT by
prover_only. Below each diagram: prover_only baseline, the oracle (union of the four modes),
and the problems solved only by prover_only ("Lost").

Usage:
    python scripts/plot_venn_complementarity.py [--out PATH] [--fontscale F]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

RUN_DIR = Path("outputs/experiments/full_ablation_05162026_2359_claude_sonnet")
BENCHES = [("minif2f", "miniF2F"), ("putnambench", "PutnamBench"),
           ("matholympiadbench", "MathOlympiadBench")]
MODES = ["prover_only", "with_kg", "with_mathlib", "full_system"]


VALID_JSON = Path("outputs/validation/valid_solves.json")  # corrected solve sets (statement check)


def solved(bench, mode, use_valid=True, run_dir=None):
    """Solve set for one run: the corrected set from outputs/validation/valid_solves.json when that file
    exists and has this run's key; otherwise (or with use_valid=False, or when run_dir is given) every
    'verified' record of the run."""
    if run_dir is not None:
        f = glob.glob(str(run_dir / f"{bench}_{mode}" / f"eval_{bench}_{mode}.json"))[0]
        return {r["problem_id"] for r in json.load(open(f))["results"] if r["status"] == "verified"}
    if use_valid and VALID_JSON.exists():
        key = f"Sonnet 4.6|{bench}|{mode}"
        valid = json.load(open(VALID_JSON))
        if key in valid:
            return set(valid[key])
    f = glob.glob(str(RUN_DIR / f"{bench}_{mode}" / f"eval_{bench}_{mode}.json"))[0]
    res = json.load(open(f))["results"]
    return {r["problem_id"] for r in res if r["status"] == "verified"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_iclr/figures/plot_venn_complementarity.pdf")
    ap.add_argument("--fontscale", type=float, default=1.0)
    ap.add_argument("--raw-benchmarks", default="",
                    help="comma-separated benchmark keys to draw from raw 'verified' records instead of the corrected sets")
    ap.add_argument("--bench-dir", default="",
                    help="BENCH=DIR: read that benchmark's result files from DIR instead of RUN_DIR, counting its "
                         "'verified' records as final (used for the merged PutnamBench run)")
    ap.add_argument("--exclude-ids", default="",
                    help="BENCH=FILE: drop the problem ids listed in FILE (one per line) from that benchmark's panel, "
                         "e.g. putnambench=outputs/validation/putnam_answer_slot_ids.txt for the 326 slot-free problems")
    args = ap.parse_args()
    raw = set(b for b in args.raw_benchmarks.split(",") if b)
    bench_dir = {}
    if args.bench_dir:
        bk, bd = args.bench_dir.split("=", 1)
        bench_dir[bk] = Path(bd)
    excl = {}
    if args.exclude_ids:
        bench_key, path = args.exclude_ids.split("=", 1)
        excl[bench_key] = set(Path(path).read_text().split())
    fs = args.fontscale

    fig, axes = plt.subplots(1, 3, figsize=(13.95, 5.4))
    for ax, (bench, title) in zip(axes, BENCHES):
        S = {m: solved(bench, m, use_valid=bench not in raw, run_dir=bench_dir.get(bench)) - excl.get(bench, set()) for m in MODES}
        if bench in excl:
            n_total = len({r["problem_id"] for r in json.load(open(glob.glob(str(RUN_DIR / f"{bench}_prover_only" / "eval_*.json"))[0]))["results"]}) - len(excl[bench])
            title = f"{title} ({n_total})"
        po = S["prover_only"]
        K, M, F = (S[m] - po for m in MODES[1:])
        # equal-sized circles in every panel: layout uses fixed unit areas, labels show true counts
        v = venn3([K, M, F], set_labels=("+KG", "+Mathlib", "+Full"),
                  set_colors=("#3cb371", "#1e90ff", "#ff8c00"), alpha=0.55, ax=ax,
                  layout_algorithm=DefaultLayoutAlgorithm(normalize_to=0.86,
                                                          fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1)))
        for lbl in v.set_labels:
            if lbl is not None:
                lbl.set_fontsize(15 * fs)
        # move "+Full" to the lower-right of the bottom circle, clear of the footer line
        full_lbl = v.get_label_by_id("C")
        full_circle = v.get_circle_center(2)
        full_r = v.get_circle_radius(2)
        if full_lbl is not None:
            full_lbl.set_position((full_circle.x + 0.72 * full_r, full_circle.y - 0.72 * full_r))
            full_lbl.set_ha("left")
            full_lbl.set_va("center")
        for lbl in v.subset_labels:
            if lbl is not None:
                lbl.set_fontsize(14 * fs)
        ax.set_title(title, fontsize=16 * fs, fontweight="bold")
        union = po | K | M | F
        lost = len(po - (S["with_kg"] | S["with_mathlib"] | S["full_system"]))
        ax.text(0.5, -0.02, f"Baseline: {len(po)} | Oracle: {len(union)} | Lost: {lost}",
                transform=ax.transAxes, ha="center", va="top", fontsize=15 * fs)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
