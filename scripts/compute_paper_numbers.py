#!/usr/bin/env python3
"""
Compute the paper's result tables from the released per-problem result files.

Reads outputs/experiments/<run>/<benchmark>_<mode>/eval_<benchmark>_<mode>.json, the files
released with the paper, and prints:

    Table 2    cross-model pass@32 solve counts on miniF2F
    Table 3    solve-count deltas per model, with exact McNemar p-values (Appendix J, Table 11)
    Table 4    oracle (union of the four modes) per model, and the Figure 3 relative gains
    Table 12   oracle at a matched attempt budget (8 attempts per mode)
    Table 8    Sonnet 4.6 on miniF2F, MathOlympiadBench, PutnamBench: four modes, oracle, recovered, lost
    Table 9    problems each augmented mode solves beyond prover_only, and the Figure 4 Venn regions
    Table 10   Sonnet 4.6 pass@k
    Appendix I context-overflow counts (Goedel-Prover-V2-32B)

A solve is a record with status "verified"; the oracle is the union of the problems solved by any
of the four modes (equal in count to a selector that picks the best mode per problem). pass@k
counts the solved records whose verifying attempt number is at most k. The --check values are
those printed in arXiv:2609.34460v1; run the script from the root of the public repository, whose
result files carry the released statuses.

Usage:
    python scripts/compute_paper_numbers.py                      # print the tables
    python scripts/compute_paper_numbers.py --check              # also compare with the paper's values
    python scripts/compute_paper_numbers.py --results-dir DIR    # result files elsewhere
"""
import argparse
import json
import math
import sys
from pathlib import Path

MODES = ["prover_only", "with_kg", "with_mathlib", "full_system"]
MODELS = ["Qwen3-8B", "Qwen3-32B", "Goedel-8B", "Goedel-32B", "Sonnet 4.6"]
RUNS = {
    "Qwen3-8B": "full_ablation_05182026_2052_qwen3_8b",
    "Qwen3-32B": "full_ablation_08292026_qwen3_32b",
    "Goedel-8B": "full_ablation_08272026_goedel_v2_8b",
    "Goedel-32B": "full_ablation_08272026_goedel_v2_32b",
    "Sonnet 4.6": "full_ablation_05162026_2359_claude_sonnet",
}
SONNET_BENCH = {  # benchmark -> run directory (PutnamBench: the merged 672-problem run)
    "minif2f": RUNS["Sonnet 4.6"],
    "matholympiadbench": RUNS["Sonnet 4.6"],
    "putnambench": "putnam_merged_09262026_claude_sonnet",
}
BENCH_LABEL = {"minif2f": "miniF2F", "matholympiadbench": "MathOlympiadBench", "putnambench": "PutnamBench"}
BENCH_ORDER = ["matholympiadbench", "minif2f", "putnambench"]
PASSK = [1, 3, 5, 10, 20, 32]

# ---- values printed in the paper (arXiv:2609.34460), used by --check ----
EXPECTED = {
    "table2": {"Qwen3-8B": [33, 35, 31, 36], "Qwen3-32B": [62, 48, 62, 46], "Goedel-8B": [205, 219, 204, 205],
               "Goedel-32B": [237, 234, 235, 208], "Sonnet 4.6": [365, 358, 359, 366]},
    "table3": {  # kg-PO, mathlib-PO, full-mathlib, full-max(kg,mathlib)
        "Qwen3-8B": [2, -2, 5, 1], "Qwen3-32B": [-14, 0, -16, -16], "Goedel-8B": [14, -1, 1, -14],
        "Goedel-32B": [-3, -2, -27, -27], "Sonnet 4.6": [-7, -6, 7, 7]},
    "table11_p": {  # McNemar p-values as printed (compared with a tolerance of 0.0105; "<.001" means p < 0.001)
        "Qwen3-8B": [.79, .82, .30, 1.0], "Qwen3-32B": [.007, 1.0, .002, .002], "Goedel-8B": [.029, 1.0, 1.0, .024],
        "Goedel-32B": [.76, .86, "<.001", "<.001"], "Sonnet 4.6": [.25, .31, .28, .28]},
    "table4": {  # best single mode, oracle
        "Qwen3-8B": [36, 52], "Qwen3-32B": [62, 80], "Goedel-8B": [219, 241], "Goedel-32B": [237, 266], "Sonnet 4.6": [366, 388]},
    "fig3_relgain": {"Qwen3-8B": 58, "Qwen3-32B": 29, "Goedel-8B": 18, "Goedel-32B": 12, "Sonnet 4.6": 6},
    "table12_matched": {"Qwen3-8B": 51, "Qwen3-32B": 79, "Goedel-8B": 232, "Goedel-32B": 259, "Sonnet 4.6": 370},
    "table8": {  # PO, KG, ML, full, oracle
        "matholympiadbench": [39, 33, 36, 38, 50], "minif2f": [365, 358, 359, 366, 388], "putnambench": [34, 30, 31, 25, 45],
        "total": [438, 421, 426, 429, 483]},
    "table8_relgain": {"matholympiadbench": 28, "minif2f": 6, "putnambench": 32, "total": 10},
    "recovered_lost": [45, 13],
    "table9": {  # +KG, +Mathlib, +Full beyond PO, lost
        "matholympiadbench": [6, 4, 9, 3], "minif2f": [10, 9, 18, 7], "putnambench": [5, 4, 4, 3]},
    "fig4_regions": {  # KG only, ML only, Full only, KG&ML, KG&Full, ML&Full, all three
        "minif2f": [2, 1, 9, 2, 3, 3, 3], "putnambench": [4, 3, 3, 0, 0, 0, 1], "matholympiadbench": [1, 1, 4, 0, 2, 0, 3]},
    "table10": {
        "minif2f": {1: [198, 201, 210, 206], 3: [299, 284, 290, 290], 5: [326, 311, 311, 316],
                    10: [349, 339, 339, 341], 20: [358, 352, 352, 352], 32: [365, 358, 359, 366]},
        "putnambench": {1: [6, 7, 6, 5], 3: [15, 13, 9, 11], 5: [17, 16, 11, 14], 10: [21, 24, 18, 21],
                        20: [28, 25, 24, 25], 32: [34, 30, 31, 25]},
        "matholympiadbench": {1: [6, 3, 6, 4], 3: [11, 5, 11, 8], 5: [16, 13, 18, 14], 10: [25, 22, 25, 26],
                              20: [33, 28, 33, 34], 32: [39, 33, 36, 38]}},
    "overflow_goedel32b": {"with_kg": 10, "full_system": 28, "both": 1},
}


def load_records(results_dir: Path, run: str, bench: str, mode: str):
    path = results_dir / run / f"{bench}_{mode}" / f"eval_{bench}_{mode}.json"
    if not path.exists():
        sys.exit(f"missing result file: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def solved(records, k=None):
    """Problem ids solved (status verified), optionally within the first k attempts."""
    return {r["problem_id"] for r in records
            if r["status"] == "verified" and (k is None or (r.get("attempts") or 0) <= k)}


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value on the discordant problems of solve sets a and b."""
    n1, n2 = len(a - b), len(b - a)
    n = n1 + n2
    if n == 0:
        return 1.0
    k = min(n1, n2)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fmt_p(p):
    """Format a p-value as the paper does: '<.001', three decimals below .05, two above, '1.0' at 1."""
    if p < 0.001:
        return "<.001"
    if p < 0.05:
        return f"{p:.3f}"[1:]
    if p >= 0.995:
        return "1.0"
    return f"{p:.2f}"[1:]


def fmt_delta(d):
    """Signed solve-count delta, printed as the paper does ('0' for no change)."""
    return f"{d:+d}" if d else "0"


def rnd(x):
    """Round half up (the paper's rounding of relative gains); x is non-negative here."""
    return math.floor(x + 0.5)


def pct(x, n):
    return f"{100 * x / n:.1f}%"


class Checker:
    def __init__(self, enabled):
        self.enabled, self.failures, self.count = enabled, [], 0

    def eq(self, label, got, expected):
        if not self.enabled:
            return
        self.count += 1
        if got != expected:
            self.failures.append(f"{label}: computed {got}, paper {expected}")

    def p(self, label, got, expected):
        if not self.enabled:
            return
        self.count += 1
        ok = got < 0.001 if expected == "<.001" else abs(got - expected) <= 0.0105
        if not ok:
            self.failures.append(f"{label}: computed p={got:.4f}, paper {expected}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="outputs/experiments", help="directory holding the run directories")
    ap.add_argument("--check", action="store_true", help="compare every number with the values printed in the paper")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    chk = Checker(args.check)

    # ---- miniF2F, five models ----
    sets, n_problems = {}, {}
    for m in MODELS:
        recs = {mode: load_records(rd, RUNS[m], "minif2f", mode) for mode in MODES}
        n_problems[m] = len(recs["prover_only"])
        sets[m] = {mode: solved(recs[mode]) for mode in MODES}
        sets[m]["@8"] = set().union(*[solved(recs[mode], 8) for mode in MODES])
    n = n_problems["Sonnet 4.6"]

    print(f"\nTable 2. Cross-model pass@32 solve counts on miniF2F ({n} problems)")
    print(f"{'model':12s} " + " ".join(f"{mode:>14s}" for mode in MODES))
    for m in MODELS:
        counts = [len(sets[m][mode]) for mode in MODES]
        print(f"{m:12s} " + " ".join(f"{c:>6d} ({pct(c, n_problems[m])})" for c in counts))
        chk.eq(f"Table 2 {m}", counts, EXPECTED["table2"][m])

    print("\nTable 3. Solve-count deltas on miniF2F, with exact McNemar p-values (Table 11)")
    print(f"{'model':12s} {'kg-PO':>12s} {'mathlib-PO':>12s} {'full-mathlib':>12s} {'full-max(kg,ml)':>16s}")
    for m in MODELS:
        s = sets[m]
        best_single = max(("with_kg", "with_mathlib"), key=lambda x: len(s[x]))
        pairs = [(s["with_kg"], s["prover_only"]), (s["with_mathlib"], s["prover_only"]),
                 (s["full_system"], s["with_mathlib"]), (s["full_system"], s[best_single])]
        deltas = [len(a) - len(b) for a, b in pairs]
        ps = [mcnemar_exact(a, b) for a, b in pairs]
        print(f"{m:12s} " + " ".join(f"{fmt_delta(d):>5s} ({fmt_p(p):>5s})" for d, p in zip(deltas, ps)))
        chk.eq(f"Table 3 {m}", deltas, EXPECTED["table3"][m])
        for j, (p, e) in enumerate(zip(ps, EXPECTED["table11_p"][m])):
            chk.p(f"Table 11 {m} col {j + 1}", p, e)

    print("\nTable 4. Oracle on miniF2F (union of the four modes); Figure 3 relative gain; Table 12 matched budget")
    print(f"{'model':12s} {'prover_only':>12s} {'best single':>12s} {'oracle':>8s} {'gain':>14s} {'rel. gain':>9s} {'matched@8':>10s}")
    for m in MODELS:
        s = sets[m]
        po = len(s["prover_only"]); best = max(len(s[mode]) for mode in MODES)
        oracle = len(set().union(*[s[mode] for mode in MODES]))
        rel = rnd(100 * (oracle - po) / po)
        matched = len(s["@8"])
        print(f"{m:12s} {po:>12d} {best:>12d} {oracle:>8d} {oracle - po:>+6d} ({100 * (oracle - po) / n_problems[m]:+.1f}%) {rel:>+8d}% {matched:>10d}")
        chk.eq(f"Table 4 {m}", [best, oracle], EXPECTED["table4"][m])
        chk.eq(f"Figure 3 {m}", rel, EXPECTED["fig3_relgain"][m])
        chk.eq(f"Table 12 {m}", matched, EXPECTED["table12_matched"][m])

    # ---- Sonnet 4.6, three benchmarks ----
    print("\nTable 8. Sonnet 4.6 across benchmarks (pass@32), with the oracle; recovered = solved by an augmented mode but not prover_only, lost = solved only by prover_only")
    print(f"{'benchmark':24s} {'prover_only':>12s} {'with_kg':>8s} {'with_mathlib':>12s} {'full_system':>12s} {'oracle':>8s} {'gain':>12s} {'recov':>6s} {'lost':>5s}")
    tot = [0] * 5; tot_n = rec_tot = lost_tot = 0
    sonnet = {}
    for bench in BENCH_ORDER:
        recs = {mode: load_records(rd, SONNET_BENCH[bench], bench, mode) for mode in MODES}
        nb = len(recs["prover_only"])
        s = {mode: solved(recs[mode]) for mode in MODES}
        sonnet[bench] = (recs, s, nb)
        po = s["prover_only"]; aug = s["with_kg"] | s["with_mathlib"] | s["full_system"]
        oracle = po | aug; recovered = aug - po; lost = po - aug
        counts = [len(s[mode]) for mode in MODES] + [len(oracle)]
        rel = rnd(100 * (len(oracle) - len(po)) / len(po))
        print(f"{BENCH_LABEL[bench] + f' ({nb})':24s} " + " ".join(f"{c:>8d}" for c in counts[:4])
              + f" {counts[4]:>8d} {len(oracle) - len(po):>+5d} ({rel:+d}%) {len(recovered):>6d} {len(lost):>5d}")
        chk.eq(f"Table 8 {bench}", counts, EXPECTED["table8"][bench])
        chk.eq(f"Table 8 rel. gain {bench}", rel, EXPECTED["table8_relgain"][bench])
        tot = [a + b for a, b in zip(tot, counts)]; tot_n += nb; rec_tot += len(recovered); lost_tot += len(lost)
    rel = rnd(100 * (tot[4] - tot[0]) / tot[0])
    print(f"{'Total (' + str(tot_n) + ')':24s} " + " ".join(f"{c:>8d}" for c in tot[:4]) + f" {tot[4]:>8d} {tot[4] - tot[0]:>+5d} ({rel:+d}%) {rec_tot:>6d} {lost_tot:>5d}")
    chk.eq("Table 8 total", tot, EXPECTED["table8"]["total"])
    chk.eq("Table 8 total rel. gain", rel, EXPECTED["table8_relgain"]["total"])
    chk.eq("recovered/lost", [rec_tot, lost_tot], EXPECTED["recovered_lost"])

    print("\nTable 9 / Figure 4. Problems solved beyond prover_only (overlapping) and the disjoint Venn regions")
    print(f"{'benchmark':20s} {'+KG':>5s} {'+Mathlib':>9s} {'+Full':>6s} {'lost':>5s}   regions: KG | ML | Full | KG&ML | KG&Full | ML&Full | all")
    for bench in BENCH_ORDER:
        recs, s, nb = sonnet[bench]
        po = s["prover_only"]
        K, M, F = (s[mode] - po for mode in ("with_kg", "with_mathlib", "full_system"))
        lost = len(po - (s["with_kg"] | s["with_mathlib"] | s["full_system"]))
        beyond = [len(K), len(M), len(F), lost]
        regions = [len(K - M - F), len(M - K - F), len(F - K - M), len((K & M) - F), len((K & F) - M), len((M & F) - K), len(K & M & F)]
        print(f"{BENCH_LABEL[bench]:20s} {beyond[0]:>+5d} {beyond[1]:>+9d} {beyond[2]:>+6d} {-lost:>5d}   " + " | ".join(f"{r:>2d}" for r in regions))
        chk.eq(f"Table 9 {bench}", beyond, EXPECTED["table9"][bench])
        chk.eq(f"Figure 4 regions {bench}", regions, EXPECTED["fig4_regions"][bench])

    print("\nTable 10. Sonnet 4.6 pass@k solve counts")
    for bench in ["minif2f", "putnambench", "matholympiadbench"]:
        recs, s, nb = sonnet[bench]
        print(f"  {BENCH_LABEL[bench]} ({nb}):  " + "  ".join(f"{mode}" for mode in MODES))
        for k in PASSK:
            counts = [len(solved(recs[mode], k)) for mode in MODES]
            print(f"    pass@{k:<3d} " + "  ".join(f"{c:>4d} ({pct(c, nb):>5s})" for c in counts))
            chk.eq(f"Table 10 {bench} pass@{k}", counts, EXPECTED["table10"][bench][k])

    print("\nAppendix I. Goedel-Prover-V2-32B records with status 'error' (context overflow)")
    err_sets = {}
    for mode in MODES:
        recs = load_records(rd, RUNS["Goedel-32B"], "minif2f", mode)
        err_sets[mode] = {r["problem_id"] for r in recs if r["status"] == "error"}
        print(f"  {mode:14s} {len(err_sets[mode]):>3d}")
        if mode in EXPECTED["overflow_goedel32b"]:
            chk.eq(f"Appendix I {mode}", len(err_sets[mode]), EXPECTED["overflow_goedel32b"][mode])
    both = len(err_sets["with_kg"] & err_sets["full_system"])
    print(f"  {'in both':14s} {both:>3d}")
    chk.eq("Appendix I both", both, EXPECTED["overflow_goedel32b"]["both"])

    if args.check:
        print(f"\nCheck against the paper: {chk.count - len(chk.failures)} of {chk.count} values match.")
        for f in chk.failures:
            print("  MISMATCH", f)
        sys.exit(1 if chk.failures else 0)


if __name__ == "__main__":
    main()
