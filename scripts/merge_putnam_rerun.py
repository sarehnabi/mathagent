#!/usr/bin/env python3
"""
Merge the PutnamBench rerun (346 answer-slot problems, answers inlined, statement check on) with the
May 2026 results for the 326 problems that have no answer slot, producing one result file per mode
over all 672 problems.

For every problem exactly one source applies:
  * answer-slot problem (id in outputs/validation/putnam_answer_slot_ids.txt): the rerun record;
  * slot-free problem: the May record, with status 'verified' kept only if the proof is in the
    corrected solve set (outputs/validation/valid_solves.json); otherwise 'statement_check_failed'.

Outputs (default): outputs/experiments/putnam_merged_09262026_claude_sonnet/putnambench_<mode>/
  eval_putnambench_<mode>.json   merged records (672), summary recomputed
  run_config.json                provenance: both source run_configs and the per-source counts
and prints the per-mode counts, oracle, recovered/lost, and pass@k.

Usage: python3 scripts/merge_putnam_rerun.py [--out DIR]
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAY = ROOT / "outputs/experiments/full_ablation_05162026_2359_claude_sonnet"
RERUN = ROOT / "outputs/experiments/putnam_rerun_09262026_claude_sonnet"
SLOT_IDS = ROOT / "outputs/validation/putnam_answer_slot_ids.txt"
VALID = ROOT / "outputs/validation/valid_solves.json"
MODES = ["prover_only", "with_kg", "with_mathlib", "full_system"]
KS = (1, 3, 5, 10, 20, 32)


def load(run_dir: Path, mode: str):
    f = glob.glob(str(run_dir / f"putnambench_{mode}" / f"eval_putnambench_{mode}.json"))[0]
    return json.load(open(f)), json.load(open(run_dir / f"putnambench_{mode}" / "run_config.json"))


def summary(results, total):
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    per_domain = {}
    for r in results:
        d = per_domain.setdefault(r.get("domain", "unknown"), {"total": 0, "solved": 0})
        d["total"] += 1
        d["solved"] += r["status"] == "verified"
    return {"solved": counts.get("verified", 0), "failed": counts.get("failed", 0),
            "statement_check_failed": counts.get("statement_check_failed", 0),
            "timeout": counts.get("timeout", 0), "unavailable": counts.get("unavailable", 0),
            "solve_rate": f"{100*counts.get('verified', 0)/total:.1f}%", "per_domain": per_domain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/experiments/putnam_merged_09262026_claude_sonnet"))
    args = ap.parse_args()
    out = Path(args.out)
    slot = set(SLOT_IDS.read_text().split())
    valid = json.load(open(VALID))
    solved, attempts = {}, {}
    for mode in MODES:
        may, may_cfg = load(MAY, mode)
        rer, rer_cfg = load(RERUN, mode)
        may_by = {r["problem_id"]: r for r in may["results"]}
        rer_by = {r["problem_id"]: r for r in rer["results"]}
        assert set(rer_by) == slot, f"{mode}: rerun ids differ from the slot list"
        assert slot <= set(may_by), f"{mode}: slot ids missing from the May run"
        may_valid = set(valid[f"Sonnet 4.6|putnambench|{mode}"])
        merged, n_relabeled = [], 0
        for pid in sorted(may_by):                      # May run defines the 672-problem order
            if pid in slot:
                r = dict(rer_by[pid]); r["source"] = "rerun_2026-09 (answers inlined, statement check)"
            else:
                r = dict(may_by[pid]); r["source"] = "may_2026 (no answer slot)"
                if r["status"] == "verified" and pid not in may_valid:
                    r["status"] = "statement_check_failed"; n_relabeled += 1
            merged.append(r)
        assert len(merged) == 672
        d = {k: v for k, v in may.items() if k not in ("results", "summary")}
        d.update({"eval_mode": mode, "total_problems": 672, "results": merged, "summary": summary(merged, 672)})
        mode_dir = out / f"putnambench_{mode}"; mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / f"eval_putnambench_{mode}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
        rer_solved = sum(1 for r in merged if r["source"].startswith("rerun") and r["status"] == "verified")
        may_solved = sum(1 for r in merged if r["source"].startswith("may") and r["status"] == "verified")
        cfg = {"merged_from": {"answer_slot_problems_346": {"run": RERUN.name, "solved": rer_solved, "run_config": rer_cfg},
                               "slot_free_problems_326": {"run": MAY.name, "solved": may_solved,
                                                          "relabeled_statement_check_failed": n_relabeled, "run_config": may_cfg}},
               "benchmark": "putnambench", "eval_mode": mode, "num_problems": 672, "max_attempts": 32,
               "model": rer_cfg.get("model"), "statement_check": True, "putnambench_answers": "inlined"}
        (mode_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
        solved[mode] = {r["problem_id"] for r in merged if r["status"] == "verified"}
        attempts[mode] = {r["problem_id"]: r["attempts"] for r in merged if r["status"] == "verified"}
        print(f"{mode:13s} {len(solved[mode]):3d}/672 = {100*len(solved[mode])/672:.1f}%  (rerun {rer_solved} + May {may_solved}; relabeled {n_relabeled})")
    po = solved["prover_only"]; aug = solved["with_kg"] | solved["with_mathlib"] | solved["full_system"]
    orc = po | aug
    print(f"oracle {len(orc)} ({100*len(orc)/672:.1f}%)  gain {len(orc)-len(po):+d} ({100*(len(orc)-len(po))/len(po):+.0f}%)  recovered {len(aug-po)}  lost {len(po-aug)}")
    print("recovered per mode (mode - PO):", {m: len(solved[m]-po) for m in MODES[1:]})
    K, M, F = (solved[m] - po for m in MODES[1:])
    print("venn regions [K, M, K&M, F, K&F, M&F, all]:", [len(K-M-F), len(M-K-F), len((K&M)-F), len(F-K-M), len((K&F)-M), len((M&F)-K), len(K&M&F)])
    print("pass@k", KS)
    for m in MODES:
        print(f"  {m:13s}", [sum(1 for a in attempts[m].values() if a <= k) for k in KS])
    print(f"written to {out}")


if __name__ == "__main__":
    main()
