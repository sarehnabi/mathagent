#!/usr/bin/env python3
"""
Targeted problem-solving on benchmarks (Algorithm 1 in the paper).

For each problem of miniF2F, PutnamBench, or MathOlympiadBench, the Explorer gathers context
according to the augmentation mode and the Prover runs up to --max-attempts sequential,
error-guided attempts (the paper's runs used 32, i.e. pass@32):

    prover_only    no external context
    with_kg        Layer 1: MathKG semantic search (seeds + 1-hop neighbours)
    with_mathlib   Layer 2: Mathlib declaration search (type signatures + docstrings)
    full_system    both layers, KG block first

A problem counts as solved only if the Lean file compiles cleanly and passes the statement check
(src/statement_check.py). PutnamBench answer slots are filled with the benchmark's answers.
Each run writes eval_<benchmark>_<mode>.json, an incremental .jsonl, a .log and run_config.json
into --output-dir (default: a timestamped outputs/experiments/ directory).

Usage:
    # One mode, parallel (default), with the released graph
    python -m src.benchmarks.evaluate --benchmark minif2f --eval-mode with_kg \
        --kg-path outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json \
        --max-attempts 32 --workers 20

    # Quick sequential smoke test on the first 5 problems
    python -m src.benchmarks.evaluate --benchmark matholympiadbench --first 5 --sequential

    # A subset of problems by id (one id per line); all four modes: scripts/run_full_ablation.sh
    python -m src.benchmarks.evaluate --benchmark putnambench --eval-mode prover_only --problem-ids ids.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.prover import ProverAgent, INITIAL_TEMPERATURE, REFINE_TEMPERATURE
from agents.explorer import ExplorerAgent
from build_graph import MathKG
from lean4_verifier import Lean4Verifier


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = PROJECT_DIR / "data" / "benchmarks"


class TeeLogger:
    """Duplicates stdout to both the console and a log file."""

    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.log_file.close()


def generate_experiment_dir(benchmark: str, eval_mode: str, num_problems: int,
                            max_attempts: int, workers: int) -> Path:
    """
    Generate a unique experiment directory with descriptive naming.

    Format: outputs/experiments/exp_MMDDYYYY_HHMM_{benchmark}_{mode}_{N}prob_{K}att_{W}workers/

    Example: outputs/experiments/exp_04222026_1430_matholympiadbench_full_system_360prob_32att_48workers/
    """
    timestamp = datetime.now().strftime("%m%d%Y_%H%M")
    dir_name = f"exp_{timestamp}_{benchmark}_{eval_mode}_{num_problems}prob_{max_attempts}att_{workers}workers"
    exp_dir = PROJECT_DIR / "outputs" / "experiments" / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ═══════════════════════════════════════════════════════════════════
# Benchmark Loaders
# ═══════════════════════════════════════════════════════════════════

def load_matholympiadbench(domain_filter: str = None) -> List[Dict]:
    """Load MathOlympiadBench problems."""
    path = BENCHMARKS_DIR / "matholympiadbench.json"
    with open(path) as f:
        data = json.load(f)

    problems = []
    for entry in data:
        problem = {
            "id": entry.get("problem_id") or entry.get("name", ""),
            "source": "MathOlympiadBench",
            "domain": entry.get("category", "unknown"),
            "formal_statement": entry.get("formal_statement", ""),
            "header": entry.get("header", ""),
            "informal_stmt": entry.get("informal_prefix", ""),
            "lean4_code": entry.get("lean4_code", ""),
        }
        if domain_filter and domain_filter.lower() not in problem["domain"].lower():
            continue
        problems.append(problem)
    return problems


def load_minif2f(domain_filter: str = None, split: str = None) -> List[Dict]:
    """Load miniF2F problems. Default loads all 488 (test + validation)."""
    path = BENCHMARKS_DIR / "minif2f.json"
    with open(path) as f:
        data = json.load(f)

    problems = []
    for entry in data:
        if split and entry.get("split") != split:
            continue

        pid = entry.get("problem_id", "")
        if "numbertheory" in pid.lower():
            domain = "number theory"
        elif "algebra" in pid.lower():
            domain = "algebra"
        elif "amc" in pid.lower() or "aime" in pid.lower():
            domain = "competition"
        elif "imo" in pid.lower():
            domain = "imo"
        else:
            domain = "other"

        problem = {
            "id": pid,
            "source": "miniF2F",
            "domain": domain,
            "formal_statement": entry.get("formal_statement", ""),
            "header": entry.get("header", ""),
            "informal_stmt": entry.get("informal_stmt", ""),
            "lean4_code": "",
        }
        if domain_filter and domain_filter.lower() not in problem["domain"].lower():
            continue
        problems.append(problem)
    return problems


def _inline_answers(code: str) -> str:
    """Write PutnamBench ground-truth answers into their answer slots.

    PutnamBench stores each answer as a comment on the line after
    'abbrev/def <name>_solution : T := sorry'.  This replaces the 'sorry' with that
    answer and drops the comment, producing the benchmark's standard Task-2 form
    ("given the statement and solution, produce a proof").  Leaving the slot for the
    model to fill (Task 1) lets a compiling file restate the question in the slot and
    close the theorem trivially, so answers are inlined and the statement check
    (src/statement_check.py) requires them to be reproduced.
    """
    lines = code.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r'^(\s*(?:noncomputable\s+)?(?:abbrev|def)\s+\w+.*:=)\s*sorry\s*$', line)
        if m and i + 1 < len(lines) and lines[i + 1].strip().startswith('--'):
            answer = lines[i + 1].strip()[2:].strip()
            out.append(f"{m.group(1)} {answer}")
            i += 2
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def load_putnambench(domain_filter: str = None) -> List[Dict]:
    """Load PutnamBench Lean 4 problems with domain tags from metadata.

    Passes the full .lean file as formal_statement with the answer slots filled
    from the benchmark's answer-key comments (Task 2: prove the statement with
    the known answer).
    """
    putnam_dir = BENCHMARKS_DIR / "putnambench" / "lean4" / "src"
    metadata_path = BENCHMARKS_DIR / "putnambench" / "informal" / "putnam.json"
    if not putnam_dir.exists():
        print(f"PutnamBench not found at {putnam_dir}")
        return []

    # Load domain tags from metadata
    tags_map = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        for entry in metadata:
            name = entry.get("problem_name", "")
            tags = entry.get("tags", [])
            domain = tags[0].replace("_", " ") if tags else "unknown"
            tags_map[name] = {
                "domain": domain,
                "informal_stmt": entry.get("informal_statement", ""),
            }

    problems = []
    for lean_file in sorted(putnam_dir.glob("*.lean")):
        code = lean_file.read_text()
        stem = lean_file.stem

        # Get metadata
        meta = tags_map.get(stem, {})
        domain = meta.get("domain", "unknown")

        # Extract docstring as informal statement (fallback to metadata)
        doc_match = re.search(r'/\-\-(.*?)\-/', code, re.DOTALL)
        informal = doc_match.group(1).strip() if doc_match else meta.get("informal_stmt", "")

        # Pass the full file as formal_statement with the ground-truth answers written
        # into their slots (PutnamBench Task 2); the Prover must reproduce them unchanged.
        formal = _inline_answers(code)

        problem = {
            "id": stem,
            "source": "PutnamBench",
            "domain": domain,
            "formal_statement": formal,
            "header": "",
            "informal_stmt": informal,
            "lean4_code": code,
        }
        if domain_filter and domain_filter.lower() not in problem["domain"].lower():
            continue
        problems.append(problem)
    return problems


BENCHMARK_LOADERS = {
    "matholympiadbench": load_matholympiadbench,
    "minif2f": load_minif2f,
    "putnambench": load_putnambench,
}


# ═══════════════════════════════════════════════════════════════════
# Context Builders (4 augmentation modes)
# ═══════════════════════════════════════════════════════════════════

def build_context_prover_only(problem: Dict, **kwargs) -> Dict:
    """Baseline: no external context, just the problem statement."""
    return {
        "statement": problem.get("informal_stmt", ""),
        "lean4_sketch": problem["formal_statement"],
        "lean4_proof_sketch": "",
        "supporting_theorem_ids": [],
        "domain": problem.get("domain", ""),
        "header": problem.get("header", ""),
    }


def _format_kg_results(relevant: List[Dict], types_db: dict = None) -> tuple:
    """Format KG search results into context string and supporting IDs.

    When types_db is provided, includes full Lean type signatures for each
    KG theorem so the Prover knows exactly what arguments each lemma takes.
    """
    context_lines = []
    supporting_ids = []

    for t in relevant:
        tid = t.get("theorem_id", "")
        name = t.get("name", tid)
        statement = t.get("statement", "")
        rationale = t.get("rationale", "")
        mathlib_name = t.get("lean4_mathlib_name", "")
        mathlib_import = t.get("lean4_mathlib_import", "")
        is_seed = t.get("is_seed", False)

        prefix = "SEED" if is_seed else "CONNECTED"
        context_lines.append(f"-- [{prefix}] {name}: {statement}")
        if rationale:
            context_lines.append(f"--   Why relevant: {rationale}")

        # Show full type signature if available 
        if mathlib_name and types_db:
            entry = types_db.get(mathlib_name, {})
            type_sig = entry.get("type", "")
            if type_sig:
                context_lines.append(f"--   Lean 4: {type_sig}")
            else:
                context_lines.append(f"--   Mathlib name: {mathlib_name}")
        elif mathlib_name:
            context_lines.append(f"--   Mathlib name: {mathlib_name}")

        # Include edge reasoning for connected (non-seed) nodes
        for conn in t.get("edge_connections", []):
            context_lines.append(
                f"--   Edge: {conn['type']} from {conn['from']} "
                f"(confidence {conn['confidence']:.2f}): {conn['reasoning'][:100]}"
            )

        context_lines.append("")  # blank line between theorems
        if tid:
            supporting_ids.append(tid)

    return "\n".join(context_lines), supporting_ids


def build_context_with_kg(problem: Dict, explorer: ExplorerAgent = None,
                           kg: MathKG = None, **kwargs) -> Dict:
    """Layer 1: Explorer searches KG using 4-step pipeline."""
    if not explorer or not kg:
        return build_context_prover_only(problem)

    relevant = explorer.find_relevant_theorems_for_problem(problem, kg)
    types_db = explorer._get_mathlib_types()
    context_text, supporting_ids = _format_kg_results(relevant, types_db=types_db)

    return {
        "statement": problem.get("informal_stmt", ""),
        "lean4_sketch": problem["formal_statement"],
        "lean4_proof_sketch": context_text,
        "supporting_theorem_ids": supporting_ids,
        "domain": problem.get("domain", ""),
        "header": problem.get("header", ""),
    }


def _format_mathlib_results(mathlib_results: List[Dict]) -> str:
    """Format Layer 2 Mathlib search results into context string.

    Includes type signatures and docstrings when available, so the Prover
    knows exactly what arguments each lemma takes and what it proves.
    """
    if not mathlib_results:
        return ""

    context_lines = ["MATHLIB LEMMAS (from library search):"]
    for m in mathlib_results:
        type_sig = m.get('type_signature', '')
        doc = m.get('doc', '')

        if type_sig:
            context_lines.append(f"  {type_sig}")
            if doc:
                context_lines.append(f"    -- {doc[:200]}")
        else:
            priority_tag = f"[{m.get('priority', '?')}]" if m.get('priority') else ""
            context_lines.append(
                f"  -- {m['name']} [{m['kind']}] {priority_tag} "
                f"(module: {m.get('module', '?')}, score: {m.get('score', 0):.2f})"
            )
    return "\n".join(context_lines)


def build_context_with_mathlib(problem: Dict, explorer: ExplorerAgent = None,
                                **kwargs) -> Dict:
    """Layer 2 only: Explorer searches Mathlib declarations (no KG)."""
    if not explorer:
        return build_context_prover_only(problem)

    mathlib_results = explorer.search_mathlib_declarations(problem, kg_results=None)
    context_text = _format_mathlib_results(mathlib_results)

    return {
        "statement": problem.get("informal_stmt", ""),
        "lean4_sketch": problem["formal_statement"],
        "lean4_proof_sketch": context_text,
        "supporting_theorem_ids": [],
        "domain": problem.get("domain", ""),
        "header": problem.get("header", ""),
    }


def build_context_full_system(problem: Dict, explorer: ExplorerAgent = None,
                               kg: MathKG = None, **kwargs) -> Dict:
    """Layer 1 + Layer 2: KG search (4-step pipeline) + KG-aware Mathlib search."""
    if not explorer or not kg:
        return build_context_prover_only(problem)

    # Layer 1: KG search with full pipeline
    kg_results = explorer.find_relevant_theorems_for_problem(problem, kg)
    types_db = explorer._get_mathlib_types()
    kg_context, supporting_ids = _format_kg_results(kg_results, types_db=types_db)

    # Layer 2: Mathlib search — KG-aware (boosts KG seed modules, deduplicates)
    mathlib_results = explorer.search_mathlib_declarations(problem, kg_results=kg_results)
    mathlib_context = _format_mathlib_results(mathlib_results)

    # Combine both layers
    context_parts = ["RELEVANT THEOREMS FROM KNOWLEDGE GRAPH:", kg_context]
    if mathlib_context:
        context_parts.append("")
        context_parts.append(mathlib_context)

    return {
        "statement": problem.get("informal_stmt", ""),
        "lean4_sketch": problem["formal_statement"],
        "lean4_proof_sketch": "\n".join(context_parts),
        "supporting_theorem_ids": supporting_ids,
        "domain": problem.get("domain", ""),
        "header": problem.get("header", ""),
    }


CONTEXT_BUILDERS = {
    "prover_only": build_context_prover_only,
    "with_kg": build_context_with_kg,
    "with_mathlib": build_context_with_mathlib,
    "full_system": build_context_full_system,
}


# ═══════════════════════════════════════════════════════════════════
# Single Problem Evaluation (used by both sequential and parallel)
# ═══════════════════════════════════════════════════════════════════

def evaluate_single_problem(problem: Dict, eval_mode: str, max_attempts: int,
                            lean_temp_dir: str, explorer: ExplorerAgent = None,
                            kg: MathKG = None) -> Dict:
    """
    Evaluate a single benchmark problem. Thread-safe.

    Each call creates its own ProverAgent and Lean4Verifier with a unique
    temp directory to avoid file conflicts during parallel execution.
    """
    # Create per-worker Prover with unique temp dir
    prover = ProverAgent()
    prover.verifier = Lean4Verifier()
    prover.verifier.temp_dir = Path(lean_temp_dir)
    prover.verifier.temp_dir.mkdir(parents=True, exist_ok=True)

    # Build context based on eval mode
    builder = CONTEXT_BUILDERS[eval_mode]
    conjecture = builder(problem=problem, explorer=explorer, kg=kg)

    start_time = time.time()
    proof_result = prover.attempt_proof(
        conjecture=conjecture,
        kg=kg,
        max_attempts=max_attempts,
    )
    elapsed = time.time() - start_time

    return {
        "problem_id": problem["id"],
        "domain": problem["domain"],
        "status": proof_result["status"],
        "attempts": proof_result["attempts"],
        "time_seconds": round(elapsed, 1),
        "errors": proof_result.get("final_errors", [])[:3],
        "lean4_code": proof_result.get("lean4_code", ""),
        "rejected_empty_verifications": proof_result.get("rejected_empty_verifications", 0),
    }


# ═══════════════════════════════════════════════════════════════════
# Main Evaluation Loop (Sequential + Parallel)
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(benchmark: str, problems: List[Dict],
                   eval_mode: str = "prover_only", kg_path: str = None,
                   max_attempts: int = 3, output_dir: str = None,
                   parallel: bool = None, max_workers: int = None) -> Dict:
    """
    Run evaluation on benchmark problems.

    Args:
        eval_mode: "prover_only" | "with_kg" | "with_mathlib" | "full_system"
        parallel: If True, use ThreadPoolExecutor. Defaults to config.EVAL_PARALLEL.
        max_workers: Max concurrent evaluations. Defaults to config.EVAL_MAX_WORKERS.
        output_dir: If None, auto-generates experiment directory with descriptive naming.
    """
    if parallel is None:
        parallel = getattr(config, 'EVAL_PARALLEL', False)
    if max_workers is None:
        max_workers = getattr(config, 'EVAL_MAX_WORKERS', 4)

    effective_workers = max_workers if (parallel and max_workers > 1) else 1

    # Auto-generate experiment directory if not specified
    if output_dir is None:
        exp_dir = generate_experiment_dir(
            benchmark=benchmark,
            eval_mode=eval_mode,
            num_problems=len(problems),
            max_attempts=max_attempts,
            workers=effective_workers,
        )
        output_dir = str(exp_dir)

    # Set up logging — every print goes to both console and log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / f"eval_{benchmark}_{eval_mode}.log"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    # Suppress per-attempt logging during batch evaluation — only show per-problem summary
    from agents.base_agent import BaseAgent
    BaseAgent.verbose = False

    # Load KG and/or Explorer as needed
    kg = None
    explorer = None
    if eval_mode in ("with_kg", "full_system") and kg_path:
        kg = MathKG()
        kg.load(Path(kg_path))
        explorer = ExplorerAgent("BenchmarkExplorer")
        print(f"Loaded KG: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
    elif eval_mode == "with_mathlib":
        explorer = ExplorerAgent("BenchmarkExplorer")
        print("Loaded Explorer for Mathlib search (no KG)")

    mode_str = f"PARALLEL ({max_workers} workers)" if parallel and max_workers > 1 else "SEQUENTIAL"

    print(f"\n{'='*60}")
    print(f"BENCHMARK EVALUATION — {mode_str}")
    print(f"{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"Problems: {len(problems)}")
    print(f"Mode: {eval_mode}")
    print(f"Max attempts: {max_attempts}")
    print(f"{'='*60}\n")

    results = {
        "benchmark": benchmark,
        "total_problems": len(problems),
        "eval_mode": eval_mode,
        "max_attempts": max_attempts,
        "model": os.environ.get("VLLM_MODEL") or config.MODEL_ID,   # the model that actually served the run
        "parallel": parallel,
        "max_workers": max_workers if parallel else 1,
        "results": [],
        "summary": {},
    }

    # Thread-safe counters
    lock = threading.Lock()
    counters = {"solved": 0, "failed": 0, "timeout": 0, "unavailable": 0, "completed": 0}
    all_results = []

    # Incremental results file (written after each problem)
    os.makedirs(output_dir, exist_ok=True)
    incremental_path = Path(output_dir) / f"eval_{benchmark}_{eval_mode}_incremental.jsonl"

    # --- Resume support ---
    # If an incremental file already exists (e.g. from a run interrupted by a
    # Spot reclaim / stop), skip problems that already have a final result and
    # re-run only transient failures. Keeps `problems` full for totals/summary;
    # only the set of problems actually processed is filtered.
    FINAL_STATUSES = {"verified", "failed", "timeout"}
    done_ids = set()
    if incremental_path.exists():
        prior_by_id = {}  # problem_id -> most recent result (last line wins)
        with open(incremental_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip a partial line from an interrupted write
                pid = r.get("problem_id")
                if pid is not None:
                    prior_by_id[pid] = r
        prior_done = [r for r in prior_by_id.values()
                      if r.get("status") in FINAL_STATUSES]
        done_ids = {r["problem_id"] for r in prior_done}
        if done_ids:
            # Pre-load prior results so the final summary reflects the full set
            all_results.extend(prior_done)
            for r in prior_done:
                st = r["status"]
                if st == "verified":
                    counters["solved"] += 1
                elif st == "timeout":
                    counters["timeout"] += 1
                else:
                    counters["failed"] += 1
                counters["completed"] += 1
            print(f"↻ Resuming: {len(done_ids)} problems already completed, "
                  f"{len(problems) - len(done_ids)} remaining (of {len(problems)})")

    def process_problem(idx_problem):
        idx, problem = idx_problem
        # Each worker gets a unique temp directory
        worker_id = threading.current_thread().name
        lean_tmp = f"/tmp/mathagent_lean_{os.getpid()}_{worker_id}"

        result = evaluate_single_problem(
            problem=problem,
            eval_mode=eval_mode,
            max_attempts=max_attempts,
            lean_temp_dir=lean_tmp,
            explorer=explorer,
            kg=kg,
        )

        with lock:
            counters["completed"] += 1
            status = result["status"]
            if status == "verified":
                counters["solved"] += 1
                marker = "✓ SOLVED"
            elif status == "timeout":
                counters["timeout"] += 1
                marker = "⧖ TIMEOUT"
            elif status == "unavailable":
                counters["unavailable"] += 1
                marker = "⚠ UNAVAILABLE"
            else:
                counters["failed"] += 1
                marker = "✗ FAILED"

            progress = counters["completed"]
            total = len(problems)
            print(f"  [{progress}/{total}] {result['problem_id']} ({result['domain']})"
                  f" — {marker} (att {result['attempts']}, {result['time_seconds']}s)")

            # Write result incrementally (one JSON object per line)
            with open(incremental_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        return result

    # Run evaluation
    start_time = time.time()
    # Skip already-completed problems on resume (done_ids empty on a fresh run)
    indexed_problems = [(i, p) for i, p in enumerate(problems)
                        if p["id"] not in done_ids]

    if parallel and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_problem, ip): ip for ip in indexed_problems}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    idx, problem = futures[future]
                    print(f"  ⚠ ERROR on {problem['id']}: {e}")
                    all_results.append({
                        "problem_id": problem["id"],
                        "domain": problem["domain"],
                        "status": "error",
                        "attempts": 0,
                        "time_seconds": 0,
                        "errors": [str(e)],
                    })
    else:
        for ip in indexed_problems:
            result = process_problem(ip)
            all_results.append(result)

    total_time = time.time() - start_time

    # Sort results by problem order (parallel may return out of order)
    problem_order = {p["id"]: i for i, p in enumerate(problems)}
    all_results.sort(key=lambda r: problem_order.get(r["problem_id"], 999))

    results["results"] = all_results
    results["total_time_seconds"] = round(total_time, 1)

    # Summary
    results["summary"] = {
        "solved": counters["solved"],
        "failed": counters["failed"],
        "timeout": counters["timeout"],
        "unavailable": counters["unavailable"],
        "solve_rate": f"{counters['solved']/len(problems)*100:.1f}%" if problems else "0%",
        "total_time": f"{total_time:.0f}s",
    }

    # Per-domain breakdown
    domain_stats = {}
    for r in all_results:
        d = r["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "solved": 0}
        domain_stats[d]["total"] += 1
        if r["status"] == "verified":
            domain_stats[d]["solved"] += 1
    results["summary"]["per_domain"] = domain_stats

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f"eval_{benchmark}_{eval_mode}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Capture the Mathlib version actually used for verification (best-effort).
    def _mathlib_version():
        proj = os.environ.get("LEAN_MATHLIB_PROJECT") or getattr(
            config, "LEAN_MATHLIB_PROJECT", None
        )
        info = {"project": proj}
        if not proj or not os.path.isdir(proj):
            return info
        try:
            info["git_describe"] = subprocess.run(
                ["git", "describe", "--tags", "--always", "--dirty"],
                cwd=proj, capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
            info["git_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=proj, capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
        except Exception:
            pass
        try:
            with open(os.path.join(proj, "lean-toolchain")) as tc:
                info["lean_toolchain"] = tc.read().strip()
        except Exception:
            pass
        return info

    # Save run config for reproducibility
    run_config = {
        "benchmark": benchmark,
        "eval_mode": eval_mode,
        "num_problems": len(problems),
        "max_attempts": max_attempts,
        "parallel": parallel,
        "workers": effective_workers,
        "model": os.environ.get("VLLM_MODEL") or config.MODEL_ID,
        "serving_model": os.environ.get("VLLM_MODEL") or None,
        "use_local_vllm": os.environ.get("USE_LOCAL_VLLM"),
        "vllm_max_tokens_override": os.environ.get("VLLM_MAX_TOKENS"),
        # Sampling temperatures used by the prover (env-overridable via VLLM_TEMPERATURE / VLLM_REFINE_TEMPERATURE)
        "temperature": INITIAL_TEMPERATURE,
        "refine_temperature": REFINE_TEMPERATURE,
        "vllm_request_timeout_seconds": float(os.environ.get("VLLM_REQUEST_TIMEOUT") or 1800),
        "vllm_thinking": os.environ.get("VLLM_THINKING"),
        "lean_timeout_seconds": int(
            os.environ.get("LEAN_TIMEOUT_SECONDS", getattr(config, "LEAN_TIMEOUT_SECONDS", 1200))
        ),
        "mathlib": _mathlib_version(),
        "kg_path": kg_path,
        # A solve requires the assigned theorem and supplied definitions to be reproduced
        # exactly (src/statement_check.py); PutnamBench answer slots are filled with the
        # benchmark's ground-truth answers (Task 2) by the loader.
        "statement_check": True,
        "putnambench_answers": "inlined",
        "problem_ids_file": os.environ.get("EVAL_PROBLEM_IDS_FILE"),
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": results.get("total_time_seconds", 0),
    }
    config_path = Path(output_dir) / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {benchmark} ({eval_mode})")
    print(f"{'='*60}")
    print(f"Total: {len(problems)} problems")
    print(f"Solved: {counters['solved']} ({counters['solved']/len(problems)*100:.1f}%)")
    print(f"Failed: {counters['failed']}")
    print(f"Timeout: {counters['timeout']}")
    print(f"Unavailable: {counters['unavailable']}")
    print(f"Wall time: {total_time:.0f}s ({mode_str})")
    print(f"\nPer-domain:")
    for d, stats in sorted(domain_stats.items()):
        rate = stats['solved']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"  {d:20s}: {stats['solved']}/{stats['total']} ({rate:.0f}%)")
    print(f"\nResults saved to: {output_path}")
    print(f"Log file: {log_path}")
    print(f"{'='*60}")

    tee.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("--benchmark", required=True,
                        choices=["minif2f", "putnambench", "matholympiadbench"],
                        help="Which benchmark to evaluate")
    parser.add_argument("--domain", type=str, default=None,
                        help="Filter by domain (e.g., 'number theory')")
    parser.add_argument("--first", type=int, default=None,
                        help="Only evaluate first N problems")
    parser.add_argument("--last", type=int, default=None,
                        help="Only evaluate last N problems")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Max proof attempts per problem (default 3; the paper's runs used 32)")
    parser.add_argument("--eval-mode", type=str, default="prover_only",
                        choices=["prover_only", "with_kg", "with_mathlib", "full_system"],
                        help="Evaluation mode")
    parser.add_argument("--kg-path", type=str,
                        default="outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json",
                        help="Path to the MathKG JSON file (default: the released 364-node graph)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--sequential", action="store_true",
                        help="Force sequential evaluation (overrides config)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: config.EVAL_MAX_WORKERS)")
    parser.add_argument("--problem-ids", type=str, default=None,
                        help="Path to a text file with one problem id per line; evaluate only these problems")
    args = parser.parse_args()

    # Load problems
    loader = BENCHMARK_LOADERS[args.benchmark]
    problems = loader(domain_filter=args.domain)
    print(f"Loaded {len(problems)} problems from {args.benchmark}")

    if args.problem_ids:
        os.environ["EVAL_PROBLEM_IDS_FILE"] = args.problem_ids
        wanted = {l.strip() for l in open(args.problem_ids) if l.strip()}
        problems = [p for p in problems if p["id"] in wanted]
        print(f"Restricted to {len(problems)} problems listed in {args.problem_ids}")

    if args.first:
        problems = problems[:args.first]
        print(f"Using first {len(problems)} problems")

    if args.last:
        problems = problems[-args.last:]
        print(f"Using last {len(problems)} problems")

    if not problems:
        print("No problems to evaluate!")
        return

    # Determine parallelism
    parallel = not args.sequential
    max_workers = args.workers

    # Run evaluation
    kg_path = args.kg_path if args.eval_mode in ("with_kg", "full_system") else None
    results = run_evaluation(
        benchmark=args.benchmark,
        problems=problems,
        eval_mode=args.eval_mode,
        kg_path=kg_path,
        max_attempts=args.max_attempts,
        output_dir=args.output_dir,
        parallel=parallel,
        max_workers=max_workers,
    )

    # The results path is printed by run_evaluation ("Results saved to: ...").


if __name__ == "__main__":
    main()
