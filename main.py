#!/usr/bin/env python3
"""
MathAgent - Main Entry Point

Modes:
    pipeline    Build MathKG from a theorem/definition node file (default)
    agent       Run the open-ended 4-agent discovery loop on the graph (implemented, not evaluated in the paper)
    paper-demo  Run a 3-iteration agent-loop demo and write the results as JSON

Usage:
    python main.py --mode pipeline --data data/mathlib_fundamentals_verified.json [--workers N] [--skip-inference]
    python main.py --mode agent --iterations N

Targeted problem-solving on benchmarks is run separately: python -m src.benchmarks.evaluate
"""

import sys
from typing import Optional
import argparse
from datetime import datetime
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from extract_mathlib import load_sample_theorems
from infer_edges import SemanticEdgeInferencer
from build_graph import MathKG
from query_graph import run_demo_queries
from visualize import visualize_graph


def _build_experiment_dir(data_file: Optional[str], num_nodes: int) -> Path:
    """Build a timestamped directory for this KG run so outputs are never overwritten.

    Example: outputs/experiments/exp_04282026_1430_mathlib_fundamentals_verified_364n/
    """
    stem = Path(data_file).stem if data_file else "default"
    timestamp = datetime.now().strftime("%m%d%Y_%H%M")
    exp_dir = Path("outputs/experiments") / f"exp_{timestamp}_{stem}_{num_nodes}n"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ──────────────────────────────────────────────
# Pipeline mode: build + query + visualize graph
# ──────────────────────────────────────────────

def run_pipeline(skip_inference=False, data_file=None, output_kg=None, edges_cache=None, viz_path=None):
    """Build MathKG from input theorems.

    By default, every run writes KG, edges cache, and visualization into a
    fresh timestamped directory under outputs/experiments/. This prevents an
    ad-hoc run from overwriting a carefully-built KG (e.g. the full 364-node
    production graph). Any of --output-kg, --edges-cache, or --viz-path
    override the respective output location.
    """

    print("=" * 70)
    print("MATHAGENT — BUILDING MATHEMATICAL KNOWLEDGE GRAPH")
    print("=" * 70)
    print(f"Input: {data_file or '(default NT+combinatorics dataset)'}\n")

    # Step 1: Load theorems
    print("STEP 1: Loading theorems")
    print("-" * 70)
    theorems = load_sample_theorems(filepath=data_file)

    lean4_count = sum(1 for t in theorems if t.get("lean4_formal_statement"))
    print(f"  Lean 4 verified: {lean4_count}/{len(theorems)} theorems\n")

    # Per-run experiment directory (always created; individual outputs may be redirected)
    exp_dir = _build_experiment_dir(data_file, len(theorems))
    print(f"Experiment directory: {exp_dir}\n")

    # Step 2: Infer semantic edges
    print("STEP 2: Inferring semantic relationships")
    print("-" * 70)

    edges_path = Path(edges_cache) if edges_cache else exp_dir / "edges.json"

    if skip_inference and edges_path.exists():
        print("Using cached edge inferences (--skip-inference)")
        with open(edges_path, "r") as f:
            edges = json.load(f)
        print(f"Loaded {len(edges)} pre-computed edges")
    elif skip_inference:
        print("No cached edges found — using sample edges")
        edges = _sample_nt_comb_edges()
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        with open(edges_path, "w") as f:
            json.dump(edges, f, indent=2)
    else:
        print("Running LLM edge inference (uses API credits)...")
        try:
            inferencer = SemanticEdgeInferencer()
            edges = inferencer.infer_all_edges(theorems)
            edges_path.parent.mkdir(parents=True, exist_ok=True)
            with open(edges_path, "w") as f:
                json.dump(edges, f, indent=2)
            print(f"Cached edges to {edges_path}")
        except ValueError as e:
            print(f"Error: {e}")
            print("Use --skip-inference to run without API calls")
            return

    # Step 3: Build graph
    print("\nSTEP 3: Building knowledge graph")
    print("-" * 70)
    kg_out = Path(output_kg) if output_kg else exp_dir / "mathkg.json"
    kg = MathKG()
    kg.add_theorems(theorems)
    kg.add_semantic_edges(edges)
    kg_out.parent.mkdir(parents=True, exist_ok=True)
    kg.save(kg_out)
    kg.print_statistics()

    # Step 4: Demo queries
    print("STEP 4: Demonstration queries")
    run_demo_queries(kg)

    # Step 5: Visualize
    print("STEP 5: Creating interactive visualization")
    print("-" * 70)
    viz_out = Path(viz_path) if viz_path else exp_dir / "graph_visualization.html"
    viz_out.parent.mkdir(parents=True, exist_ok=True)
    visualize_graph(kg, viz_out)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Experiment dir:  {exp_dir}")
    print(f"  Knowledge graph: {kg_out}")
    print(f"  Visualization:   {viz_out}  (open in browser)")
    print(f"  Cached edges:    {edges_path}")
    print("=" * 70 + "\n")



RELEASED_KG = "outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json"


def _load_or_build_kg(kg_path=None):
    """Load a saved MathKG (default: the released graph); if the file is absent, build a small
    graph from the node file and the hand-crafted sample edges so the loop still runs."""
    path = Path(kg_path or RELEASED_KG)
    kg = MathKG()
    if path.exists():
        kg.load(path)
        print(f"Loaded MathKG from {path}: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
        return kg
    print(f"No graph at {path}; building from the node file with sample edges")
    kg.add_theorems(load_sample_theorems())
    kg.add_semantic_edges(_sample_nt_comb_edges())
    return kg

# ──────────────────────────────────────────────
# Agent mode: 4-agent discovery loop
# ──────────────────────────────────────────────

def run_agent_mode(iterations=None, skip_inference=False, kg_path=None):
    """Load MathKG (the released graph by default) and run the 4-agent discovery loop."""
    from agents import AgentOrchestrator

    print("=" * 70)
    print("MATHAGENT — FOUR-AGENT DISCOVERY LOOP")
    print("=" * 70)

    kg = _load_or_build_kg(kg_path)

    # Run discovery loop
    orchestrator = AgentOrchestrator()
    summary = orchestrator.run_discovery_loop(kg, max_iterations=iterations)

    # Save updated graph
    kg.save(Path("data/mathkg_agent.json"))

    # Update visualization with any new agent-generated nodes
    viz_path = Path("outputs/graph_visualization_agent.html")
    viz_path.parent.mkdir(exist_ok=True)
    visualize_graph(kg, viz_path)
    print(f"Updated visualization: {viz_path}")


# ──────────────────────────────────────────────
# Paper demo mode: 3 iterations, structured output
# ──────────────────────────────────────────────

def run_paper_demo(skip_inference=False, kg_path=None):
    """Run 3 agent iterations on MathKG and emit structured JSON (demo only; not used in the paper)."""
    from agents import AgentOrchestrator

    print("=" * 70)
    print("MATHAGENT — DEMO (3 iterations)")
    print("=" * 70)

    kg = _load_or_build_kg(kg_path)

    orchestrator = AgentOrchestrator()
    summary = orchestrator.run_discovery_loop(kg, max_iterations=3)

    # Save paper-ready JSON
    paper_output = Path("outputs/paper_demo_results.json")
    paper_output.parent.mkdir(exist_ok=True)
    with open(paper_output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPaper demo results saved to: {paper_output}")

    viz_path = Path("outputs/graph_paper_demo.html")
    visualize_graph(kg, viz_path)
    print(f"Paper demo visualization: {viz_path}")


# ──────────────────────────────────────────────
# Sample edges for skip-inference mode
# ──────────────────────────────────────────────

def _sample_nt_comb_edges():
    """Hand-crafted sample edges for NT+combinatorics theorems (no API needed)."""
    return [
        {
            "source": "thm_nt_001", "target": "thm_nt_002",
            "type": "specializes", "confidence": 0.95,
            "reasoning": "Fermat's Little Theorem is a special case of Euler's Theorem when n=p is prime (phi(p)=p-1)"
        },
        {
            "source": "thm_nt_002", "target": "thm_nt_001",
            "type": "generalizes", "confidence": 0.95,
            "reasoning": "Euler's Theorem generalizes Fermat's Little Theorem from prime moduli to all coprime moduli"
        },
        {
            "source": "thm_nt_004", "target": "thm_nt_002",
            "type": "applied-in", "confidence": 0.88,
            "reasoning": "Bezout's Identity is used in the proof of Euler's Theorem (constructing multiplicative inverses)"
        },
        {
            "source": "thm_nt_005", "target": "thm_nt_001",
            "type": "applied-in", "confidence": 0.82,
            "reasoning": "Chinese Remainder Theorem and Fermat's Little Theorem both concern modular arithmetic over coprime moduli"
        },
        {
            "source": "thm_nt_001", "target": "thm_nt_003",
            "type": "analogous-to", "confidence": 0.80,
            "reasoning": "Both Fermat's Little Theorem and Wilson's Theorem characterize prime numbers via modular arithmetic"
        },
        {
            "source": "thm_comb_001", "target": "thm_comb_003",
            "type": "applied-in", "confidence": 0.90,
            "reasoning": "Pigeonhole Principle is the key lemma in the proof of Ramsey's Theorem R(3,3)=6"
        },
        {
            "source": "thm_comb_002", "target": "thm_comb_004",
            "type": "applied-in", "confidence": 0.78,
            "reasoning": "Inclusion-Exclusion is used in the shadow/shifting argument for Erdos-Ko-Rado"
        },
        {
            "source": "thm_nt_001", "target": "thm_comb_001",
            "type": "analogous-to", "confidence": 0.77,
            "reasoning": "Both are fundamental counting arguments over finite sets: FLT counts group orbits, Pigeonhole counts box occupancy"
        },
        {
            "source": "thm_nt_005", "target": "thm_comb_002",
            "type": "analogous-to", "confidence": 0.75,
            "reasoning": "CRT (decompose mod n into coprime factors) and Inclusion-Exclusion (decompose unions into intersections) both use Mobius-type inversion"
        },
        {
            "source": "thm_nt_006", "target": "thm_comb_004",
            "type": "cross-domain-bridge", "confidence": 0.72,
            "reasoning": "Quadratic Reciprocity and Erdos-Ko-Rado both involve extremal bounds on combinatorial/algebraic structures over finite fields"
        },
    ]


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MathAgent")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "agent", "paper-demo"],
        default="pipeline",
        help="pipeline: build graph; agent: run discovery loop; paper-demo: 3-iteration demo",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Pipeline mode only: use pre-computed/sample edges instead of calling the LLM",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of agent discovery iterations (agent mode only)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the node JSON file (default: data/mathlib_fundamentals_verified.json)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential edge inference (overrides config)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for edge inference (default: 50)",
    )
    parser.add_argument(
        "--output-kg",
        type=str,
        default=None,
        help="Override path to save the built KG JSON (default: <experiment-dir>/mathkg.json)",
    )
    parser.add_argument(
        "--edges-cache",
        type=str,
        default=None,
        help="Override path to save/load the edges cache (default: <experiment-dir>/edges.json)",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default=None,
        help="Agent and demo modes: saved MathKG to load (default: the released graph, "
             + RELEASED_KG + ")",
    )
    parser.add_argument(
        "--viz-path",
        type=str,
        default=None,
        help="Override path to save the visualization HTML (default: <experiment-dir>/graph_visualization.html)",
    )

    args = parser.parse_args()

    # Apply CLI overrides to config
    if args.sequential:
        config.EDGE_INFERENCE_PARALLEL = False
    if args.workers is not None:
        config.EDGE_INFERENCE_MAX_WORKERS = args.workers

    if args.mode == "pipeline":
        run_pipeline(
            skip_inference=args.skip_inference,
            data_file=args.data,
            output_kg=args.output_kg,
            edges_cache=args.edges_cache,
            viz_path=args.viz_path,
        )
    elif args.mode == "agent":
        run_agent_mode(iterations=args.iterations, skip_inference=args.skip_inference, kg_path=args.kg_path)
    elif args.mode == "paper-demo":
        run_paper_demo(skip_inference=args.skip_inference, kg_path=args.kg_path)
