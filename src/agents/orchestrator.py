"""
AgentOrchestrator: deterministic coordinator for two modes.

Targeted problem-solving (the evaluated path): Explorer -> Prover.
Open-ended discovery (implemented, not evaluated): Explorer -> Conjecturer -> Prover -> Integrator -> (repeat)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from .explorer import ExplorerAgent
from .conjecturer import ConjecturerAgent
from .prover import ProverAgent
from .integrator import IntegratorAgent


class AgentOrchestrator:
    """
    Runs the MathAgent discovery loop for N iterations.

    Each iteration:
    1. Explorer scans MathKG for gaps and analogies
    2. Conjecturer generates a candidate theorem from the top target
    3. Prover attempts Lean 4 verification
    4. Integrator evaluates and (if accepted) writes back to graph

    Results are logged to outputs/discovery_session.json (both modes write this file; later runs overwrite it).
    """

    def __init__(self):
        self.explorer = ExplorerAgent("Explorer")
        self.conjecturer = ConjecturerAgent("Conjecturer")
        self.prover = ProverAgent()
        self.integrator = IntegratorAgent()

        self.session_log: List[Dict] = []
        self.verified_theorems: List[str] = []
        self.conjectures_generated: int = 0
        self.conjectures_verified: int = 0

    def run_discovery_loop(self, kg, max_iterations: int = None) -> Dict:
        """
        Run the four-agent discovery loop.

        Args:
            kg: MathKG instance
            max_iterations: Number of iterations (defaults to config.AGENT_MAX_ITERATIONS)

        Returns:
            Session summary dict with statistics and discovered theorems.
        """
        if max_iterations is None:
            max_iterations = config.AGENT_MAX_ITERATIONS

        print(f"\n{'='*60}")
        print(f"MathAgent Discovery Loop — {max_iterations} iterations")
        print(f"Starting graph: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
        print(f"{'='*60}\n")

        start_time = datetime.now()

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            iteration_result = self._run_iteration(iteration, kg)
            self.session_log.append(iteration_result)

            if iteration_result.get("new_theorem_id"):
                self.verified_theorems.append(iteration_result["new_theorem_id"])
                print(f"[Iteration {iteration}] New theorem added: {iteration_result['new_theorem_id']}")
            else:
                print(f"[Iteration {iteration}] No new theorem added")

        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        summary = {
            "session_start": start_time.isoformat(),
            "session_end": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "iterations": max_iterations,
            "conjectures_generated": self.conjectures_generated,
            "conjectures_verified": self.conjectures_verified,
            "new_theorems": self.verified_theorems,
            "final_graph_nodes": kg.graph.number_of_nodes(),
            "final_graph_edges": kg.graph.number_of_edges(),
            "iteration_log": self.session_log,
        }

        self._save_session(summary)
        self._print_summary(summary)
        return summary

    def _run_iteration(self, iteration: int, kg) -> Dict:
        """Execute one full Explorer -> Conjecturer -> Prover -> Integrator cycle."""
        result = {
            "iteration": iteration,
            "targets_found": 0,
            "conjecture": None,
            "proof_result": None,
            "evaluation": None,
            "new_theorem_id": None,
        }

        # Step 1: Explorer
        print(f"  [Explorer] Scanning graph...")
        targets = self.explorer.suggest_investigation_targets(kg)
        result["targets_found"] = len(targets)
        print(f"  [Explorer] Found {len(targets)} targets")

        if not targets:
            print("  [Explorer] No targets found — skipping iteration")
            return result

        # Pick top target
        top_target = targets[0]

        # Step 2: Conjecturer
        print(f"  [Conjecturer] Generating conjecture for {top_target['target_type']} target...")
        conjecture = self.conjecturer.generate_conjecture(top_target, kg)

        if not conjecture:
            print("  [Conjecturer] Failed to generate conjecture")
            return result

        self.conjectures_generated += 1
        result["conjecture"] = {
            "statement": conjecture.get("statement", ""),
            "confidence": conjecture.get("confidence", 0),
            "domain": conjecture.get("domain", ""),
        }
        print(f"  [Conjecturer] Conjecture: {conjecture.get('statement', '')[:80]}...")

        # Step 3: Prover
        print(f"  [Prover] Attempting Lean 4 proof...")
        proof_result = self.prover.attempt_proof(conjecture, kg)
        result["proof_result"] = {
            "status": proof_result["status"],
            "attempts": proof_result["attempts"],
            "final_errors": proof_result.get("final_errors", [])[:3],
            "lean4_code": proof_result.get("lean4_code", ""),
        }
        print(f"  [Prover] Status: {proof_result['status']} (after {proof_result['attempts']} attempts)")

        # Step 4: Integrator
        print(f"  [Integrator] Evaluating result...")
        evaluation = self.integrator.evaluate_conjecture(conjecture, proof_result, kg)
        result["evaluation"] = {
            "accept": evaluation.get("accept"),
            "significance": evaluation.get("significance", 0),
            "novelty": evaluation.get("novelty", 0),
        }

        if evaluation.get("accept"):
            new_id = self.integrator.write_to_graph(kg, conjecture, proof_result, evaluation)
            result["new_theorem_id"] = new_id
            if proof_result["status"] == "verified":
                self.conjectures_verified += 1

        return result

    # ═══════════════════════════════════════════════════════════════════
    # TARGETED MODE: Solve a specific problem
    # ═══════════════════════════════════════════════════════════════════

    def run_targeted_loop(self, kg, problem: Dict, max_iterations: int = None) -> Dict:
        """
        Run discovery loop targeting a specific problem.

        Each iteration tries a DIFFERENT approach. The system exhausts
        multiple strategies before reporting results.

        Args:
            kg: MathKG instance
            problem: Dict with 'statement' (and optionally 'domain', 'formal_statement', 'id')
            max_iterations: Number of approach attempts

        Returns:
            Session summary with per-approach results.
        """
        if max_iterations is None:
            max_iterations = config.AGENT_MAX_ITERATIONS

        problem_id = problem.get("id", "unknown")
        problem_text = problem.get("statement", "") or problem.get("formal_statement", "")

        print(f"\n{'='*60}")
        print(f"MathAgent TARGETED Mode — Problem: {problem_id}")
        print(f"Statement: {problem_text[:100]}...")
        print(f"Approaches to try: {max_iterations}")
        print(f"{'='*60}\n")

        start_time = datetime.now()
        failed_approaches = []

        # Step 1: Find relevant theorems from KG
        print("  [Explorer] Searching KG for relevant theorems...")
        relevant = self.explorer.find_relevant_theorems_for_problem(problem, kg)
        print(f"  [Explorer] Found {len(relevant)} relevant theorems")

        # Step 2: Search Mathlib for additional specialized lemmas
        print("  [Explorer] Searching Mathlib declarations...")
        mathlib_results = self.explorer.search_mathlib_declarations(problem)
        print(f"  [Explorer] Found {len(mathlib_results)} Mathlib matches")

        # Step 3: Get ranked approaches
        print("  [Explorer] Suggesting proof approaches...")
        approaches = self.explorer.suggest_problem_approaches(problem, kg, relevant)
        print(f"  [Explorer] Suggested {len(approaches)} approaches")

        # Step 4: Try each approach
        for iteration in range(1, max_iterations + 1):
            if iteration <= len(approaches):
                approach = approaches[iteration - 1]
                approach_name = approach.get("approach", f"approach_{iteration}")
            else:
                approach_name = f"fallback_approach_{iteration}"
                approach = {"approach": approach_name, "strategy": "Try a different angle based on previous failures."}

            print(f"\n--- Approach {iteration}/{max_iterations}: {approach_name} ---")

            # Build context from approach + relevant theorems + previous failures
            context = self._build_targeted_context(approach, relevant, mathlib_results, failed_approaches)

            # Create a conjecture-like dict for the Prover
            conjecture = {
                "statement": problem_text,
                "lean4_sketch": problem.get("formal_statement", ""),
                "lean4_proof_sketch": approach.get("strategy", ""),
                "supporting_theorem_ids": [t.get("theorem_id", "") for t in relevant[:5]],
                "domain": problem.get("domain", ""),
                "confidence": approach.get("confidence", 0.5),
            }

            # Prover attempts
            print(f"  [Prover] Attempting proof with {approach_name}...")
            proof_result = self.prover.attempt_proof(conjecture, kg)
            print(f"  [Prover] Status: {proof_result['status']} ({proof_result['attempts']} attempts)")

            self.session_log.append({
                "iteration": iteration,
                "approach": approach_name,
                "proof_status": proof_result["status"],
                "proof_attempts": proof_result["attempts"],
                "errors": proof_result.get("final_errors", [])[:3],
            })

            if proof_result["status"] == "verified":
                print(f"\n  ✓ SOLVED on approach {iteration}: {approach_name}")
                self.conjectures_verified += 1

                # Integrate into KG
                evaluation = self.integrator.evaluate_conjecture(conjecture, proof_result, kg)
                if evaluation.get("accept"):
                    new_id = self.integrator.write_to_graph(kg, conjecture, proof_result, evaluation)
                    self.verified_theorems.append(new_id)

                return self._build_targeted_summary(
                    problem, start_time, "solved", iteration, proof_result
                )

            # Record failure for next iteration
            failed_approaches.append({
                "approach": approach_name,
                "errors": proof_result.get("final_errors", [])[:5],
            })

        # All approaches exhausted
        return self._build_targeted_summary(
            problem, start_time, "unsolved", max_iterations, None
        )

    def _build_targeted_context(self, approach: Dict, relevant: List[Dict],
                                 mathlib_results: List[Dict],
                                 failed_approaches: List[Dict]) -> str:
        """Build context string for the Prover from all available information."""
        lines = []
        if approach.get("strategy"):
            lines.append(f"APPROACH: {approach['strategy']}")
        if relevant:
            lines.append("\nRELEVANT KG THEOREMS:")
            for t in relevant[:5]:
                lines.append(f"  - {t.get('name', t.get('theorem_id', ''))}")
        if mathlib_results:
            lines.append("\nMATHLIB LEMMAS:")
            for m in mathlib_results[:5]:
                lines.append(f"  - {m['name']} ({m['kind']})")
        if failed_approaches:
            lines.append("\nPREVIOUS FAILED APPROACHES (avoid repeating):")
            for fa in failed_approaches:
                lines.append(f"  - {fa['approach']}: {'; '.join(fa['errors'][:2])}")
        return "\n".join(lines)

    def _build_targeted_summary(self, problem: Dict, start_time,
                                 status: str, iterations: int,
                                 proof_result: Optional[Dict]) -> Dict:
        """Build summary for a targeted problem-solving session."""
        end_time = datetime.now()
        summary = {
            "mode": "targeted",
            "problem_id": problem.get("id", "unknown"),
            "problem_domain": problem.get("domain", ""),
            "status": status,
            "iterations": iterations,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "session_start": start_time.isoformat(),
            "session_end": end_time.isoformat(),
            "iteration_log": self.session_log,
        }
        if proof_result and status == "solved":
            summary["lean4_code"] = proof_result.get("lean4_code", "")
            summary["proof_attempts"] = proof_result.get("attempts", 0)

        self._save_session(summary)
        self._print_targeted_summary(summary)
        return summary

    def _print_targeted_summary(self, summary: Dict):
        """Print targeted mode summary."""
        print(f"\n{'='*60}")
        print(f"TARGETED SESSION: {summary['problem_id']}")
        print(f"{'='*60}")
        print(f"Status:     {summary['status'].upper()}")
        print(f"Approaches: {summary['iterations']}")
        print(f"Duration:   {summary['duration_seconds']:.1f}s")
        if summary['status'] == 'solved':
            print(f"Proof attempts: {summary.get('proof_attempts', '?')}")
        print(f"{'='*60}\n")

    def _save_session(self, summary: Dict):
        """Save session log to outputs/discovery_session.json."""
        output_path = Path("outputs") / "discovery_session.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSession log saved to {output_path}")

    def _print_summary(self, summary: Dict):
        """Print a formatted summary of the discovery session."""
        print(f"\n{'='*60}")
        print("DISCOVERY SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration:              {summary['duration_seconds']:.1f}s")
        print(f"Iterations:            {summary['iterations']}")
        print(f"Conjectures generated: {summary['conjectures_generated']}")
        print(f"Conjectures verified:  {summary['conjectures_verified']}")
        print(f"New theorems added:    {len(summary['new_theorems'])}")
        print(f"Final graph:           {summary['final_graph_nodes']} nodes, {summary['final_graph_edges']} edges")
        if summary["new_theorems"]:
            print(f"New theorem IDs:       {', '.join(summary['new_theorems'])}")
        print(f"{'='*60}\n")
