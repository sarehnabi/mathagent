"""
ConjecturerAgent: generates candidate theorems grounded in MathKG structure.
"""

from typing import Dict, List, Optional

import config
from .base_agent import BaseAgent
from src.extract_mathlib import format_for_prompt


class ConjecturerAgent(BaseAgent):
    """
    Given an investigation target (gap or analogy) from the ExplorerAgent,
    generates a candidate theorem statement with a Lean 4 sketch.

    Grounding: uses Lean 4 formal statements from existing theorems as
    few-shot examples so the output is anchored in Mathlib conventions.
    """

    SYSTEM_PROMPT = (
        "You are a creative yet rigorous mathematician specializing in number theory "
        "and combinatorics. You generate novel but plausible mathematical conjectures "
        "grounded in known results. You are familiar with Lean 4 and Mathlib syntax."
    )

    def generate_conjecture(self, target: Dict, kg) -> Optional[Dict]:
        """
        Generate a candidate theorem from an investigation target.

        Args:
            target: Output dict from ExplorerAgent.suggest_investigation_targets()
            kg: The current MathKG graph

        Returns:
            {
                "statement": str (natural language),
                "lean4_sketch": str (Lean 4 type signature),
                "lean4_proof_sketch": str (informal proof idea),
                "confidence": float,
                "domain": str,
                "supporting_theorem_ids": list[str],
                "novelty_rationale": str
            }
            or None if conjecture generation fails.
        """
        graph = kg.graph

        # Gather supporting theorems with Lean 4 data for few-shot grounding
        supporting_ids = self._get_supporting_ids(target, graph)
        lean4_examples = self._build_lean4_examples(supporting_ids, graph)

        target_type = target.get("target_type", "unknown")
        target_data = target.get("data", {})

        if target_type == "analogy":
            return self._conjecture_from_analogy(target_data, lean4_examples, graph)
        elif target_type == "gap":
            return self._conjecture_from_gap(target_data, lean4_examples, graph)
        else:
            self.log(f"Unknown target type: {target_type}")
            return None

    def _conjecture_from_analogy(self, analogy: Dict, lean4_examples: str, graph) -> Optional[Dict]:
        """Generate a conjecture by deepening a known analogy."""
        thm_a_id = analogy.get("theorem_a_id", "")
        thm_b_id = analogy.get("theorem_b_id", "")

        thm_a = dict(graph.nodes.get(thm_a_id, {}))
        thm_b = dict(graph.nodes.get(thm_b_id, {}))

        prompt = f"""Two theorems share a deep structural analogy:

THEOREM A ({thm_a.get('domain','')}: {thm_a.get('name','')}):
{thm_a.get('statement', '')}
{f"Lean 4: {thm_a.get('lean4_formal_statement', '')}" if thm_a.get('lean4_formal_statement') else ''}

THEOREM B ({thm_b.get('domain','')}: {thm_b.get('name','')}):
{thm_b.get('statement', '')}
{f"Lean 4: {thm_b.get('lean4_formal_statement', '')}" if thm_b.get('lean4_formal_statement') else ''}

Shared principle: {analogy.get('shared_principle', '')}
Conjecture potential: {analogy.get('conjecture_potential', '')}

EXISTING LEAN 4 FORMAL STATEMENTS FOR REFERENCE:
{lean4_examples}

Based on this analogy, propose ONE novel mathematical conjecture that:
1. Generalizes both theorems to a common framework, OR
2. Transfers a technique from one domain to solve an open question in the other, OR
3. Identifies a third theorem in a different domain that completes the structural pattern

The conjecture should be non-trivial, potentially true, and amenable to Lean 4 formalization.

Respond in JSON:
{{
  "statement": "natural language statement of the conjecture",
  "lean4_sketch": "Lean 4 type signature (theorem name : statement), following Mathlib conventions",
  "lean4_proof_sketch": "informal description of a proof strategy",
  "confidence": 0.0-1.0,
  "domain": "Number Theory | Combinatorics | Both",
  "novelty_rationale": "why this is a non-trivial new result",
  "supporting_theorem_ids": ["{thm_a_id}", "{thm_b_id}"]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result and result.get("confidence", 0) >= config.AGENT_CONJECTURE_CONFIDENCE_THRESHOLD:
            self.log(f"Generated analogy-based conjecture: {result.get('statement', '')[:80]}...")
            return result
        elif result:
            self.log(f"Conjecture confidence {result.get('confidence', 0):.2f} below threshold, skipping")
        return None

    def _conjecture_from_gap(self, gap: Dict, lean4_examples: str, graph) -> Optional[Dict]:
        """Generate a conjecture to fill a structural gap."""
        thm_id = gap.get("theorem_id", "")
        thm = dict(graph.nodes.get(thm_id, {}))

        prompt = f"""A theorem in our knowledge graph has a structural gap:

THEOREM ({thm.get('domain','')}: {thm.get('name','')}):
{thm.get('statement', '')}
{f"Lean 4: {thm.get('lean4_formal_statement', '')}" if thm.get('lean4_formal_statement') else ''}

Gap analysis: {gap.get('rationale', '')}
Missing connection type: {gap.get('gap_type', '')}
Suggested connection domain: {gap.get('suggested_connection_domain', '')}

EXISTING LEAN 4 FORMAL STATEMENTS FOR REFERENCE:
{lean4_examples}

Propose ONE conjecture that fills this gap — a new theorem that would connect
this result to the domain identified above. The conjecture should:
1. Be mathematically plausible
2. Create a meaningful new connection
3. Be formalizable in Lean 4 / Mathlib

Respond in JSON:
{{
  "statement": "natural language statement",
  "lean4_sketch": "Lean 4 type signature following Mathlib conventions",
  "lean4_proof_sketch": "informal proof idea",
  "confidence": 0.0-1.0,
  "domain": "Number Theory | Combinatorics | Both",
  "novelty_rationale": "why this connection is non-trivial",
  "supporting_theorem_ids": ["{thm_id}"]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result and result.get("confidence", 0) >= config.AGENT_CONJECTURE_CONFIDENCE_THRESHOLD:
            self.log(f"Generated gap-filling conjecture: {result.get('statement', '')[:80]}...")
            return result
        return None

    def _get_supporting_ids(self, target: Dict, graph) -> List[str]:
        """Extract theorem IDs referenced by the target."""
        data = target.get("data", {})
        ids = []
        for key in ("theorem_id", "theorem_a_id", "theorem_b_id"):
            if data.get(key) and data[key] in graph.nodes:
                ids.append(data[key])
        return ids

    def _build_lean4_examples(self, theorem_ids: List[str], graph) -> str:
        """Build a few-shot Lean 4 examples string from existing theorems."""
        examples = []
        for tid in theorem_ids:
            node = dict(graph.nodes.get(tid, {}))
            if node.get("lean4_formal_statement"):
                examples.append(
                    f"-- {node.get('name', tid)} ({node.get('domain', '')})\n"
                    f"{node['lean4_formal_statement']}"
                )
        # Also add a few more theorems with lean4 data for broader context
        for nid in list(graph.nodes)[:10]:
            if nid not in theorem_ids:
                node = dict(graph.nodes[nid])
                if node.get("lean4_formal_statement"):
                    examples.append(
                        f"-- {node.get('name', nid)} ({node.get('domain', '')})\n"
                        f"{node['lean4_formal_statement']}"
                    )
            if len(examples) >= 5:
                break

        return "\n\n".join(examples) if examples else "No Lean 4 examples available."
