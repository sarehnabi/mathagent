"""
IntegratorAgent: evaluates novelty and writes verified results back to MathKG.
"""

from typing import Dict, List, Optional

from .base_agent import BaseAgent
from src.infer_edges import SemanticEdgeInferencer


class IntegratorAgent(BaseAgent):
    """
    Given a conjecture and its proof result:
    1. Evaluates mathematical significance and novelty
    2. If worthwhile, adds the theorem as a new node in MathKG
    3. Adds rule-based edges to the conjecture's supporting theorems (same domain -> specializes,
       otherwise cross-domain-bridge)
    4. Records the proof result's lean4_status on the new node

    This creates the compounding feedback loop: verified results become
    new context for future Explorer / Conjecturer cycles.
    """

    SYSTEM_PROMPT = (
        "You are a senior mathematician and editor. You evaluate whether a new mathematical "
        "result is significant, novel, and correctly stated. You distinguish between trivial "
        "reformulations and genuinely new contributions."
    )

    def __init__(self):
        super().__init__("Integrator")
        self._id_counter = 1000  # Start agent-generated IDs from 1000

    def evaluate_conjecture(self, conjecture: Dict, proof_result: Dict, kg) -> Dict:
        """
        Evaluate whether a proved conjecture is worth adding to the graph.

        Returns:
            {
                "accept": bool,
                "significance": 0.0-1.0,
                "novelty": 0.0-1.0,
                "rationale": str,
                "suggested_name": str
            }
        """
        statement = conjecture.get("statement", "")
        proof_status = proof_result.get("status", "failed")

        # Only evaluate if proof has some positive signal
        if proof_status == "failed":
            return {
                "accept": False,
                "significance": 0.0,
                "novelty": 0.0,
                "rationale": "Proof failed — cannot integrate unverified result",
                "suggested_name": None,
            }

        # Build existing theorem names for novelty check
        graph = kg.graph
        existing_names = [data.get("name", "") for _, data in graph.nodes(data=True)]
        existing_sample = "\n".join(f"- {n}" for n in existing_names[:20])

        prompt = f"""Evaluate this mathematical result for inclusion in a knowledge graph.

CONJECTURE STATEMENT:
{statement}

LEAN 4 PROOF STATUS: {proof_status}
{f"LEAN 4 CODE:\\n{proof_result.get('lean4_code', '')[:500]}" if proof_result.get('lean4_code') else ''}

NOVELTY CONTEXT — existing theorems in the graph:
{existing_sample}

Evaluate:
1. Is this genuinely novel (not a trivial reformulation of an existing result)?
2. Is the statement mathematically significant?
3. Is the statement correctly formulated?

Respond in JSON:
{{
  "accept": true/false,
  "significance": 0.0-1.0,
  "novelty": 0.0-1.0,
  "rationale": "brief explanation",
  "suggested_name": "a short name for this theorem (or null if rejecting)"
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result:
            self.log(f"Evaluation: accept={result.get('accept')}, significance={result.get('significance'):.2f}")
            return result

        # Default: reject if LLM fails
        return {"accept": False, "significance": 0.0, "novelty": 0.0, "rationale": "Evaluation failed", "suggested_name": None}

    def write_to_graph(self, kg, conjecture: Dict, proof_result: Dict, evaluation: Dict) -> Optional[str]:
        """
        Add a verified, evaluated conjecture as a new node in MathKG.

        Returns the new theorem ID, or None if not added.
        """
        if not evaluation.get("accept", False):
            self.log("Conjecture rejected — not adding to graph")
            return None

        new_id = f"thm_agent_{self._id_counter}"
        self._id_counter += 1

        name = evaluation.get("suggested_name") or f"Agent Conjecture {self._id_counter}"
        domain = conjecture.get("domain", "Number Theory")
        # Normalize domain if LLM returns "Both"
        if domain == "Both":
            domain = "Number Theory"

        node_data = {
            "name": name,
            "domain": domain,
            "statement": conjecture.get("statement", ""),
            "type": "theorem",
            "keywords": [],
            "lean4_formal_statement": conjecture.get("lean4_sketch", ""),
            "lean4_proof_code": proof_result.get("lean4_code", ""),
            "lean4_status": proof_result.get("status", "unverified"),
            "agent_generated": True,
            "significance": evaluation.get("significance", 0.0),
            "novelty": evaluation.get("novelty", 0.0),
        }

        kg.graph.add_node(new_id, **node_data)
        self.log(f"Added new theorem node: {new_id} — {name}")

        # Infer edges to supporting theorems
        supporting_ids = conjecture.get("supporting_theorem_ids", [])
        self._infer_and_add_edges(kg, new_id, supporting_ids)

        return new_id

    def _infer_and_add_edges(self, kg, new_id: str, supporting_ids: List[str]):
        """Add edges from new theorem to its supporting theorems."""
        new_node = dict(kg.graph.nodes[new_id])
        for sup_id in supporting_ids:
            if sup_id not in kg.graph.nodes:
                continue
            sup_node = dict(kg.graph.nodes[sup_id])
            # Infer relationship: new theorem likely specializes or applies supporting theorems
            edge_type = "specializes" if sup_node.get("domain") == new_node.get("domain") else "cross-domain-bridge"
            kg.graph.add_edge(new_id, sup_id, edge_type=edge_type, confidence=0.85, reasoning="Agent-inferred edge from proof dependency")
            self.log(f"Added edge: {new_id} --[{edge_type}]--> {sup_id}")
