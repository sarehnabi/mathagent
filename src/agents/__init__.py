"""
MathAgent: four LLM agents over MathKG. Targeted problem-solving (Explorer -> Prover) is the
evaluated path; the open-ended discovery loop (Explorer -> Conjecturer -> Prover -> Integrator)
is implemented but not evaluated.

Agents:
- ExplorerAgent: gathers context for a problem (Layer 1 KG semantic search, Layer 2 Mathlib
  declaration search); in open-ended mode, finds structural gaps and cross-domain analogies
- ConjecturerAgent: generates candidate theorems grounded in graph structure
- ProverAgent: attempts Lean 4 proof synthesis with error-guided refinement
- IntegratorAgent: evaluates novelty and writes verified results back to graph

Coordinator:
- AgentOrchestrator: deterministic controller for both loops (no LLM calls)
"""

from .explorer import ExplorerAgent
from .conjecturer import ConjecturerAgent
from .prover import ProverAgent
from .integrator import IntegratorAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    "ExplorerAgent",
    "ConjecturerAgent",
    "ProverAgent",
    "IntegratorAgent",
    "AgentOrchestrator",
]
