"""
ProverAgent: generates and refines Lean 4 proofs for conjectures and benchmark problems.
"""
import os

from typing import Dict, List, Optional

import config
from .base_agent import BaseAgent
from src.lean4_verifier import Lean4Verifier
from src.statement_check import check_statement_fidelity

# Sampling temperatures (env-overridable; module constants so evaluate.py logs the exact values in
# run_config.json).
INITIAL_TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE") or 0.3)
REFINE_TEMPERATURE = float(os.environ.get("VLLM_REFINE_TEMPERATURE") or 0.2)


class ProverAgent(BaseAgent):
    """
    Given a conjecture (open-ended mode) or a benchmark problem (targeted mode):
    1. Generates an initial Lean 4 proof attempt
    2. Runs Lean 4 verification
    3. Iteratively refines based on error messages (up to max_attempts, default config.AGENT_MAX_PROOF_ATTEMPTS)

    Returns the proof result including final status and all attempt history.
    """

    SYSTEM_PROMPT = (
        "You are an expert in Lean 4 and Mathlib. You write correct, idiomatic Lean 4 proofs. "
        "You are familiar with Mathlib tactics: exact, apply, simp, ring, omega, norm_num, decide, "
        "have, obtain, rw, induction, rcases, constructor, push_neg, by_contra, linarith, nlinarith, "
        "polyrith, positivity, field_simp, gcongr, aesop, tauto, native_decide, and others. "
        "IMPORTANT Lean 4 syntax rules: "
        "1) Function application uses SPACES not parentheses: write `f x` not `f(x)`. "
        "2) `import Mathlib` must be the VERY FIRST line — nothing before it. "
        "3) Stop immediately once all goals are closed — do NOT add extra tactics after proof is done. "
        "4) Use Lean 4 syntax ONLY, never Lean 3. "
        "5) NEVER use `sorry` — it is not a valid proof and will be rejected. You must complete every proof with real tactics. "
        "When given error messages, you diagnose them precisely and fix the proof."
    )

    def __init__(self):
        super().__init__("Prover")
        self.verifier = Lean4Verifier()

    def attempt_proof(self, conjecture: Dict, kg, max_attempts: int = None) -> Dict:
        """
        Try to prove a conjecture in Lean 4.

        Args:
            conjecture: Dict from ConjecturerAgent with at minimum:
                        - statement (natural language)
                        - lean4_sketch (Lean 4 type signature)
                        - lean4_proof_sketch (informal proof idea)
            kg: MathKG for context (supporting theorems). Can be None for benchmark mode.
            max_attempts: Override config.AGENT_MAX_PROOF_ATTEMPTS (32 for the paper's pass@32 benchmark runs)

        Returns:
            {
                "lean4_code": str,
                "status": "verified" | "failed" | "unavailable",   # per-attempt timeouts are absorbed and retried
                "attempts": int,
                "attempt_history": list of {code, result},
                "final_errors": list[str]
            }
        """
        lean4_sketch = conjecture.get("lean4_sketch", "")
        statement = conjecture.get("statement", "")
        proof_sketch = conjecture.get("lean4_proof_sketch", "")
        supporting_ids = conjecture.get("supporting_theorem_ids", [])

        # Build context from supporting theorems
        context = self._build_proof_context(supporting_ids, kg) if kg else ""

        attempt_history = []
        current_code = None
        current_errors = []
        rejected_empty_verifications = 0

        attempts_limit = max_attempts or config.AGENT_MAX_PROOF_ATTEMPTS
        for attempt_num in range(1, attempts_limit + 1):
            self.log(f"Proof attempt {attempt_num}/{attempts_limit}")

            if attempt_num == 1:
                current_code = self._generate_initial_proof(lean4_sketch, statement, proof_sketch, context)
            else:
                current_code = self._refine_proof(current_code, current_errors, lean4_sketch, context)

            if not current_code:
                self.log("LLM failed to generate proof code")
                break

            verify_result = self.verifier.verify_proof(current_code, theorem_name=f"conjecture_attempt_{attempt_num}")
            attempt_history.append({"attempt": attempt_num, "code": current_code, "result": verify_result})

            if verify_result["status"] == "verified":
                # Solve criterion: Lean accepted the file (no errors, no sorry warning; checked above by
                # Lean4Verifier) AND the statement check passes: assigned theorem present with its
                # statement unchanged, supplied definitions intact, no sorry/admit/axiom (src/statement_check.py).
                ok, reason = check_statement_fidelity(current_code, lean4_sketch)
                if not ok:
                    rejected_empty_verifications += 1
                    self.log(f"Attempt {attempt_num} compiled but failed the statement check: {reason} (total rejected: {rejected_empty_verifications})")
                    current_errors = [f"Code compiles but does not prove the assigned theorem: {reason}. "
                                      "Reproduce the theorem statement and all supplied definitions exactly as given and prove the theorem."]
                    continue

                self.log(f"Proof verified on attempt {attempt_num}")
                return {
                    "lean4_code": current_code,
                    "status": "verified",
                    "attempts": attempt_num,
                    "attempt_history": attempt_history,
                    "final_errors": [],
                    "rejected_empty_verifications": rejected_empty_verifications,
                }
            elif verify_result["status"] == "unavailable":
                self.log(f"Lean unavailable — stopping")
                return {
                    "lean4_code": current_code,
                    "status": "unavailable",
                    "attempts": attempt_num,
                    "attempt_history": attempt_history,
                    "final_errors": verify_result.get("errors", []),
                }
            elif verify_result["status"] == "timeout":
                self.log(f"Attempt {attempt_num} timed out — continuing")
                current_errors = verify_result.get("errors", [])
                continue

            current_errors = verify_result.get("errors", [])
            self.log(f"Attempt {attempt_num} failed: {len(current_errors)} errors")

        return {
            "lean4_code": current_code or "",
            "status": "failed",
            "attempts": len(attempt_history),
            "attempt_history": attempt_history,
            "final_errors": current_errors,
            "rejected_empty_verifications": rejected_empty_verifications,
        }

    def _generate_initial_proof(
        self, lean4_sketch: str, statement: str, proof_sketch: str, context: str
    ) -> Optional[str]:
        """Generate first proof attempt."""
        prompt = f"""Write a complete Lean 4 proof for the following theorem.

NATURAL LANGUAGE STATEMENT:
{statement}

LEAN 4 TYPE SIGNATURE:
{lean4_sketch}

INFORMAL PROOF SKETCH:
{proof_sketch}

RELEVANT MATHLIB CONTEXT:
{context}

CRITICAL RULES:
1. The FIRST line MUST be `import Mathlib` — nothing before it.
2. Do NOT include any `import` statement other than the first line.
3. Use Lean 4 syntax ONLY. Function application uses spaces: `f x` not `f(x)`.
4. Stop immediately once all goals are solved — do NOT add extra tactics.
5. For inequality goals: try `linarith` first, then `nlinarith`, `polyrith`, or `positivity`.
6. For arithmetic: try `omega`, `norm_num`, or `ring`.
7. For decidable propositions on finite types: try `decide` or `native_decide`.
8. Only use Mathlib names you are CERTAIN exist. If a name does not exist, try a different proof strategy entirely.
9. Use Lean 4 syntax ONLY. Do NOT use Lean 3 syntax (no begin/end blocks). Use by followed by tactic lines.

Return ONLY the Lean 4 code, no explanation."""

        response = self.call_llm(prompt, self.SYSTEM_PROMPT, max_tokens=int(os.environ.get("VLLM_MAX_TOKENS") or 4096), temperature=INITIAL_TEMPERATURE)
        return self._extract_lean_code(response)

    def _refine_proof(
        self, previous_code: str, errors: List[str], lean4_sketch: str, context: str
    ) -> Optional[str]:
        """Refine proof based on Lean error messages."""
        error_text = "\n".join(errors[:10])  # Cap at 10 errors for prompt length

        # Identify unknown Mathlib identifiers from error messages so we can
        # warn the model explicitly rather than letting it hallucinate fixes.
        import re
        unknown_ids = re.findall(
            r"Unknown (?:constant|identifier) [`'\"]([^`'\"]+)[`'\"]", "\n".join(errors)
        )
        unknown_id_note = ""
        if unknown_ids:
            unknown_id_note = (
                "\n\nCRITICAL — the following identifiers do NOT exist in Mathlib. "
                "Do NOT use them. Find the correct Mathlib name or use a completely "
                "different proof strategy that avoids these identifiers:\n"
                + "\n".join(f"  - {uid}" for uid in unknown_ids)
            )

        prompt = f"""The following Lean 4 proof has errors. Fix it.

LEAN 4 CODE:
```lean
{previous_code}
```

ERRORS FROM LEAN:
{error_text}{unknown_id_note}

THEOREM TYPE SIGNATURE (must be preserved):
{lean4_sketch}

MATHLIB CONTEXT:
{context}

CRITICAL RULES:
- The FIRST line MUST be `import Mathlib` — nothing before it, no duplicate imports.
- Use Lean 4 syntax ONLY: function application uses spaces `f x` not `f(x)`.
- Stop immediately once all goals are solved — do NOT add extra tactics.

Common fixes:
- If 'unknown identifier' / 'unknown constant': the name does not exist in Mathlib.
  Try a different lemma or a different proof strategy entirely.
- If 'type mismatch': add explicit type annotations or use `show` tactic.
- If 'No goals to be solved' / 'goals accomplished': you have EXTRA tactics after
  the proof is done. REMOVE all tactics after the point where goals are closed.
- If 'unexpected token' / 'unexpected identifier': you have a SYNTAX error.
  Check: no Lean 3 syntax, correct indentation, correct keyword usage.
- If 'Function expected': you are applying something that is not a function.
  Check parentheses and application syntax.
- If 'failed to synthesize': add missing typeclass instances with `@[instance]` or `haveI`.
- If 'linarith failed': try `nlinarith`, `polyrith`, or `positivity` instead.
- For arithmetic goals: try `omega`, `norm_num`, `ring`, or `field_simp`.
- For decidable propositions: try `decide` or `native_decide`.
- Only use Mathlib names you are CERTAIN exist. If a name doesn't exist, try a different lemma or a different proof strategy entirely.
9. Use Lean 4 syntax ONLY. Do NOT use Lean 3 syntax (no begin/end blocks). Use by followed by tactic lines.

Return ONLY the corrected Lean 4 code, no explanation."""

        refine_max_tokens = int(os.environ.get("VLLM_MAX_TOKENS") or 4096)
        response = self.call_llm(prompt, self.SYSTEM_PROMPT, max_tokens=refine_max_tokens, temperature=REFINE_TEMPERATURE)
        return self._extract_lean_code(response)

    def _build_proof_context(self, supporting_ids: List[str], kg) -> str:
        """Build relevant Mathlib context from supporting theorems."""
        lines = []
        graph = kg.graph
        for tid in supporting_ids:
            node = dict(graph.nodes.get(tid, {}))
            if node.get("lean4_formal_statement"):
                lines.append(f"-- {node.get('name', tid)}")
                lines.append(node["lean4_formal_statement"])
                if node.get("lean4_mathlib_name"):
                    lines.append(f"-- Mathlib name: {node['lean4_mathlib_name']}")
                if node.get("lean4_mathlib_import"):
                    lines.append(f"-- Import: {node['lean4_mathlib_import']}")
                lines.append("")
        return "\n".join(lines) if lines else "No specific Mathlib context available. Use `import Mathlib`."

    def _extract_lean_code(self, response: str) -> Optional[str]:
        """Extract Lean 4 code from LLM response and fix common issues."""
        import re

        if not response:
            return None  # no response (e.g. call failed) -> clean failed attempt, not a crash

        # First try: the LAST fenced ```lean / ```lean4 block
        matches = re.findall(r"```lean4?\n(.*?)\n```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()  # Take the last match
            if code:
                return self._fix_lean_code(code)

        # Fallback: remove markdown fences and extract raw code
        cleaned = re.sub(r"```(?:lean4?|lean)?\s*", "", response)
        cleaned = re.sub(r"```", "", cleaned)

        # Remove explanatory text before code
        lines = cleaned.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("theorem ") or stripped.startswith("def "):
                in_code = True
            if in_code:
                if stripped.startswith("### ") or stripped.startswith("**") or stripped.startswith("Explanation"):
                    break
                code_lines.append(line)

        code = "\n".join(code_lines).strip()

        if not code:
            # Last resort: accept the whole response if it starts with `import` or `theorem`
            stripped = response.strip()
            if stripped.startswith("import") or stripped.startswith("theorem"):
                code = stripped
            else:
                return None

        return self._fix_lean_code(code)


    def _fix_lean_code(self, code: str) -> str:
        """Fix common Lean 4 code issues before verification."""
        import re
        lines = code.split("\n")

        # Fix 1: Ensure `import Mathlib` is the very first line
        # Remove any duplicate or misplaced import statements
        import_lines = []
        other_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import "):
                import_lines.append(stripped)
            else:
                other_lines.append(line)

        # Always start with import Mathlib
        if not any("Mathlib" in imp for imp in import_lines):
            import_lines = ["import Mathlib"]
        else:
            # Keep only Mathlib imports; all other imports are dropped
            import_lines = [imp for imp in import_lines if "Mathlib" in imp]
            if not import_lines:
                import_lines = ["import Mathlib"]

        # Fix 2: Remove blank lines between imports and code
        while other_lines and other_lines[0].strip() == "":
            other_lines.pop(0)

        # Reconstruct
        code = "\n".join(import_lines + [""] + other_lines)

        return code
