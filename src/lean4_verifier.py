"""
Lean 4 verification module for MathAgent.

Wraps the lean CLI to verify generated proofs. Falls back gracefully
when Lean 4 is not installed.
"""

import os
import resource
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

LEAN_MEMORY_LIMIT_BYTES = int(os.environ.get(
    "LEAN_MEMORY_LIMIT_GB", "200"
)) * 1024**3


def _setup_process_group():
    """Start subprocess in its own process group + set memory limit."""
    os.setpgrp()
    resource.setrlimit(resource.RLIMIT_AS, (LEAN_MEMORY_LIMIT_BYTES, LEAN_MEMORY_LIMIT_BYTES))

import config


class Lean4Verifier:
    """
    Calls the lean CLI to check Lean 4 proof code.

    Two modes of operation:
    - lean available: write .lean file, run `lean`, parse output
    - lean unavailable: return status='unavailable' without error
    """

    LEAN4_HEADER = "import Mathlib\nopen BigOperators\n\n"

    def __init__(self):
        self.lean_path = shutil.which("lean")
        self.available = self.lean_path is not None
        self.temp_dir = Path(config.LEAN_TEMP_DIR)
        self.timeout = int(os.environ.get("LEAN_TIMEOUT_SECONDS", config.LEAN_TIMEOUT_SECONDS))

        # Prefer running via `lake env lean` inside the Mathlib project so that
        # `import Mathlib` resolves correctly.  Fall back to bare `lean` if the
        # project directory is not configured or does not exist.
        self.mathlib_project = (
            os.environ.get("LEAN_MATHLIB_PROJECT")
            or getattr(config, "LEAN_MATHLIB_PROJECT", None)
        )
        self.lake_path = shutil.which("lake")
        self.use_lake = (
            self.lake_path is not None
            and self.mathlib_project is not None
            and os.path.isdir(self.mathlib_project)
        )

        if not hasattr(Lean4Verifier, '_init_logged'):
            if self.available:
                if self.use_lake:
                    print(f"Lean 4 found at: {self.lean_path} (using lake env from {self.mathlib_project})")
                else:
                    print(f"Lean 4 found at: {self.lean_path} (WARNING: lake/mathlib project not found — Mathlib imports will fail)")
            else:
                print("Lean 4 not found on PATH — verification will return 'unavailable'")
            Lean4Verifier._init_logged = True

        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def verify_proof(self, lean_code: str, theorem_name: str = "theorem") -> Dict:
        """
        Attempt to verify a Lean 4 proof.

        Args:
            lean_code: Complete Lean 4 code (imports + theorem + proof).
                       If it doesn't start with 'import', LEAN4_HEADER is prepended.
            theorem_name: Name used for the temp file.

        Returns:
            {
                "status": "verified" | "failed" | "timeout" | "unavailable",
                "errors": list of error strings,
                "warnings": list of warning strings,
                "time_ms": int,
                "lean_output": str   # first 2,000 chars of stdout; absent on the timeout and exception paths
            }
        """
        if not self.available:
            return {"status": "unavailable", "errors": [], "warnings": [], "time_ms": 0}

        if not lean_code.lstrip().startswith("import"):
            lean_code = self.LEAN4_HEADER + lean_code

        lean_file = self.temp_dir / f"{theorem_name.replace(' ', '_')}.lean"
        lean_file.write_text(lean_code, encoding="utf-8")

        start = time.time()
        try:
            if self.use_lake:
                cmd = [self.lake_path, "env", self.lean_path, lean_file.as_posix()]
                cwd = self.mathlib_project
            else:
                cmd = [self.lean_path, lean_file.as_posix()]
                cwd = None

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                preexec_fn=_setup_process_group,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return {
                    "status": "timeout",
                    "errors": [f"Lean timed out after {self.timeout}s"],
                    "warnings": [],
                    "time_ms": self.timeout * 1000,
                }

            elapsed_ms = int((time.time() - start) * 1000)

            diagnostic_output = stdout + ("\n" + stderr if stderr else "")
            errors = self._parse_diagnostics(diagnostic_output, "error")
            warnings = self._parse_diagnostics(diagnostic_output, "warning")

            status = "verified" if proc.returncode == 0 and not errors else "failed"

            if status == "verified":
                has_sorry_warning = any('sorry' in w.lower() for w in warnings)
                if has_sorry_warning:
                    status = "failed"
                    errors.append("Proof uses 'sorry' — not a complete proof")

            return {
                "status": status,
                "errors": errors,
                "warnings": warnings,
                "time_ms": elapsed_ms,
                "lean_output": stdout[:2000] if stdout else "",
            }
        except Exception as e:
            return {
                "status": "failed",
                "errors": [f"Subprocess error: {str(e)}"],
                "warnings": [],
                "time_ms": 0,
            }
        finally:
            lean_file.unlink(missing_ok=True)

    def check_mathlib_name(self, mathlib_name: str) -> Dict:
        """
        Check if a Mathlib declaration exists by running `#check <name>`.

        Returns:
            {"exists": bool | None, "type_signature": str | None, "status": str, "errors": list (failed path only)}
        """
        if not self.available:
            return {"exists": None, "type_signature": None, "status": "unavailable"}

        code = f"{self.LEAN4_HEADER}#check @{mathlib_name}\n"
        result = self.verify_proof(code, theorem_name=f"check_{mathlib_name.replace('.', '_')}")

        if result["status"] == "verified":
            return {"exists": True, "type_signature": result.get("lean_output", ""), "status": "verified"}
        elif result["status"] == "failed":
            is_not_found = any("unknown identifier" in e or "unknown constant" in e for e in result["errors"])
            return {"exists": not is_not_found, "type_signature": None, "status": "failed", "errors": result["errors"]}
        else:
            return {"exists": None, "type_signature": None, "status": result["status"]}

    def batch_verify(self, theorems: List[Dict]) -> List[Dict]:
        """
        Verify every theorem that has lean4_proof_code; theorems already marked 'verified' are re-checked when Lean is available.

        Updates each theorem's lean4_status in-place and returns the updated list.
        """
        updated = []
        for thm in theorems:
            if not thm.get("lean4_proof_code"):
                updated.append(thm)
                continue

            # Re-check already-verified theorems when Lean is available
            if thm.get("lean4_status") == "verified" and self.available:
                result = self.verify_proof(thm["lean4_proof_code"], thm.get("name", thm["id"]))
                thm = dict(thm)
                thm["lean4_status"] = result["status"]
                thm["lean4_verify_result"] = result
            elif thm.get("lean4_status") != "verified":
                result = self.verify_proof(thm["lean4_proof_code"], thm.get("name", thm["id"]))
                thm = dict(thm)
                thm["lean4_status"] = result["status"]
                thm["lean4_verify_result"] = result

            updated.append(thm)

        verified = sum(1 for t in updated if t.get("lean4_status") == "verified")
        print(f"Lean 4 batch verification: {verified}/{len(updated)} verified")
        return updated

    def _parse_diagnostics(self, output: str, level: str) -> List[str]:
        """Extract error or warning lines from lean stderr/stdout output.

        Lean 4 uses two formats:
          - Classic:  'file:line:col: error: message'
          - Typed:    'file:line:col: error(lean.code): message'
        Both start with ': error' after the location prefix.
        """
        import re
        lines = output.splitlines()
        # Match ': error:' or ': error(' (typed errors) or line starting with level keyword
        pattern = re.compile(rf": {re.escape(level)}[:(]", re.IGNORECASE)
        return [line for line in lines if pattern.search(line) or line.strip().lower().startswith(level + ":") or line.strip().lower().startswith(level + "(")]
