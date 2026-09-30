"""
Load MathKG nodes (theorems and definitions) from a JSON node file.

Node schema:
- Basic: id, name, domain, statement, type, keywords
- Lean 4: lean4_mathlib_name, lean4_mathlib_import, lean4_status
- Optional legacy fields (absent from the released node file): lean4_formal_statement, lean4_proof_code
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

LEAN4_FIELDS = [
    "lean4_formal_statement",
    "lean4_mathlib_name",
    "lean4_proof_code",
    "lean4_mathlib_import",
    "lean4_status",
]


def load_sample_theorems(filepath: Optional[str] = None) -> List[Dict]:
    """
    Load theorems from a JSON file.

    Args:
        filepath: Path to the node JSON file. Defaults to data/mathlib_fundamentals_verified.json.

    Returns:
        List of theorem dicts. Extended theorems include lean4_* fields.
    """
    if filepath is not None:
        data_path = Path(filepath)
    else:
        data_path = Path(__file__).parent.parent / "data" / "mathlib_fundamentals_verified.json"

    with open(data_path, 'r') as f:
        theorems = json.load(f)

    lean4_count = sum(1 for t in theorems if has_lean4_data(t))
    print(f"Loaded {len(theorems)} theorems from {data_path.name}")
    print(f"  Domains: {sorted(set(t['domain'] for t in theorems))}")
    print(f"  With Lean 4 data: {lean4_count}/{len(theorems)}")

    return theorems


def has_lean4_data(theorem: Dict) -> bool:
    """Return True if theorem has at least a lean4_formal_statement."""
    return bool(theorem.get("lean4_formal_statement"))


def get_lean4_metadata(theorem: Dict) -> Dict:
    """Extract Lean 4 fields from a theorem dict. Missing fields return None."""
    return {field: theorem.get(field) for field in LEAN4_FIELDS}


def extract_by_domain(theorems: List[Dict], domain: str) -> List[Dict]:
    """Filter theorems by domain."""
    filtered = [t for t in theorems if t['domain'] == domain]
    print(f"Filtered to {len(filtered)} theorems in {domain}")
    return filtered


def get_theorem_by_id(theorems: List[Dict], theorem_id: str) -> Optional[Dict]:
    """Retrieve a specific theorem by ID."""
    for theorem in theorems:
        if theorem['id'] == theorem_id:
            return theorem
    return None


def format_for_prompt(theorem: Dict, include_lean4: bool = True) -> str:
    """
    Format a theorem for inclusion in an LLM prompt.
    If include_lean4 is True and Lean 4 data exists, includes formal statement.
    """
    lines = [
        f"Name: {theorem['name']}",
        f"Domain: {theorem['domain']}",
        f"Statement: {theorem['statement']}",
        f"Keywords: {', '.join(theorem.get('keywords', []))}",
    ]
    if include_lean4 and theorem.get("lean4_formal_statement"):
        lines.append(f"Lean 4 formal statement: {theorem['lean4_formal_statement']}")
        if theorem.get("lean4_status"):
            lines.append(f"Lean 4 status: {theorem['lean4_status']}")
    return "\n".join(lines)


if __name__ == "__main__":
    theorems = load_sample_theorems()

    print("\n--- Sample Theorems ---")
    for thm in theorems[:3]:
        print(f"\n{thm['name']} ({thm['domain']})")
        print(f"  {thm['statement']}")
        if has_lean4_data(thm):
            print(f"  Lean 4: {thm['lean4_formal_statement'][:80]}...")
        print(f"  Keywords: {', '.join(thm['keywords'])}")
