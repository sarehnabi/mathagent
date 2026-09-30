"""
Statement-fidelity check for verified proofs.

A Lean file that compiles is not a solve unless it proves the theorem it was asked to prove.
`check_statement_fidelity(proof_code, reference_code)` compares the model's file against the
benchmark's formal statement and returns (ok, reason):

  * the reference theorem (by name) must be present, with statement text identical to the
    reference (the text between the theorem name and the ':=' that opens the body) up to
    whitespace and one notation alias (`∑ x in s` for `∑ x ∈ s`, see `_canon`);
    if the reference does not contain exactly one theorem/lemma, the check is skipped and returns ok;
  * every auxiliary declaration the reference supplies (abbrev/def/structure/inductive/...), including
    filled-in answer definitions, must be present with identical text under the same comparison;
  * no `sorry`, `admit`, or `axiom` may appear outside comments.
"""
import re
from typing import Tuple

DECL_KW = r"(?:noncomputable\s+)?(?:private\s+|protected\s+)?(?:theorem|lemma|example|def|abbrev|instance|structure|inductive|class|axiom|opaque)\b"


def strip_comments(code: str) -> str:
    """Remove block comments (/- ... -/) and line comments (-- ...)."""
    code = re.sub(r"/-.*?-/", "", code, flags=re.S)
    return "\n".join(l.split("--")[0] for l in code.split("\n"))


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _canon(s: str) -> str:
    """Whitespace-insensitive text with one syntax alias allowed: the deprecated big-operator binder
    `∑ x in s` / `∏ x in s` is the same syntax as `∑ x ∈ s` / `∏ x ∈ s` (Mathlib renamed it)."""
    s = _ws(s)
    return re.sub(r"([∑∏]\s*[^,∈]*?)\s+in\s+", r"\1 ∈ ", s)


def uncomment_if_fully_commented(code: str) -> str:
    """miniF2F stores a few statements fully commented out ('-- Error: ...' then '-- theorem ...');
    the intended statement is the uncommented text."""
    lines = [l for l in code.split("\n") if l.strip()]
    if lines and all(l.lstrip().startswith("--") for l in lines):
        return "\n".join(l[3:] if l.startswith("-- ") else l.lstrip("-").lstrip() for l in code.split("\n") if not l.startswith("-- Error"))
    return code


def split_decls(code: str):
    """Return [(kind, name, text)] for every line-initial declaration (any indentation) in comment-stripped code."""
    starts = [m.start() for m in re.finditer(r"(?m)^\s*" + DECL_KW, code)]
    decls = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(code)
        text = code[s:e].strip()
        m = re.match(r"(?:noncomputable\s+)?(?:private\s+|protected\s+)?(\w+)\s+(\S+)?", text)
        decls.append((m.group(1) if m else "?", (m.group(2) or "") if m else "", text))
    return decls


def statement_of(text: str) -> str:
    """Theorem text after the name up to the depth-0 ':=' that opens the proof body."""
    m = re.match(r"(?:noncomputable\s+)?(?:private\s+|protected\s+)?(?:theorem|lemma|example)\s+\S+", text)
    body = text[m.end():] if m else text
    depth = 0
    for i, ch in enumerate(body):
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        elif ch == ":" and depth == 0 and body[i:i + 2] == ":=":
            return _canon(body[:i])
    return _canon(body)


def check_statement_fidelity(proof_code: str, reference_code: str) -> Tuple[bool, str]:
    """Check that a compiling proof file proves the reference statement (rules in the module docstring).

    Returns (True, "ok") on success, (True, reason) when the reference has no single theorem and the
    check is skipped, or (False, reason) with a message the Prover feeds back to the model."""
    ref = strip_comments(uncomment_if_fully_commented(reference_code or ""))
    prf = strip_comments(proof_code or "")
    if re.search(r"\bsorry\b|\badmit\b", prf) or re.search(r"(?m)^\s*axiom\b", prf):
        return False, "the file contains sorry, admit, or an axiom"
    r_decls = split_decls(ref)
    r_thms = [(n, t) for k, n, t in r_decls if k in ("theorem", "lemma")]
    if len(r_thms) != 1:
        return True, "reference has no single theorem; fidelity check skipped"
    tname, r_thm = r_thms[0]
    p_decls = split_decls(prf)
    p_thm = [t for k, n, t in p_decls if k in ("theorem", "lemma") and n == tname]
    if not p_thm:
        return False, f"the file does not contain theorem {tname}"
    if statement_of(p_thm[0]) != statement_of(r_thm):
        return False, f"the statement of theorem {tname} was changed; it must be reproduced exactly"
    p_map = {}
    for k, n, t in p_decls:
        p_map.setdefault(n, []).append(t)
    for k, n, t in r_decls:
        if k in ("theorem", "lemma"):
            continue
        cands = p_map.get(n, [])
        if not cands:
            return False, f"the supplied declaration {n} is missing; it must be reproduced exactly"
        if all(_canon(c) != _canon(t) for c in cands):
            return False, f"the supplied declaration {n} was changed; it must be reproduced exactly"
    return True, "ok"
