"""
Re-run post-processing on an existing edges.json file.
Applies type-restriction enforcement, dedup, conflict resolution, and symmetry
without re-running the LLM edge inference.

Usage:
    python scripts/reprocess_edges.py \
        --edges outputs/experiments/exp_.../edges.json \
        --nodes data/mathlib_fundamentals_verified.json \
        --output outputs/experiments/exp_.../edges_clean.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.infer_edges import SemanticEdgeInferencer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--nodes', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    edges = json.load(open(args.edges))
    nodes = json.load(open(args.nodes))
    nodes_by_id = {n['id']: n for n in nodes}

    # Strip auto-inverse edges (they'll be re-added by post-processing)
    raw = [e for e in edges if not e.get('auto_inverse')]
    print(f"Input: {len(edges)} total edges ({len(raw)} raw, {len(edges)-len(raw)} auto-inverse stripped)")
    print()

    # Type distribution before
    types_before = Counter(e['type'] for e in raw)
    print("Edge types BEFORE post-processing:")
    for t, c in types_before.most_common():
        print(f"  {t:25s} {c}")
    print()

    # Run post-processing
    inferencer = SemanticEdgeInferencer()
    clean = inferencer._post_process_edges(raw, nodes_by_id=nodes_by_id)

    # Type distribution after
    raw_clean = [e for e in clean if not e.get('auto_inverse')]
    auto_clean = [e for e in clean if e.get('auto_inverse')]
    types_after = Counter(e['type'] for e in raw_clean)
    print()
    print("Edge types AFTER post-processing:")
    for t, c in types_after.most_common():
        print(f"  {t:25s} {c}")
    print()

    # Summary
    dropped = len(raw) - len(raw_clean)
    print(f"Summary:")
    print(f"  Raw edges before:     {len(raw)}")
    print(f"  Dropped (violations): {dropped}")
    print(f"  Raw edges after:      {len(raw_clean)}")
    print(f"  Auto-inverse added:   {len(auto_clean)}")
    print(f"  Final total:          {len(clean)}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\n✓ Saved to {out}")


if __name__ == '__main__':
    main()
