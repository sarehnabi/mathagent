"""
LLM-based comprehensive semantic edge inference between mathematical theorems.

This module systematically checks ALL theorem pairs to find semantic relationships.
Core innovation: using Claude to identify semantic relationships beyond syntactic dependencies.

Scales to any number of theorems (25, 100, 1000+).
"""

import json
import threading
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

# Import based on configuration
if config.USE_AWS_BEDROCK:
    import boto3
else:
    import anthropic


class SemanticEdgeInferencer:
    """Infer semantic relationships between theorems using Claude."""

    EDGE_TYPES = [
        "prerequisite",          # A (definition) is needed to state/prove B
        "generalizes",           # A strictly contains B as a special case (theorem→theorem)
        "specializes",           # A is a strict special case of B (theorem→theorem)
        "equivalent-to",         # A ↔ B (same abstraction, different formulation)
        "applied-in",            # A's result is used inside B's proof
        "analogous-to",          # Same formal structure, different domain
        "cross-domain-bridge",   # A's and B's domains meet through this pair
        "instance-of",           # A (concrete object) exemplifies structure B
    ]

    def __init__(self):
        """Initialize the Claude client (either AWS Bedrock or direct API)."""
        self.use_bedrock = config.USE_AWS_BEDROCK
        self.inference_count = 0

        if self.use_bedrock:
            # Initialize AWS Bedrock client
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.AWS_REGION
            )
            import os
            self.model_id = (
                os.environ.get('EDGE_INFERENCE_MODEL_ID')
                or getattr(config, 'EDGE_INFERENCE_MODEL_ID', config.MODEL_ID)
            )
            print(f"✓ Using AWS Bedrock with model: {self.model_id}")
        else:
            # Initialize direct Anthropic API client
            if config.ANTHROPIC_API_KEY == "your-api-key-here":
                raise ValueError(
                    "Please set your ANTHROPIC_API_KEY in config.py"
                )
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model_id = config.MODEL_NAME
            print(f"✓ Using Anthropic API with model: {self.model_id}")

    def infer_relationship(
        self,
        theorem_a: Dict,
        theorem_b: Dict
    ) -> Tuple[str, float, str]:
        """
        Infer semantic relationship between two theorems.

        Args:
            theorem_a: First theorem dict
            theorem_b: Second theorem dict

        Returns:
            Tuple of (edge_type, confidence, reasoning) or (None, 0.0, "") if no relationship
        """
        prompt = self._build_prompt(theorem_a, theorem_b)

        # Hybrid retry schedule for Bedrock throttling:
        #   Early retries (short) — recover from transient network blips
        #   Later retries  (long) — wait out a full ~60s TPM rolling window
        # Jitter added to each delay to prevent thundering-herd retries when
        # many parallel workers are throttled at the same instant.
        RETRY_DELAYS = [2, 4, 8, 15, 30, 45, 60, 90]   # cumulative ≈ 254s
        max_retries = len(RETRY_DELAYS) + 1            # +1 because no sleep before attempt 0
        import random
        import time
        for attempt in range(max_retries):
            try:
                if self.use_bedrock:
                    response = self._call_bedrock(prompt)
                else:
                    response = self._call_anthropic(prompt)

                self.inference_count += 1
                return self._parse_response(response)

            except Exception as e:
                if attempt >= len(RETRY_DELAYS):
                    print(f"  ⚠ Edge inference failed after {max_retries} attempts: {e}")
                    return None, 0.0, ""
                delay = RETRY_DELAYS[attempt] + random.uniform(0, 1.0)
                print(f"  ⚠ API error, retrying in {delay:.1f}s... ({type(e).__name__}) [attempt {attempt+1}/{max_retries}]")
                time.sleep(delay)

    def _call_bedrock(self, prompt: str) -> str:
        """Call AWS Bedrock API."""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    def _call_anthropic(self, prompt: str) -> str:
        """Call direct Anthropic API."""
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _build_prompt(self, theorem_a: Dict, theorem_b: Dict) -> str:
        """Build the prompt for the LLM with the 8-type ontology, type-aware guidance, and few-shot examples."""
        def format_node(label: str, thm: Dict) -> str:
            kind = thm.get("type", "theorem").upper()
            lines = [
                f"{label} ({kind}):",
                f"Name: {thm['name']}",
                f"Domain: {thm['domain']}",
                f"Statement: {thm['statement']}",
                f"Keywords: {', '.join(thm.get('keywords', []))}",
            ]
            if thm.get("lean4_mathlib_name"):
                lines.append(f"Mathlib name: {thm['lean4_mathlib_name']}")
            if thm.get("lean4_mathlib_import"):
                lines.append(f"Mathlib import: {thm['lean4_mathlib_import']}")
            if thm.get("lean4_formal_statement"):
                lines.append(f"Lean 4 formal statement: {thm['lean4_formal_statement']}")
            return "\n".join(lines)

        return f"""You are a mathematical reasoning expert building a knowledge graph of fundamental mathematics. You must classify the relationship from NODE A to NODE B (directional) using a precise ontology. Be conservative: only identify strong, clear relationships.

{format_node("NODE A", theorem_a)}

{format_node("NODE B", theorem_b)}

==== ONTOLOGY (8 edge types) ====

1. prerequisite
   A (typically a DEFINITION) is required to state or prove B.
   Example: "Prime Number" --prerequisite--> "Fermat's Little Theorem"
   Restriction: A must be a DEFINITION. If A is a theorem used inside B's proof, use "applied-in" instead.

2. generalizes
   A is strictly more general; B is recoverable from A by fixing parameters/structure.
   Example: "Binomial Theorem" --generalizes--> "Sum of Binomial Coefficients = 2^n"
   Restriction: Both A and B must be THEOREMS (not definitions).

3. specializes
   Inverse of generalizes. A is a strict special case of B.
   Example: "Fermat's Little Theorem" --specializes--> "Lagrange's Theorem (group theory)"
   Restriction: Both A and B must be THEOREMS.

4. equivalent-to
   A ↔ B at the same level of abstraction. Different formulations of the same fact.
   Example: "Heine-Borel (compact iff closed and bounded in R^n)" --equivalent-to--> "Bolzano-Weierstrass (bounded sequence has convergent subsequence in R^n)"
   Restriction: Both A and B must be THEOREMS. Pick only when both directions genuinely hold. Do NOT use for "related" or "both about X".

5. applied-in
   A's RESULT is used as a step in B's proof.
   Example: "Fundamental Theorem of Arithmetic" --applied-in--> "Lifting the Exponent Lemma"
   Restriction: Both A and B must be THEOREMS. A is a tool; B is what uses the tool. Definitions do not have proofs, so they cannot appear as B. If A is a definition, use "prerequisite" instead.

6. analogous-to
   A and B have the same formal structure (same theorem schema or proof pattern) in different domains or contexts.
   Example: "Hensel's Lemma (lifting roots p-adically)" --analogous-to--> "Implicit Function Theorem (lifting solutions smoothly)"
   Restriction: Both A and B must be THEOREMS. Must be a real structural correspondence, not just "both involve X". If you cannot state the shared schema in one sentence, it is not analogous.

7. cross-domain-bridge
   A pair where A's domain and B's domain meet: use this when ANY of the following hold:
     (a) one side's object is defined using the other's structure, OR
     (b) a canonical/standard theorem establishes a formal correspondence between them
         (e.g., gcd(a,b)·Z = (a,b) as an ideal — the formal link between GCD and Ideal), OR
     (c) translation between the two domains routinely passes through this pair.
   Example: "Vector Space" --cross-domain-bridge--> "Field" (vector spaces are modules over fields)
   Example: "Greatest Common Divisor" --cross-domain-bridge--> "Ideal" (gcd(a,b) generates the ideal (a,b) in Z)
   Restriction: Nodes must be in DIFFERENT domains. The connection must be a recognized correspondence,
   not mere surface similarity.

8. instance-of
   A is a concrete object that exemplifies the abstract structure B.
   Example: "Ring ZMod n" --instance-of--> "Ring" / "Commutative Ring"
   Restriction: Both A and B must be DEFINITIONS (not theorems). A must be a concrete/named object; B must be an abstract structure or class that A belongs to.

==== WHEN TO RETURN "none" ====

Return "none" if:
- The connection is only "both are about X" (e.g., both about primes, both about groups) without a structural relationship.
- The analogy is vague ("both establish invariants", "both are foundational").
- You would rate confidence below 0.75.

It is perfectly acceptable — and preferred — to return "none" often. A sparse, high-quality graph beats a dense noisy one.

==== FEW-SHOT EXAMPLES ====

Example 1 — strong generalizes:
  NODE A: "Binomial Theorem" (theorem, Combinatorics): (x+y)^n = Σ C(n,k) x^(n-k) y^k
  NODE B: "Sum of Binomial Coefficients" (theorem, Combinatorics): Σ C(n,k) = 2^n
  → {{"relationship": "generalizes", "confidence": 0.97, "reasoning": "Setting x=y=1 in the Binomial Theorem yields Σ C(n,k) = 2^n. B is literally a substitution instance of A."}}

Example 2 — strong prerequisite:
  NODE A: "Prime Number" (definition, Number Theory): natural number p ≥ 2 divisible only by 1 and p
  NODE B: "Fermat's Little Theorem" (theorem, Number Theory): if p prime, a^p ≡ a (mod p)
  → {{"relationship": "prerequisite", "confidence": 0.98, "reasoning": "The definition of prime number is required to state B: the hypothesis 'p prime' is essential and primality is used in the proof."}}

Example 3 — "none" case (both about primes but no structural link):
  NODE A: "Wilson's Theorem" (theorem, Number Theory): (p-1)! ≡ -1 (mod p) for p prime
  NODE B: "Bertrand's Postulate" (theorem, Number Theory): for n ≥ 1 there is a prime in (n, 2n]
  → {{"relationship": "none", "confidence": 1.0, "reasoning": "Both concern primes but share no structural relationship. Neither is used in the other's proof, neither generalizes the other, and they address unrelated questions (factorial modular identity vs. prime gap bound)."}}

Example 4 — analogous-to (different domains, shared schema):
  NODE A: "Hensel's Lemma" (theorem, Number Theory): lifts a simple root of f mod p to a root in Z_p
  NODE B: "Implicit Function Theorem" (theorem, Analysis): lifts an approximate solution to an exact smooth solution under non-degeneracy
  → {{"relationship": "analogous-to", "confidence": 0.85, "reasoning": "Both are lifting theorems with the same schema: 'given an approximate solution and a non-degeneracy condition (nonzero derivative / Jacobian), a true solution exists and is unique locally'. Direct structural parallel across different domains."}}

Example 5 — cross-domain-bridge:
  NODE A: "Vector Space" (definition, Linear Algebra): module over a field
  NODE B: "Field" (definition, Abstract Algebra): commutative ring where every nonzero element is invertible
  → {{"relationship": "cross-domain-bridge", "confidence": 0.92, "reasoning": "A vector space is defined as a module over a field; B's structure is the scalar domain of A. This is the foundational bridge between linear algebra and field theory."}}

Example 6 — instance-of:
  NODE A: "Ring ZMod n" (definition, Number Theory): Z/nZ with addition and multiplication mod n
  NODE B: "Commutative Ring" (definition, Abstract Algebra): ring where multiplication is commutative
  → {{"relationship": "instance-of", "confidence": 0.95, "reasoning": "ZMod n is a concrete example of a commutative ring. The abstract structure B is instantiated by the concrete object A."}}

Example 7 — applied-in:
  NODE A: "Fundamental Theorem of Arithmetic" (theorem, Number Theory): unique prime factorization
  NODE B: "Lifting the Exponent Lemma" (theorem, Number Theory): v_p(a^n - b^n) = v_p(a-b) + v_p(n) under conditions
  → {{"relationship": "applied-in", "confidence": 0.88, "reasoning": "LTE relies on p-adic valuations, which are well-defined only because FTA guarantees unique prime factorizations. FTA's result is invoked in B's proof."}}

==== TASK ====

Classify the relationship from NODE A to NODE B (directional). Use Mathlib names/imports to assess structural proximity when useful.

Respond with EXACTLY this JSON (no other text):
{{
  "relationship": "<one of: prerequisite | generalizes | specializes | equivalent-to | applied-in | analogous-to | cross-domain-bridge | instance-of | none>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<1-4 sentences. State the specific structural correspondence, not just 'both relate to X'.>"
}}

If relationship is "none", set confidence to 1.0 (confident no relationship exists)."""

    def _parse_response(self, response: str) -> Tuple[str, float, str]:
        """Parse Claude's JSON response, extracting the first complete JSON object."""
        try:
            start = response.find('{')
            if start == -1:
                return None, 0.0, ""

            # Find matching closing brace by counting nesting depth,
            # ignoring braces inside quoted strings.
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(response)):
                ch = response[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = response[start:i + 1]
                        break
            else:
                return None, 0.0, ""

            data = json.loads(json_str)

            relationship = data.get('relationship', 'none')
            confidence = float(data.get('confidence', 0.0))
            reasoning = data.get('reasoning', '')

            if relationship == 'none' or confidence < config.SIMILARITY_THRESHOLD:
                return None, 0.0, ""

            return relationship, confidence, reasoning

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠ Failed to parse response: {e}")
            return None, 0.0, ""

    # Relative priority for dedup: higher = kept when multiple edges exist for the same (source, target)
    _TYPE_PRIORITY = {
        "instance-of": 9,
        "prerequisite": 8,
        "generalizes": 7,
        "specializes": 7,
        "equivalent-to": 6,
        "applied-in": 5,
        "cross-domain-bridge": 4,
        "analogous-to": 3,
    }

    # Pairs of edge types where A→B type_a implies B→A type_b
    _INVERSE_TYPES = {
        "generalizes": "specializes",
        "specializes": "generalizes",
        "instance-of": None,              # no automatic inverse (too strong to infer "is-instantiated-by")
        "prerequisite": None,             # no useful inverse
        "applied-in": None,               # no useful inverse
        "equivalent-to": "equivalent-to", # symmetric
        "analogous-to": "analogous-to",   # symmetric
        "cross-domain-bridge": "cross-domain-bridge",  # symmetric
    }

    # Node-type restrictions per edge type: (allowed A types, allowed B types)
    # None means any type is allowed for that position.
    _TYPE_RESTRICTIONS = {
        "prerequisite":        ({"definition"}, None),
        "generalizes":         ({"theorem"}, {"theorem"}),
        "specializes":         ({"theorem"}, {"theorem"}),
        "equivalent-to":       ({"theorem"}, {"theorem"}),
        "applied-in":          ({"theorem"}, {"theorem"}),
        "analogous-to":        ({"theorem"}, {"theorem"}),
        "cross-domain-bridge": (None, None),  # domain check handled separately
        "instance-of":         ({"definition"}, {"definition"}),
    }

    def _post_process_edges(self, edges: List[Dict], nodes_by_id: Dict = None) -> List[Dict]:
        """Deduplicate, enforce type restrictions, resolve conflicts, and add inverses.

        Rules:
          1. Dedup (source, target): if multiple edges exist, keep the highest-priority type.
          2. Type-restriction enforcement: drop edges that violate node-type constraints
             (e.g., applied-in where B is a definition) and same-domain cross-domain-bridge.
          3. Conflict resolution: if A-generalizes->B AND B-generalizes->A both exist, drop both.
          4. Symmetry: for symmetric types, ensure the reverse edge exists.
          5. Inverse pairs: if A-generalizes->B exists, ensure B-specializes->A exists.
        """
        # Step 1: Dedup by (source, target), keep highest priority
        best: Dict[Tuple[str, str], Dict] = {}
        for e in edges:
            key = (e['source'], e['target'])
            if key not in best:
                best[key] = e
            else:
                current_pri = self._TYPE_PRIORITY.get(best[key]['type'], 0)
                new_pri = self._TYPE_PRIORITY.get(e['type'], 0)
                if (new_pri, e['confidence']) > (current_pri, best[key]['confidence']):
                    best[key] = e
        deduped = list(best.values())

        # Step 2: Enforce type restrictions (requires node metadata)
        if nodes_by_id:
            before = len(deduped)
            cleaned = []
            type_violations = 0
            domain_violations = 0
            for e in deduped:
                src_node = nodes_by_id.get(e['source'], {})
                tgt_node = nodes_by_id.get(e['target'], {})
                src_type = src_node.get('type', 'theorem')
                tgt_type = tgt_node.get('type', 'theorem')
                etype = e['type']

                # Check node-type restrictions
                restrictions = self._TYPE_RESTRICTIONS.get(etype)
                if restrictions:
                    allowed_a, allowed_b = restrictions
                    if allowed_a and src_type not in allowed_a:
                        type_violations += 1
                        continue
                    if allowed_b and tgt_type not in allowed_b:
                        type_violations += 1
                        continue

                # Check same-domain violation for cross-domain-bridge
                if etype == 'cross-domain-bridge':
                    if src_node.get('domain') == tgt_node.get('domain'):
                        domain_violations += 1
                        continue

                cleaned.append(e)
            deduped = cleaned
            if type_violations:
                print(f"  ⚠ Dropped {type_violations} edges violating node-type restrictions")
            if domain_violations:
                print(f"  ⚠ Dropped {domain_violations} same-domain cross-domain-bridge edges")

        # Step 3: Drop contradictory generalizes↔generalizes and specializes↔specializes
        drop_keys = set()
        idx = {(e['source'], e['target']): e for e in deduped}
        for e in deduped:
            rev_key = (e['target'], e['source'])
            rev = idx.get(rev_key)
            if rev is None:
                continue
            if e['type'] == 'generalizes' and rev['type'] == 'generalizes':
                drop_keys.add((e['source'], e['target']))
                drop_keys.add(rev_key)
            if e['type'] == 'specializes' and rev['type'] == 'specializes':
                drop_keys.add((e['source'], e['target']))
                drop_keys.add(rev_key)
        if drop_keys:
            deduped = [e for e in deduped if (e['source'], e['target']) not in drop_keys]
            print(f"  ⚠ Dropped {len(drop_keys)} contradictory edges (bidirectional generalizes/specializes)")

        # Step 4 & 5: Enforce symmetry and add inverses
        idx = {(e['source'], e['target']): e for e in deduped}
        additions = []
        for e in deduped:
            inverse_type = self._INVERSE_TYPES.get(e['type'])
            if inverse_type is None:
                continue
            rev_key = (e['target'], e['source'])
            if rev_key in idx:
                continue
            additions.append({
                'source': e['target'],
                'target': e['source'],
                'type': inverse_type,
                'confidence': e['confidence'],
                'reasoning': f"[auto-inverse of {e['source']} → {e['target']} ({e['type']})] {e['reasoning']}",
                'auto_inverse': True,
            })

        if additions:
            print(f"  ✓ Added {len(additions)} inverse edges for symmetric/paired types")
        return deduped + additions

    def _infer_pair(self, theorem_a: Dict, theorem_b: Dict) -> List[Dict]:
        """
        Infer edges for a single pair in both directions.
        Returns list of edge dicts (0, 1, or 2 edges).
        Thread-safe.
        """
        results = []

        # Check A → B
        edge_type, confidence, reasoning = self.infer_relationship(theorem_a, theorem_b)
        if edge_type and confidence >= config.SIMILARITY_THRESHOLD:
            results.append({
                'source': theorem_a['id'],
                'target': theorem_b['id'],
                'type': edge_type,
                'confidence': confidence,
                'reasoning': reasoning
            })

        # Check B → A (relationships may be directional)
        edge_type, confidence, reasoning = self.infer_relationship(theorem_b, theorem_a)
        if edge_type and confidence >= config.SIMILARITY_THRESHOLD:
            results.append({
                'source': theorem_b['id'],
                'target': theorem_a['id'],
                'type': edge_type,
                'confidence': confidence,
                'reasoning': reasoning
            })

        return results

    def infer_all_edges(
        self,
        theorems: List[Dict],
        max_edges_per_theorem: int = None,
        comprehensive: bool = True,
        parallel: bool = None,
        max_workers: int = None,
    ) -> List[Dict]:
        """
        Comprehensively infer semantic edges for ALL theorem pairs.

        Args:
            theorems: List of theorems
            max_edges_per_theorem: Ignored (kept for API compatibility)
            comprehensive: Ignored (kept for API compatibility); all ordered pairs are always checked
            parallel: If True, use ThreadPoolExecutor. Defaults to config.EDGE_INFERENCE_PARALLEL.
            max_workers: Max concurrent threads. Defaults to config.EDGE_INFERENCE_MAX_WORKERS.

        Returns:
            List of edge dictionaries with source, target, type, confidence, reasoning
        """
        if parallel is None:
            parallel = getattr(config, 'EDGE_INFERENCE_PARALLEL', False)
        if max_workers is None:
            max_workers = getattr(config, 'EDGE_INFERENCE_MAX_WORKERS', 50)

        edges = []
        n = len(theorems)
        total_pairs = n * (n - 1) // 2

        # Build all pairs
        pairs = [
            (theorems[i], theorems[j])
            for i in range(n) for j in range(i + 1, n)
        ]

        mode_str = f"PARALLEL ({max_workers} workers)" if parallel else "SEQUENTIAL"

        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE SEMANTIC EDGE INFERENCE — {mode_str}")
        print(f"{'='*70}")
        print(f"Theorems: {n}")
        print(f"Total pairs to check: {total_pairs}")
        print(f"Confidence threshold: {config.SIMILARITY_THRESHOLD}")
        print(f"Estimated API calls: {total_pairs * 2} (checking both directions)")
        print(f"{'='*70}\n")

        # Track progress (thread-safe)
        checked = [0]
        found = [0]
        lock = threading.Lock()

        def process_pair(pair):
            theorem_a, theorem_b = pair
            result = self._infer_pair(theorem_a, theorem_b)
            with lock:
                checked[0] += 1
                found[0] += len(result)
                if checked[0] % 10 == 0:
                    progress = checked[0] / total_pairs * 100
                    print(f"  Progress: {checked[0]}/{total_pairs} ({progress:.1f}%) — Found {found[0]} edges")
                for edge in result:
                    print(f"  ✓ {edge['source']} → {edge['target']}: {edge['type']} ({edge['confidence']:.2f})")
            return result

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(process_pair, pair): pair for pair in pairs}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        edges.extend(result)
                    except Exception as e:
                        print(f"  ⚠ Pair failed: {e}")
        else:
            for pair in pairs:
                result = process_pair(pair)
                edges.extend(result)

        raw_count = len(edges)
        print(f"\n{'='*70}")
        print(f"POST-PROCESSING (dedup, type-restriction enforcement, conflict resolution, symmetry)")
        print(f"{'='*70}")
        nodes_by_id = {t['id']: t for t in theorems}
        edges = self._post_process_edges(edges, nodes_by_id=nodes_by_id)

        print(f"\n{'='*70}")
        print(f"INFERENCE COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Pairs checked: {checked[0]}")
        print(f"✓ Raw edges from API: {raw_count}")
        print(f"✓ Final edges after post-processing: {len(edges)}")
        if total_pairs > 0:
            print(f"✓ Edge density: {len(edges)/(total_pairs*2)*100:.2f}%")
        print(f"✓ Total API calls: {self.inference_count}")
        print(f"✓ Mode: {mode_str}")
        print(f"{'='*70}\n")

        return edges


if __name__ == "__main__":
    # Test edge inference
    from extract_mathlib import load_sample_theorems

    theorems = load_sample_theorems()
    inferencer = SemanticEdgeInferencer()

    # Test on two theorems
    thm1 = theorems[0]  # Lagrange's Theorem
    thm2 = theorems[3]  # Sylow's First Theorem

    print(f"\n--- Testing Edge Inference ---")
    print(f"Theorem 1: {thm1['name']}")
    print(f"Theorem 2: {thm2['name']}")

    edge_type, confidence, reasoning = inferencer.infer_relationship(thm1, thm2)

    if edge_type:
        print(f"\n✓ Relationship: {edge_type}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Reasoning: {reasoning}")
    else:
        print("\n- No strong relationship detected")
