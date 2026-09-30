"""
ExplorerAgent: identifies structural gaps and cross-domain analogies in MathKG.

KG Search Pipeline (for targeted problem-solving):
  Step 1: Domain pre-filter — filter to same domain + cross-domain bridges
  Step 2: Score candidates — LLM scoring (default) or keyword matching
  Step 3: Edge walk — traverse KG edges from K seeds (configurable hops)
  Step 4: Enrich context — attach Mathlib names + edge reasoning
"""

from collections import deque
from typing import Dict, List, Set

import config
from .base_agent import BaseAgent
from src.extract_mathlib import format_for_prompt


class ExplorerAgent(BaseAgent):
    """
    Analyzes the current MathKG to find:
    1. Nodes with low connectivity (potential gaps in the graph)
    2. Cross-domain analogies (currently restricted to the Number Theory / Combinatorics pair)

    Returns investigation targets for the ConjecturerAgent. In targeted problem-solving mode the
    Explorer instead gathers context for a given problem: Layer 1 MathKG semantic search (seed
    scoring + 1-hop edge walk) and Layer 2 Mathlib declaration search.
    """

    SYSTEM_PROMPT = (
        "You are an expert mathematician specializing in number theory and combinatorics. "
        "You excel at identifying structural patterns, analogies between mathematical domains, "
        "and gaps in mathematical knowledge. You reason carefully about formal mathematics."
    )

    def find_structural_gaps(self, kg) -> List[Dict]:
        """
        Find theorems with low connectivity that may represent underexplored areas.

        Returns list of {theorem_id, name, domain, degree, gap_type, rationale}.
        """
        graph = kg.graph
        node_degrees = dict(graph.degree())

        # Identify low-degree nodes (degree <= median)
        degrees = list(node_degrees.values())
        if not degrees:
            return []
        median_degree = sorted(degrees)[len(degrees) // 2]

        low_connectivity = [
            {"id": nid, **dict(graph.nodes[nid])}
            for nid, deg in node_degrees.items()
            if deg <= median_degree
        ]

        if not low_connectivity:
            return []

        # Ask the LLM which gaps are mathematically significant
        theorem_summaries = "\n\n".join(
            f"ID: {t['id']}\nName: {t.get('name', 'Unknown')}\nDomain: {t.get('domain', 'Unknown')}\n"
            f"Statement: {t.get('statement', '')}\nCurrent connections: {node_degrees.get(t['id'], 0)}"
            for t in low_connectivity[:8]
        )

        prompt = f"""The following theorems have low connectivity in our mathematical knowledge graph
(fewer semantic edges than average). Each has at most {median_degree} connections.

{theorem_summaries}

Analyze which of these represent genuine mathematical gaps (areas where more connections
should exist but haven't been captured), versus theorems that are naturally isolated.

For the top 3 most interesting gaps, explain:
1. Why this theorem should have more connections
2. What kind of relationship is missing (analogy, generalization, application, etc.)
3. What area of mathematics the missing connection might come from

Respond in JSON:
{{
  "gaps": [
    {{
      "theorem_id": "...",
      "theorem_name": "...",
      "gap_type": "missing_analogy | missing_generalization | missing_application | isolated",
      "rationale": "...",
      "suggested_connection_domain": "..."
    }}
  ]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result and "gaps" in result:
            self.log(f"Found {len(result['gaps'])} structural gaps")
            return result["gaps"]
        return []

    def find_cross_domain_analogies(self, kg) -> List[Dict]:
        """
        Find Number Theory <-> Combinatorics theorem pairs with high analogy potential.

        Returns list of {theorem_a_id, theorem_b_id, analogy_score, rationale}.
        """
        graph = kg.graph
        nt_theorems = [
            {"id": nid, **dict(graph.nodes[nid])}
            for nid in graph.nodes
            if graph.nodes[nid].get("domain") == "Number Theory"
        ]
        comb_theorems = [
            {"id": nid, **dict(graph.nodes[nid])}
            for nid in graph.nodes
            if graph.nodes[nid].get("domain") == "Combinatorics"
        ]

        if not nt_theorems or not comb_theorems:
            self.log("Cannot find analogies: missing Number Theory or Combinatorics nodes")
            return []

        # Build compact representations
        nt_list = "\n".join(f"- [{t['id']}] {t.get('name','')}: {t.get('statement','')[:100]}" for t in nt_theorems)
        comb_list = "\n".join(f"- [{t['id']}] {t.get('name','')}: {t.get('statement','')[:100]}" for t in comb_theorems)

        prompt = f"""I have theorems from two mathematical domains and want to find deep structural analogies.

NUMBER THEORY THEOREMS:
{nt_list}

COMBINATORICS THEOREMS:
{comb_list}

Identify the 3 most compelling cross-domain analogies between these theorems.
A strong analogy means: same underlying structure, proof technique, or mathematical principle
operating in different settings (e.g., Fermat's Little Theorem and Pigeonhole both involve
counting arguments over finite sets).

Respond in JSON:
{{
  "analogies": [
    {{
      "theorem_a_id": "...",
      "theorem_b_id": "...",
      "analogy_score": 0.0-1.0,
      "shared_principle": "one-sentence description of the shared structure",
      "rationale": "detailed explanation of the analogy",
      "conjecture_potential": "what new theorem might emerge from this analogy"
    }}
  ]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result and "analogies" in result:
            self.log(f"Found {len(result['analogies'])} cross-domain analogies")
            return result["analogies"]
        return []

    def suggest_investigation_targets(self, kg) -> List[Dict]:
        """
        Combine gap analysis and analogy detection into a ranked list of targets
        for the ConjecturerAgent to investigate.

        Returns list of {target_type, data, priority, rationale}.
        """
        self.log("Scanning graph for gaps and analogies...")

        gaps = self.find_structural_gaps(kg)
        analogies = self.find_cross_domain_analogies(kg)

        targets = []
        for gap in gaps:
            targets.append({
                "target_type": "gap",
                "data": gap,
                "priority": 0.8 if gap.get("gap_type") == "missing_analogy" else 0.6,
                "rationale": gap.get("rationale", ""),
            })
        for analogy in analogies:
            targets.append({
                "target_type": "analogy",
                "data": analogy,
                "priority": analogy.get("analogy_score", 0.7),
                "rationale": analogy.get("rationale", ""),
            })

        # Sort by priority descending
        targets.sort(key=lambda x: x["priority"], reverse=True)
        self.log(f"Returning {len(targets)} investigation targets")
        return targets

    # ═══════════════════════════════════════════════════════════════════
    # TARGETED PROBLEM-SOLVING (benchmark problems: miniF2F, PutnamBench, MathOlympiadBench)
    # ═══════════════════════════════════════════════════════════════════

    def find_relevant_theorems_for_problem(self, problem: Dict, kg,
                                            scoring_method: str = None,
                                            num_seeds: int = None,
                                            hop_depth: int = None) -> List[Dict]:
        """
        Search the KG for theorems relevant to a specific problem.

        4-step pipeline:
          Step 1: Domain pre-filter
          Step 2: Score candidates (LLM or keyword)
          Step 3: Edge walk from seeds
          Step 4: Enrich with Mathlib names + edge reasoning

        Args:
            problem: Dict with 'statement'/'formal_statement' and optionally 'domain'
            kg: MathKG instance
            scoring_method: "llm" or "keyword" (defaults to config)
            num_seeds: K seeds to select (defaults to config)
            hop_depth: Edge traversal depth (defaults to config)

        Returns:
            List of {theorem_id, name, statement, relevance_score, rationale,
                     lean4_mathlib_name, lean4_mathlib_import, edge_connections}
        """
        if scoring_method is None:
            scoring_method = getattr(config, 'KG_SEARCH_SCORING_METHOD', 'llm')
        if num_seeds is None:
            num_seeds = getattr(config, 'KG_SEARCH_NUM_SEEDS', 5)
        if hop_depth is None:
            hop_depth = getattr(config, 'KG_SEARCH_HOP_DEPTH', 3)

        graph = kg.graph
        problem_text = problem.get("statement", "") or problem.get("formal_statement", "")
        problem_domain = problem.get("domain", "").lower()

        # ── Step 1: Domain pre-filter ──
        candidates = self._step1_domain_filter(graph, problem_domain)
        self.log(f"Step 1 (domain filter): {len(candidates)} candidates")

        if not candidates:
            self.log("No candidates after domain filter — using all nodes")
            candidates = list(graph.nodes)

        # ── Step 2: Score candidates and select K seeds ──
        if scoring_method == "llm":
            seeds = self._step2_llm_scoring(graph, candidates, problem_text, problem_domain, num_seeds)
        else:
            seeds = self._step2_keyword_scoring(graph, candidates, problem_text, num_seeds)
        self.log(f"Step 2 ({scoring_method} scoring): {len(seeds)} seeds selected")

        if not seeds:
            self.log("No seeds found — returning empty")
            return []

        # ── Step 3: Edge walk from seeds ──
        expanded = self._step3_edge_walk(graph, seeds, hop_depth)
        self.log(f"Step 3 (edge walk, {hop_depth} hops): {len(expanded)} theorems with connections")

        # ── Step 4: Enrich with Mathlib names + edge reasoning ──
        result = self._step4_enrich_context(graph, seeds, expanded)
        self.log(f"Step 4 (enrich): {len(result)} theorems ready for Prover")

        return result

    def _step1_domain_filter(self, graph, problem_domain: str) -> List[str]:
        """
        Step 1: Filter to same domain + cross-domain-bridge neighbors.
        Returns list of node IDs.
        """
        if not problem_domain:
            return list(graph.nodes)

        # Primary: nodes in the same domain
        primary = [
            nid for nid in graph.nodes
            if problem_domain in graph.nodes[nid].get("domain", "").lower()
        ]

        # Secondary: nodes connected to primary domain via cross-domain-bridge
        secondary = set()
        for nid in primary:
            for _, neighbor, data in graph.edges(nid, data=True):
                edge_type = data.get("edge_type", data.get("type", ""))
                if edge_type == "cross-domain-bridge":
                    secondary.add(neighbor)
            for neighbor, _, data in graph.in_edges(nid, data=True):
                edge_type = data.get("edge_type", data.get("type", ""))
                if edge_type == "cross-domain-bridge":
                    secondary.add(neighbor)

        # Combine and deduplicate
        all_candidates = list(set(primary) | secondary)
        return all_candidates

    def _step2_llm_scoring(self, graph, candidates: List[str],
                            problem_text: str, problem_domain: str,
                            num_seeds: int) -> List[Dict]:
        """
        Step 2 (LLM): Ask LLM to score candidates and select top K seeds.
        Sends full metadata (no truncation) for each candidate.
        """
        # Build detailed candidate list
        candidate_lines = []
        for nid in candidates:
            node = dict(graph.nodes[nid])
            name = node.get("name", nid)
            statement = node.get("statement", "")
            keywords = ", ".join(node.get("keywords", []))
            mathlib_name = node.get("lean4_mathlib_name", "")
            candidate_lines.append(
                f"- [{nid}] {name}\n"
                f"  Statement: {statement}\n"
                f"  Keywords: {keywords}\n"
                f"  Mathlib: {mathlib_name}"
            )

        candidate_text = "\n".join(candidate_lines)

        prompt = f"""Given this mathematical problem, score and select the {num_seeds} most relevant theorems from the candidates below.

PROBLEM:
{problem_text}

{f"DOMAIN: {problem_domain}" if problem_domain else ""}

CANDIDATE THEOREMS ({len(candidates)} total):
{candidate_text}

Select exactly {num_seeds} theorems most relevant for solving this problem.
For each, explain WHY it's relevant (proof technique, key lemma, analogous structure, etc.).

Respond in JSON:
{{
  "seeds": [
    {{
      "theorem_id": "...",
      "name": "...",
      "relevance_score": 0.0-1.0,
      "rationale": "why this theorem is relevant"
    }}
  ]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result and "seeds" in result:
            return result["seeds"][:num_seeds]
        return []

    def _step2_keyword_scoring(self, graph, candidates: List[str],
                                problem_text: str, num_seeds: int) -> List[Dict]:
        """
        Step 2 (keyword): Algorithmic scoring — no LLM call.
        Scores each candidate by word overlap with problem text.
        """
        import re

        # Extract problem words (lowercase alphabetic, length > 2, stopwords removed)
        stopwords = {"the", "and", "for", "that", "this", "with", "are", "all",
                     "any", "has", "have", "from", "let", "show", "prove", "find",
                     "such", "there", "every", "some", "each", "then", "when",
                     "where", "which", "than", "not", "can", "does", "its"}
        problem_words = set(
            w for w in re.findall(r'[a-z]+', problem_text.lower())
            if len(w) > 2 and w not in stopwords
        )

        scored = []
        for nid in candidates:
            node = dict(graph.nodes[nid])
            # Combine all text from the node
            node_text = " ".join([
                node.get("name", ""),
                node.get("statement", ""),
                " ".join(node.get("keywords", [])),
            ]).lower()
            node_words = set(re.findall(r'[a-z]+', node_text))

            overlap = problem_words & node_words
            score = len(overlap) / max(len(problem_words), 1)

            scored.append({
                "theorem_id": nid,
                "name": node.get("name", nid),
                "relevance_score": round(score, 3),
                "rationale": f"Keyword overlap: {', '.join(sorted(overlap)[:5])}" if overlap else "Domain match only",
            })

        # Sort by score descending, then by node degree as tiebreaker
        scored.sort(key=lambda x: (-x["relevance_score"], -graph.degree(x["theorem_id"])))
        return scored[:num_seeds]

    def _step3_edge_walk(self, graph, seeds: List[Dict],
                          hop_depth: int) -> Dict[str, List[Dict]]:
        """
        Step 3: BFS edge walk from seeds up to hop_depth.
        Returns dict: {node_id: [{from_node, from_name, edge_type, confidence, reasoning}]};
        from_node is the node the walk came from (for an in-edge this is the edge's target, not its source)
        """
        seed_ids = {s.get("theorem_id", s.get("id", "")) for s in seeds}

        # BFS from all seeds simultaneously
        visited = {}  # node_id → list of edge connections that led here
        queue = deque()

        # Initialize with seeds at depth 0
        for sid in seed_ids:
            visited[sid] = []  # seeds have no incoming edge
            queue.append((sid, 0))

        while queue:
            current, depth = queue.popleft()
            if depth >= hop_depth:
                continue

            # Walk outgoing edges
            for _, neighbor, data in graph.edges(current, data=True):
                edge_info = {
                    "from_node": current,
                    "from_name": graph.nodes[current].get("name", current),
                    "edge_type": data.get("edge_type", data.get("type", "unknown")),
                    "confidence": data.get("confidence", 0),
                    "reasoning": data.get("reasoning", ""),
                }
                if neighbor not in visited:
                    visited[neighbor] = [edge_info]
                    queue.append((neighbor, depth + 1))
                else:
                    # Already visited — add this as an additional connection
                    visited[neighbor].append(edge_info)

            # Walk incoming edges (relationships are directional)
            for neighbor, _, data in graph.in_edges(current, data=True):
                edge_info = {
                    "from_node": current,
                    "from_name": graph.nodes[current].get("name", current),
                    "edge_type": data.get("edge_type", data.get("type", "unknown")),
                    "confidence": data.get("confidence", 0),
                    "reasoning": data.get("reasoning", ""),
                }
                if neighbor not in visited:
                    visited[neighbor] = [edge_info]
                    queue.append((neighbor, depth + 1))
                else:
                    visited[neighbor].append(edge_info)

        return visited

    def _step4_enrich_context(self, graph, seeds: List[Dict],
                               expanded: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Step 4: Enrich results with Mathlib names + edge reasoning.
        Seeds come first, then edge-connected neighbors.
        """
        result = []
        seed_ids = {s.get("theorem_id", s.get("id", "")) for s in seeds}

        # Add seeds first (highest priority)
        for seed in seeds:
            tid = seed.get("theorem_id", seed.get("id", ""))
            node = dict(graph.nodes.get(tid, {}))
            result.append({
                "theorem_id": tid,
                "name": node.get("name", tid),
                "statement": node.get("statement", ""),
                "domain": node.get("domain", ""),
                "relevance_score": seed.get("relevance_score", 1.0),
                "rationale": seed.get("rationale", "Selected as seed"),
                "lean4_mathlib_name": node.get("lean4_mathlib_name", ""),
                "lean4_mathlib_import": node.get("lean4_mathlib_import", ""),
                "is_seed": True,
                "edge_connections": [],
            })

        # Add edge-connected neighbors
        max_candidates = getattr(config, 'KG_SEARCH_MAX_CANDIDATES', None)
        for nid, connections in expanded.items():
            if nid in seed_ids:
                continue  # already added as seed

            node = dict(graph.nodes.get(nid, {}))
            # Format edge connections for context
            edge_context = []
            for conn in connections:
                edge_context.append({
                    "from": conn["from_name"],
                    "type": conn["edge_type"],
                    "confidence": conn["confidence"],
                    "reasoning": conn["reasoning"],
                })

            result.append({
                "theorem_id": nid,
                "name": node.get("name", nid),
                "statement": node.get("statement", ""),
                "domain": node.get("domain", ""),
                "relevance_score": max(c["confidence"] for c in connections) if connections else 0,
                "rationale": f"Connected via: {connections[0]['edge_type']} from {connections[0]['from_name']}" if connections else "",
                "lean4_mathlib_name": node.get("lean4_mathlib_name", ""),
                "lean4_mathlib_import": node.get("lean4_mathlib_import", ""),
                "is_seed": False,
                "edge_connections": edge_context,
            })

        # Sort non-seeds by relevance score
        seeds_list = [r for r in result if r["is_seed"]]
        neighbors_list = sorted(
            [r for r in result if not r["is_seed"]],
            key=lambda x: -x["relevance_score"]
        )

        final = seeds_list + neighbors_list

        # Apply max candidates limit if set
        if max_candidates and len(final) > max_candidates:
            final = final[:max_candidates]

        return final

    def suggest_problem_approaches(self, problem: Dict, kg, relevant_theorems: List[Dict] = None) -> List[Dict]:
        """
        Suggest multiple ranked approaches for solving a specific problem.

        Args:
            problem: Dict with problem statement
            kg: MathKG instance
            relevant_theorems: Optional pre-computed relevant theorems

        Returns:
            List of {approach, strategy, theorems_to_use, confidence, rationale}
        """
        problem_text = problem.get("statement", "") or problem.get("formal_statement", "")

        # Build context from relevant theorems
        if relevant_theorems:
            context = "\n".join(
                f"- {t.get('name', t.get('theorem_id', ''))}: {t.get('rationale', '')}"
                for t in relevant_theorems[:10]
            )
        else:
            context = "No specific theorems pre-selected."

        prompt = f"""Suggest multiple approaches for solving this mathematical problem.

PROBLEM:
{problem_text[:500]}

RELEVANT THEOREMS FROM KNOWLEDGE GRAPH:
{context}

Suggest 3-5 different proof strategies, ranked by likelihood of success.
For each approach, specify:
1. The strategy name (direct proof, contradiction, induction, computational, etc.)
2. Which theorems to use
3. Step-by-step outline
4. Confidence level

Respond in JSON:
{{
  "approaches": [
    {{
      "approach": "strategy name",
      "strategy": "detailed description",
      "theorems_to_use": ["theorem_id_1", "theorem_id_2"],
      "confidence": 0.0-1.0,
      "rationale": "why this approach might work"
    }}
  ]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=8192)
        if result and "approaches" in result:
            approaches = result["approaches"]
            approaches.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            self.log(f"Suggested {len(approaches)} approaches")
            return approaches
        return []

    # ═══════════════════════════════════════════════════════════════════
    # LAYER 2: MATHLIB DECLARATION SEARCH
    #
    # Pipeline:
    #   Step 0: Cache declaration database
    #   Step 1: Extract search terms (LLM + formal statement identifiers)
    #   Step 2: Prefix match against the cached Mathlib declaration database (~404K declarations)
    #   Step 3: Score and rank (source, precision, depth, kind, module)
    #   Step 4: Enrich with type signatures + docstrings, then return
    #
    # In full_system mode, also uses KG seed modules for boosting.
    # ═══════════════════════════════════════════════════════════════════

    _mathlib_cache = None  # Class-level cache for declaration database (name → {kind, docLink})
    _mathlib_types_cache = None  # Class-level cache for type signatures (name → {type, doc, kind})

    # Domain → Mathlib module prefix mapping for module proximity scoring
    DOMAIN_MODULE_MAP = {
        "number theory": ["Mathlib.NumberTheory", "Mathlib.Data.Nat", "Mathlib.Data.ZMod", "Mathlib.Data.Int"],
        "combinatorics": ["Mathlib.Combinatorics", "Mathlib.Data.Finset"],
        "algebra": ["Mathlib.Algebra", "Mathlib.Data.Polynomial", "Mathlib.Data.Complex", "Mathlib.Data.Real"],
        "analysis": ["Mathlib.Analysis", "Mathlib.MeasureTheory", "Mathlib.Topology"],
        "geometry": ["Mathlib.Geometry", "Mathlib.Analysis.InnerProductSpace"],
        "linear algebra": ["Mathlib.LinearAlgebra", "Mathlib.Data.Matrix"],
        "abstract algebra": ["Mathlib.GroupTheory", "Mathlib.RingTheory", "Mathlib.FieldTheory"],
    }

    def _get_mathlib_decls(self, mathlib_data_path: str = None) -> dict:
        """Step 0: Load and cache Mathlib declaration database."""
        if ExplorerAgent._mathlib_cache is not None:
            return ExplorerAgent._mathlib_cache

        import json as _json
        from pathlib import Path as _Path
        if mathlib_data_path is None:
            # Look in project data directory first, fall back to /tmp
            project_path = _Path(__file__).resolve().parent.parent.parent / "data" / "mathlib_declarations" / "declaration-data.bmp"
            if project_path.exists():
                mathlib_data_path = str(project_path)
            else:
                mathlib_data_path = "/tmp/mathlib_decl_data.bmp"

        try:
            with open(mathlib_data_path) as f:
                data = _json.load(f)
            ExplorerAgent._mathlib_cache = data["declarations"]
            self.log(f"Cached {len(ExplorerAgent._mathlib_cache)} Mathlib declarations")
            return ExplorerAgent._mathlib_cache
        except (FileNotFoundError, _json.JSONDecodeError):
            self.log(f"Mathlib declaration data not found at {mathlib_data_path}")
            return {}

    def _get_mathlib_types(self) -> dict:
        """Load and cache the rich declaration database with type signatures + docstrings."""
        if ExplorerAgent._mathlib_types_cache is not None:
            return ExplorerAgent._mathlib_types_cache

        import json as _json
        from pathlib import Path

        # Look in project data directory
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "data" / "mathlib_declarations" / "declarations_with_types.json",
            Path("data/mathlib_declarations/declarations_with_types.json"),
        ]

        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        ExplorerAgent._mathlib_types_cache = _json.load(f)
                    self.log(f"Cached {len(ExplorerAgent._mathlib_types_cache)} Mathlib type signatures")
                    return ExplorerAgent._mathlib_types_cache
                except (FileNotFoundError, _json.JSONDecodeError):
                    continue

        self.log("Mathlib type signatures not found (declarations_with_types.json)")
        ExplorerAgent._mathlib_types_cache = {}
        return {}

    def _step1_extract_search_terms(self, problem: Dict) -> List[Dict]:
        """
        Step 1: Extract search terms from LLM (1a) + formal statement (1b).
        Returns list of {term, source, priority}.
        """
        import re
        terms = {}  # term → {source, priority}

        # ── Step 1b: Extract Lean identifiers from formal statement (free) ──
        formal = problem.get("formal_statement", "") or problem.get("lean4_sketch", "")
        if formal:
            # Match capitalized identifiers like Nat.Prime, ZMod, Finset, Function.Injective
            lean_ids = re.findall(r'\b([A-Z][a-zA-Z]*(?:\.[A-Z][a-zA-Z]*)*)', formal)
            for lid in lean_ids:
                if lid not in ("Type", "Prop", "Sort", "True", "False", "Set", "Or", "And", "Not", "Iff", "Eq"):
                    terms[lid] = {"source": "formal_statement", "priority": "essential"}

        # ── Step 1a: LLM suggests Mathlib search terms ──
        problem_text = problem.get("statement", "") or problem.get("informal_stmt", "")
        formal_text = problem.get("formal_statement", "") or ""

        prompt = f"""Given this mathematical problem, suggest 10-15 Lean 4 Mathlib namespaces or identifier prefixes that would be useful to search for when proving it.

PROBLEM (natural language):
{problem_text}

PROBLEM (formal Lean 4):
{formal_text[:500]}

For each suggestion, classify its importance:
- "essential": directly needed for the proof
- "helpful": likely useful as a supporting lemma
- "speculative": might be relevant, worth checking

Focus on specific Mathlib namespaces and declaration families, e.g.:
- "Nat.Prime" (not just "prime")
- "Function.Injective" (not just "injective")
- "Finset.card" (not just "card")

Respond in JSON:
{{
  "search_terms": [
    {{"term": "Nat.Prime", "priority": "essential"}},
    {{"term": "Function.Injective", "priority": "helpful"}},
    {{"term": "Nat.strongRecOn", "priority": "speculative"}}
  ]
}}"""

        result = self.call_llm_json(prompt, self.SYSTEM_PROMPT, max_tokens=4096)
        if result and "search_terms" in result:
            for item in result["search_terms"]:
                term = item.get("term", "")
                priority = item.get("priority", "speculative")
                if term and term not in terms:
                    terms[term] = {"source": "llm", "priority": priority}

        term_list = [{"term": t, **info} for t, info in terms.items()]
        self.log(f"Step 1: {len(term_list)} search terms ({sum(1 for t in term_list if t['source']=='formal_statement')} from formal, {sum(1 for t in term_list if t['source']=='llm')} from LLM)")
        return term_list

    def _step2_prefix_match(self, search_terms: List[Dict], decls: dict) -> Dict[str, List[Dict]]:
        """
        Step 2: Prefix match each search term against the cached Mathlib declaration database (~404K declarations).
        Returns {term: [list of matching declarations]}.
        """
        results = {}
        for term_info in search_terms:
            term = term_info["term"]
            matches = []
            for name, info in decls.items():
                if info.get("kind") not in ("theorem", "def"):
                    continue
                # Strict prefix: name == term OR name starts with term.
                # e.g., "Nat.Prime" matches "Nat.Prime" and "Nat.Prime.dvd_mul"
                # but NOT "Nat.primeCounting" or "Nat.primeFactorsList"
                is_exact = (name == term)
                is_prefix = name.startswith(term + ".")
                if is_exact or is_prefix:
                    module = info.get("docLink", "").split("#")[0].replace("./", "").replace(".html", "").replace("/", ".")
                    matches.append({
                        "name": name,
                        "kind": info["kind"],
                        "module": module,
                        "is_exact_match": is_exact,
                    })
            results[term] = matches
        return results

    def _step3_score_and_rank(self, search_terms: List[Dict],
                               raw_matches: Dict[str, List[Dict]],
                               problem_domain: str = "",
                               kg_seed_modules: List[str] = None,
                               kg_result_names: Set[str] = None) -> List[Dict]:
        """
        Step 3: Score and rank matched declarations.

        Scoring signals:
          - Source priority: formal_statement > llm essential > helpful > speculative
          - Match precision: exact > prefix
          - Name depth: fewer dots = more fundamental
          - Kind: theorem > def
          - Module proximity: matches problem domain → boost
          - KG co-occurrence: same module as KG seed → boost (full_system only)
          - KG dedup: skip if already in KG results (full_system only)
        """
        # Priority weights
        PRIORITY_SCORES = {
            "essential": 0.9,
            "helpful": 0.7,
            "speculative": 0.5,
        }

        # Caps per priority level
        CAPS_PER_TERM = {
            "essential": 8,
            "helpful": 5,
            "speculative": 2,
        }

        # Domain module prefixes for proximity scoring
        domain_modules = []
        if problem_domain:
            for domain_key, prefixes in self.DOMAIN_MODULE_MAP.items():
                if domain_key in problem_domain.lower():
                    domain_modules = prefixes
                    break

        scored_all = []
        seen_names = set()

        for term_info in search_terms:
            term = term_info["term"]
            source = term_info["source"]
            priority = term_info["priority"]
            matches = raw_matches.get(term, [])

            # Score each match
            scored_term = []
            for m in matches:
                name = m["name"]

                # KG dedup: skip if already in Layer 1 results
                if kg_result_names and name in kg_result_names:
                    continue

                # Skip if already seen from a higher-priority term
                if name in seen_names:
                    continue

                # Source score
                if source == "formal_statement":
                    source_score = 1.0
                else:
                    source_score = PRIORITY_SCORES.get(priority, 0.5)

                # Precision score
                precision_score = 1.0 if m["is_exact_match"] else 0.8

                # Depth score (fewer dots = more fundamental)
                dot_count = name.count(".")
                depth_score = 1.0 / max(dot_count, 1)

                # Kind score
                kind_score = 1.0 if m["kind"] == "theorem" else 0.7

                # Module proximity score
                module_score = 1.0
                if domain_modules and m.get("module"):
                    for prefix in domain_modules:
                        if m["module"].startswith(prefix):
                            module_score = 1.2
                            break

                # KG seed co-occurrence boost (full_system only)
                kg_boost = 1.0
                if kg_seed_modules and m.get("module"):
                    for seed_module in kg_seed_modules:
                        if m["module"] == seed_module:
                            kg_boost = 1.5
                            break

                # Combined score
                score = source_score * precision_score * depth_score * kind_score * module_score * kg_boost

                scored_term.append({
                    "name": name,
                    "kind": m["kind"],
                    "module": m.get("module", ""),
                    "score": round(score, 4),
                    "search_term": term,
                    "priority": priority,
                    "source": source,
                })

            # Sort and cap per term
            scored_term.sort(key=lambda x: -x["score"])
            cap = CAPS_PER_TERM.get(priority, 5)
            for item in scored_term[:cap]:
                if item["name"] not in seen_names:
                    seen_names.add(item["name"])
                    scored_all.append(item)

        # Final sort by score
        scored_all.sort(key=lambda x: -x["score"])
        return scored_all

    def search_mathlib_declarations(self, problem: Dict,
                                     kg_results: List[Dict] = None,
                                     mathlib_data_path: str = None) -> List[Dict]:
        """
        Layer 2: Search Mathlib declaration database.

        Works in two modes:
          - with_mathlib: standalone (kg_results=None)
          - full_system: KG-aware (kg_results from Layer 1)

        Args:
            problem: Dict with problem statement
            kg_results: Optional Layer 1 results for KG-aware scoring
            mathlib_data_path: Path to declaration data

        Returns:
            List of {name, kind, module, score, search_term, priority, source}
        """
        # Step 0: Load cached declarations
        decls = self._get_mathlib_decls(mathlib_data_path)
        if not decls:
            return []

        # Step 1: Extract search terms
        search_terms = self._step1_extract_search_terms(problem)
        if not search_terms:
            self.log("No search terms extracted")
            return []

        # Step 2: Prefix match
        raw_matches = self._step2_prefix_match(search_terms, decls)
        total_raw = sum(len(v) for v in raw_matches.values())
        self.log(f"Step 2: {total_raw} raw matches across {len(search_terms)} terms")

        # Prepare KG-aware signals (full_system mode)
        kg_seed_modules = None
        kg_result_names = None
        if kg_results:
            kg_seed_modules = [
                r.get("lean4_mathlib_import", "")
                for r in kg_results if r.get("is_seed") and r.get("lean4_mathlib_import")
            ]
            kg_result_names = {
                r.get("lean4_mathlib_name", "")
                for r in kg_results if r.get("lean4_mathlib_name")
            }

        # Step 3: Score and rank
        problem_domain = problem.get("domain", "")
        results = self._step3_score_and_rank(
            search_terms, raw_matches, problem_domain,
            kg_seed_modules=kg_seed_modules,
            kg_result_names=kg_result_names,
        )
        self.log(f"Step 3: {len(results)} declarations after scoring and ranking")

        # Step 4: Enrich with type signatures and docstrings (if available)
        types_db = self._get_mathlib_types()
        if types_db:
            for r in results:
                entry = types_db.get(r["name"], {})
                r["type_signature"] = entry.get("type", "")
                r["doc"] = entry.get("doc", "")

        return results
