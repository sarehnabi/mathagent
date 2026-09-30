"""
Query interface for the Mathematical Knowledge Graph.
Demonstrates the value of semantic edges for mathematical discovery.
"""

from typing import List, Dict, Set
import networkx as nx
from build_graph import MathKG


class MathKGQuerier:
    """Query engine for semantic graph traversal."""

    def __init__(self, kg: MathKG):
        """Initialize with a knowledge graph."""
        self.kg = kg
        self.graph = kg.graph

    def find_analogies(self, domain_a: str, domain_b: str) -> List[Dict]:
        """
        Find analogous theorems between two domains.

        Args:
            domain_a: First domain (e.g., "Abstract Algebra")
            domain_b: Second domain (e.g., "Number Theory")

        Returns:
            List of analogous theorem pairs
        """
        analogies = []

        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'analogous-to':
                node_u = self.graph.nodes[u]
                node_v = self.graph.nodes[v]

                if (node_u['domain'] == domain_a and node_v['domain'] == domain_b) or \
                   (node_u['domain'] == domain_b and node_v['domain'] == domain_a):
                    analogies.append({
                        'theorem_1': {
                            'id': u,
                            'name': node_u['name'],
                            'domain': node_u['domain']
                        },
                        'theorem_2': {
                            'id': v,
                            'name': node_v['name'],
                            'domain': node_v['domain']
                        },
                        'confidence': data['confidence'],
                        'reasoning': data['reasoning']
                    })

        return analogies

    def find_generalizations(self, theorem_id: str) -> List[Dict]:
        """
        Find all theorems that generalize a given theorem.

        Args:
            theorem_id: ID of the theorem

        Returns:
            List of generalizing theorems
        """
        generalizations = []

        # Look for edges where this theorem is the target of a 'generalizes' edge
        for u, v, data in self.graph.edges(data=True):
            if v == theorem_id and data.get('edge_type') == 'generalizes':
                node_u = self.graph.nodes[u]
                generalizations.append({
                    'id': u,
                    'name': node_u['name'],
                    'domain': node_u['domain'],
                    'confidence': data['confidence'],
                    'reasoning': data['reasoning']
                })

        return generalizations

    def find_cross_domain_connections(self, theorem_id: str) -> List[Dict]:
        """
        Find theorems in other domains connected to this theorem.

        Args:
            theorem_id: ID of the theorem

        Returns:
            List of cross-domain connections
        """
        source_domain = self.graph.nodes[theorem_id]['domain']
        connections = []

        # Find all neighbors (in and out)
        neighbors = set(self.graph.predecessors(theorem_id)) | \
                   set(self.graph.successors(theorem_id))

        for neighbor in neighbors:
            neighbor_domain = self.graph.nodes[neighbor]['domain']

            if neighbor_domain != source_domain:
                # Get edge data (could be in either direction)
                if self.graph.has_edge(theorem_id, neighbor):
                    edge_data = self.graph[theorem_id][neighbor]
                else:
                    edge_data = self.graph[neighbor][theorem_id]

                connections.append({
                    'id': neighbor,
                    'name': self.graph.nodes[neighbor]['name'],
                    'domain': neighbor_domain,
                    'edge_type': edge_data.get('edge_type'),
                    'confidence': edge_data.get('confidence'),
                    'reasoning': edge_data.get('reasoning')
                })

        return connections

    def find_theorems_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Find all theorems containing a specific keyword.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching theorems
        """
        matches = []

        for node_id, data in self.graph.nodes(data=True):
            if keyword.lower() in [k.lower() for k in data.get('keywords', [])]:
                matches.append({
                    'id': node_id,
                    'name': data['name'],
                    'domain': data['domain'],
                    'statement': data['statement']
                })

        return matches

    def get_neighborhood(self, theorem_id: str, radius: int = 1) -> Set[str]:
        """
        Get all theorems within radius steps of a given theorem.

        Args:
            theorem_id: ID of the center theorem
            radius: Number of steps to traverse

        Returns:
            Set of theorem IDs in the neighborhood
        """
        # Convert to undirected for neighborhood search
        undirected = self.graph.to_undirected()

        neighborhood = set()
        current_level = {theorem_id}

        for _ in range(radius):
            next_level = set()
            for node in current_level:
                next_level.update(undirected.neighbors(node))
            neighborhood.update(next_level)
            current_level = next_level

        return neighborhood


def run_demo_queries(kg: MathKG):
    """Run demonstration queries on the knowledge graph."""
    querier = MathKGQuerier(kg)

    print("\n" + "="*60)
    print("DEMONSTRATION QUERIES")
    print("="*60)

    # Query 1: Find analogies between Abstract Algebra and Number Theory
    print("\n📊 Query 1: What abstract algebra theorems have analogues in number theory?")
    print("-" * 60)
    analogies = querier.find_analogies("Abstract Algebra", "Number Theory")

    if analogies:
        for analogy in analogies:
            print(f"\n  {analogy['theorem_1']['name']} (Abstract Algebra)")
            print(f"  ↔ {analogy['theorem_2']['name']} (Number Theory)")
            print(f"  Confidence: {analogy['confidence']:.2f}")
            print(f"  Reasoning: {analogy['reasoning']}")
    else:
        print("  No analogies found in current graph.")

    # Query 2: Find cross-domain connections
    print("\n\n📊 Query 2: Which results connect algebra to number theory?")
    print("-" * 60)

    # Check all abstract algebra nodes for number theory connections
    for node_id, data in kg.graph.nodes(data=True):
        if data['domain'] == "Abstract Algebra":
            connections = querier.find_cross_domain_connections(node_id)
            nt_connections = [c for c in connections if c['domain'] == "Number Theory"]

            if nt_connections:
                print(f"\n  {data['name']} connects to:")
                for conn in nt_connections:
                    print(f"    → {conn['name']} ({conn['edge_type']})")

    # Query 3: Find theorems by keyword
    print("\n\n📊 Query 3: Find all theorems about 'prime' numbers")
    print("-" * 60)
    prime_theorems = querier.find_theorems_by_keyword("prime")

    for thm in prime_theorems[:5]:  # Show first 5
        print(f"\n  {thm['name']} ({thm['domain']})")
        print(f"  {thm['statement'][:80]}...")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    from extract_mathlib import load_sample_theorems
    from build_graph import MathKG

    # Build a test graph
    theorems = load_sample_theorems()
    kg = MathKG()
    kg.add_theorems(theorems)

    # Add test edges
    test_edges = [
        {
            'source': 'thm_003',
            'target': 'thm_008',
            'type': 'analogous-to',
            'confidence': 0.92,
            'reasoning': 'First Isomorphism Theorem has the same structure for groups and rings'
        },
        {
            'source': 'thm_005',
            'target': 'thm_012',
            'type': 'applied-in',
            'confidence': 0.85,
            'reasoning': "Cauchy's theorem about group elements is used in number theory"
        }
    ]
    kg.add_semantic_edges(test_edges)

    # Run demo queries
    run_demo_queries(kg)
