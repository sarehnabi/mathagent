"""
Build the Mathematical Knowledge Graph using NetworkX.
"""

import networkx as nx
from typing import List, Dict
import json
from pathlib import Path


class MathKG:
    """Mathematical Knowledge Graph with semantic edges."""

    def __init__(self):
        """Initialize empty directed graph."""
        self.graph = nx.DiGraph()

    def add_theorems(self, theorems: List[Dict]):
        """
        Add nodes (theorems and definitions) to the graph.

        Args:
            theorems: List of node dictionaries (type "theorem" or "definition")
        """
        for theorem in theorems:
            self.graph.add_node(
                theorem['id'],
                name=theorem['name'],
                domain=theorem['domain'],
                statement=theorem['statement'],
                type=theorem['type'],
                keywords=theorem['keywords']
            )
        print(f"✓ Added {len(theorems)} nodes to graph")

    def add_semantic_edges(self, edges: List[Dict]):
        """
        Add semantic relationship edges to the graph.

        Args:
            edges: List of edge dictionaries with source, target, type, etc.
        """
        for edge in edges:
            self.graph.add_edge(
                edge['source'],
                edge['target'],
                edge_type=edge['type'],
                confidence=edge['confidence'],
                reasoning=edge['reasoning']
            )
        print(f"✓ Added {len(edges)} semantic edges to graph")

    def get_statistics(self) -> Dict:
        """Compute graph statistics."""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'domains': self._count_domains(),
            'edge_types': self._count_edge_types(),
            'density': nx.density(self.graph)
        }
        return stats

    def _count_domains(self) -> Dict[str, int]:
        """Count nodes by domain."""
        domains = {}
        for node, data in self.graph.nodes(data=True):
            domain = data.get('domain', 'Unknown')
            domains[domain] = domains.get(domain, 0) + 1
        return domains

    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by type."""
        edge_types = {}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        return edge_types

    def save(self, filepath: Path):
        """Save graph to JSON."""
        # Convert graph to JSON-serializable format
        data = {
            'nodes': [
                {
                    'id': node,
                    **attrs
                }
                for node, attrs in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **attrs
                }
                for u, v, attrs in self.graph.edges(data=True)
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved graph to {filepath}")

    def load(self, filepath: Path):
        """Load graph from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Clear existing graph
        self.graph.clear()

        # Add nodes
        for node in data['nodes']:
            node_id = node.pop('id')
            self.graph.add_node(node_id, **node)

        # Add edges
        for edge in data['edges']:
            source = edge.pop('source')
            target = edge.pop('target')
            self.graph.add_edge(source, target, **edge)

        print(f"✓ Loaded graph from {filepath}")

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        print("\n" + "="*50)
        print("MATHEMATICAL KNOWLEDGE GRAPH STATISTICS")
        print("="*50)
        print(f"Nodes (thms + defns): {stats['num_nodes']}")
        print(f"Edges (relationships): {stats['num_edges']}")
        print(f"Average degree:        {stats['avg_degree']:.2f}")
        print(f"Graph density:         {stats['density']:.3f}")

        print(f"\nDomains:")
        for domain, count in sorted(stats['domains'].items()):
            print(f"  {domain}: {count}")

        print(f"\nSemantic Edge Types:")
        for edge_type, count in sorted(stats['edge_types'].items()):
            print(f"  {edge_type}: {count}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Test graph construction
    from extract_mathlib import load_sample_theorems
    from infer_edges import SemanticEdgeInferencer

    print("Building test graph...")

    # Load theorems
    theorems = load_sample_theorems()

    # Build graph
    kg = MathKG()
    kg.add_theorems(theorems)

    # Add some manual test edges
    test_edges = [
        {
            'source': 'thm_001',
            'target': 'thm_004',
            'type': 'specializes',
            'confidence': 0.85,
            'reasoning': 'Sylow theorems are specialized versions of group order divisibility'
        },
        {
            'source': 'thm_003',
            'target': 'thm_008',
            'type': 'analogous-to',
            'confidence': 0.90,
            'reasoning': 'Isomorphism theorems have same structure for groups and rings'
        }
    ]
    kg.add_semantic_edges(test_edges)

    # Print statistics
    kg.print_statistics()
