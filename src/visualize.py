"""
Visualization for the Mathematical Knowledge Graph.
Creates an interactive HTML visualization using pyvis.
"""

from pyvis.network import Network
from build_graph import MathKG
from pathlib import Path


def visualize_graph(kg: MathKG, output_path: Path):
    """
    Create an interactive HTML visualization of the knowledge graph.

    Args:
        kg: MathKG instance
        output_path: Path to save HTML file
    """
    # Create pyvis network
    net = Network(
        height="950px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True
    )

    # Set physics options optimized for dense graphs (many edges)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -50000,
          "centralGravity": 0.1,
          "springLength": 500,
          "springConstant": 0.02,
          "damping": 0.15,
          "avoidOverlap": 1.0
        },
        "maxVelocity": 30,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 25,
          "fit": true
        }
      },
      "nodes": {
        "font": {
          "size": 20,
          "face": "Arial",
          "bold": true
        },
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "roundness": 0.3
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.8
          }
        }
      }
    }
    """)

    # Define color scheme by domain (all 7 active domains)
    domain_colors = {
        "Number Theory": "#FF9800",       # Orange
        "Combinatorics": "#E91E63",       # Pink
        "Algebra": "#4CAF50",             # Green
        "Analysis": "#9C27B0",            # Purple
        "Geometry": "#F44336",            # Red
        "Linear Algebra": "#2196F3",      # Blue
        "Abstract Algebra": "#00BCD4",    # Teal
    }

    # Define shapes by theorem type (agent-generated conjectures use triangle)
    type_shapes = {
        "theorem": "dot",
        "definition": "diamond",
        "lemma": "dot",
        "conjecture": "triangle",
    }

    # Calculate node sizes based on degree (number of connections)
    degrees = dict(kg.graph.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # Add nodes
    for node_id, data in kg.graph.nodes(data=True):
        domain = data.get('domain', 'Unknown')
        node_type = data.get('type', 'theorem')

        # Size based on number of connections — capped to stay readable
        degree = degrees.get(node_id, 0)
        node_size = 15 + min(degree * 2, 30)  # Base 15, max 45px

        # Shorter labels for cleaner look
        short_label = data['name'][:35] + '...' if len(data['name']) > 35 else data['name']

        lean4_status = data.get("lean4_status", "")
        lean4_badge = f"\nLean 4: {lean4_status}" if lean4_status else ""
        agent_badge = "\n[Agent-generated]" if data.get("agent_generated") else ""

        title = (
            f"{data['name']}\n"
            f"Domain: {domain}\n"
            f"Type: {node_type}  |  Connections: {degree}"
            f"{lean4_badge}{agent_badge}\n\n"
            f"{data.get('statement', '')}"
        )

        net.add_node(
            node_id,
            label=short_label,
            title=title,
            color=domain_colors.get(domain, "#808080"),
            shape=type_shapes.get(node_type, "dot"),
            size=node_size,
            borderWidth=2
        )

    # Define edge colors by type (all 8 edge types in current ontology)
    edge_colors = {
        "prerequisite": "#FFB74D",
        "generalizes": "#FF6B6B",
        "specializes": "#4ECDC4",
        "equivalent-to": "#7E57C2",
        "applied-in": "#F38181",
        "analogous-to": "#95E1D3",
        "cross-domain-bridge": "#FCBAD3",
        "instance-of": "#AA96DA",
    }

    # Add all edges — JavaScript slider will handle filtering by confidence
    all_edges = list(kg.graph.edges(data=True))
    for u, v, data in all_edges:
        edge_type = data.get('edge_type', 'unknown')
        confidence = data.get('confidence', 0.0)
        reasoning = data.get('reasoning', '')

        title = (
            f"Type: {edge_type}\n"
            f"Confidence: {confidence:.2f}\n\n"
            f"Reasoning: {reasoning}"
        )

        net.add_edge(
            u, v,
            title=title,
            color={'color': edge_colors.get(edge_type, "#808080"), 'opacity': 0.85},
            width=1 + confidence * 3,
            smooth={'type': 'curvedCW', 'roundness': 0.2}
        )

    print(f"  Added {len(all_edges)} edges (confidence slider will filter in browser)")

    # Add legend
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white;
                padding: 20px; border: 2px solid #333; border-radius: 8px;
                font-family: Arial; z-index: 1000; box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                max-width: 280px;">
        <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 18px;
                   border-bottom: 2px solid #333; padding-bottom: 8px; color: #333;">
            MathKG Legend
        </h3>
        <h4 style="margin: 12px 0 8px 0; font-size: 14px; color: #555; font-weight: 600;">
            Domains (Node Colors)
        </h4>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #FF9800; font-size: 20px; font-weight: bold;">&#9679;</span> Number Theory
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #E91E63; font-size: 20px; font-weight: bold;">&#9679;</span> Combinatorics
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #4CAF50; font-size: 20px; font-weight: bold;">&#9679;</span> Algebra
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #9C27B0; font-size: 20px; font-weight: bold;">&#9679;</span> Analysis
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #F44336; font-size: 20px; font-weight: bold;">&#9679;</span> Geometry
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #2196F3; font-size: 20px; font-weight: bold;">&#9679;</span> Linear Algebra
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #00BCD4; font-size: 20px; font-weight: bold;">&#9679;</span> Abstract Algebra
        </div>
        <div style="margin: 5px 0; font-size: 13px;">
            <span style="color: #808080; font-size: 20px; font-weight: bold;">&#9650;</span> Agent-Generated (triangle; coloured by domain)
        </div>
        <h4 style="margin: 12px 0 8px 0; font-size: 14px; color: #555; font-weight: 600;">
            Node Types
        </h4>
        <div style="margin: 5px 0; font-size: 12px;">&#9679; Circle = theorem / lemma</div>
        <div style="margin: 5px 0; font-size: 12px;">&#9670; Diamond = definition</div>
        <div style="margin: 5px 0; font-size: 12px;">&#9650; Triangle = agent conjecture</div>
        <h4 style="margin: 15px 0 8px 0; font-size: 14px; color: #555; font-weight: 600;">
            Semantic Edges (8 types)
        </h4>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #FFB74D; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> prerequisite
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #FF6B6B; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> generalizes
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #4ECDC4; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> specializes
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #7E57C2; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> equivalent-to
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #F38181; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> applied-in
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #95E1D3; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> analogous-to
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #FCBAD3; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> cross-domain-bridge
        </div>
        <div style="margin: 5px 0; font-size: 12px;">
            <span style="color: #AA96DA; font-weight: bold; font-size: 16px;">&#9473;&#9473;</span> instance-of
        </div>
        <p style="margin-top: 15px; font-size: 11px; color: #666; line-height: 1.5;
                  border-top: 1px solid #ddd; padding-top: 12px;">
            <strong>Tips:</strong><br>
            &#8226; Hover over nodes/edges for details<br>
            &#8226; Drag nodes to rearrange<br>
            &#8226; Node size = # of connections<br>
            &#8226; Edge thickness = confidence score<br>
            &#8226; Use the slider (top right) to filter edges
        </p>
    </div>
    """

    # Save the network
    net.save_graph(str(output_path))

    # Add legend and stabilization script to HTML
    with open(output_path, 'r') as f:
        html_content = f.read()

    # CSS to fix vis.js tooltip: wrap text and respect newlines
    tooltip_css = """
    <style>
        div.vis-tooltip {
            max-width: 300px !important;
            white-space: pre-line !important;
            line-height: 1.6 !important;
            padding: 10px 14px !important;
        }
    </style>
    """

    # Confidence slider UI + stabilization script
    stabilization_script = """
    <div id="conf-panel" style="position: absolute; top: 10px; right: 10px;
         background: white; padding: 16px 20px; border: 2px solid #333;
         border-radius: 8px; font-family: Arial; z-index: 1000;
         box-shadow: 0 4px 8px rgba(0,0,0,0.15); min-width: 220px;">
        <b style="font-size: 14px;">Confidence Filter</b>
        <div style="margin-top: 10px; font-size: 13px;">
            Min confidence: <b id="conf-label">0.85</b>
        </div>
        <input id="conf-slider" type="range" min="0" max="100" value="85" step="5"
               style="width: 100%; margin-top: 8px; cursor: pointer;">
        <div style="margin-top: 8px; font-size: 12px; color: #555;">
            Showing <b id="edge-count">—</b> edges
        </div>
    </div>

    <script type="text/javascript">
        // Disable physics after stabilization
        network.once('stabilizationIterationsDone', function() {
            network.setOptions({ physics: false });
        });

        // Confidence slider filtering
        var allEdgesData = edges.get();

        function applyConfidenceFilter(threshold) {
            var filtered = allEdgesData.filter(function(e) {
                var m = e.title.match(/Confidence:\\s*([\\d.]+)/);
                return m ? parseFloat(m[1]) >= threshold : false;
            });
            var filteredIds = filtered.map(function(e) { return e.id; });
            var currentIds = edges.getIds();

            var toRemove = currentIds.filter(function(id) { return filteredIds.indexOf(id) === -1; });
            var toAdd = filtered.filter(function(e) { return currentIds.indexOf(e.id) === -1; });
            edges.remove(toRemove);
            edges.add(toAdd);
            document.getElementById('edge-count').textContent = filtered.length;
        }

        var slider = document.getElementById('conf-slider');
        var label = document.getElementById('conf-label');

        // Apply default threshold on load
        applyConfidenceFilter(0.85);

        slider.addEventListener('input', function() {
            var val = parseInt(this.value) / 100;
            label.textContent = val.toFixed(2);
            applyConfidenceFilter(val);
        });
    </script>
    """

    # Insert tooltip CSS into head, legend and script before closing body tag
    html_content = html_content.replace('</head>', f'{tooltip_css}</head>')
    html_content = html_content.replace('</body>', f'{legend_html}{stabilization_script}</body>')

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Visualization saved to {output_path}")
    print(f"  Open the file in a web browser to view the interactive graph")


if __name__ == "__main__":
    from extract_mathlib import load_sample_theorems
    from build_graph import MathKG

    # Build test graph
    theorems = load_sample_theorems()
    kg = MathKG()
    kg.add_theorems(theorems)

    # Add test edges
    test_edges = [
        {
            'source': 'thm_001',
            'target': 'thm_004',
            'type': 'generalizes',
            'confidence': 0.85,
            'reasoning': 'Lagrange theorem generalizes to Sylow theorems'
        },
        {
            'source': 'thm_003',
            'target': 'thm_008',
            'type': 'analogous-to',
            'confidence': 0.92,
            'reasoning': 'Isomorphism theorems have parallel structure in groups and rings'
        },
        {
            'source': 'thm_012',
            'target': 'thm_013',
            'type': 'generalizes',
            'confidence': 0.90,
            'reasoning': "Euler's theorem generalizes Fermat's Little Theorem"
        }
    ]
    kg.add_semantic_edges(test_edges)

    # Create visualization
    output_path = Path(__file__).parent.parent / "outputs" / "graph_visualization.html"
    visualize_graph(kg, output_path)
