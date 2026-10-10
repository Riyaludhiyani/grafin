"""PyVis interactive network visualizer with style encoding for financial impact graphs.

Visual specifications:
- Direct/Observed shock links: Solid, prominent lines
- Curated prior links: Dashed lines
- Inferred multi-hop transmission links: Dotted lines
- Direction colors: Green (#10B981) for positive, Red (#EF4444) for negative, Amber (#F59E0B) for ambiguous
- Interactive nodes with rich hover tooltips and category shapes
"""

from typing import Dict, List, Optional, Set
from pyvis.network import Network

from src.graph_engine.loader import KnowledgeGraph
from src.graph_engine.propagator import PropagationResult


def generate_impact_graph_html(
    kg: KnowledgeGraph,
    propagation_result: Optional[PropagationResult] = None,
    height: str = "580px",
    width: str = "100%",
) -> str:
    """Generate an interactive PyVis HTML visualization of the knowledge graph and active shock paths.

    Args:
        kg: Validated KnowledgeGraph instance.
        propagation_result: Optional PropagationResult to highlight active shock transmission paths.
        height: CSS height string.
        width: CSS width string.

    Returns:
        Complete HTML string ready for Streamlit HTML component rendering.
    """
    net = Network(height=height, width=width, directed=True, bgcolor="#0E1117", font_color="#F8FAFC")

    # Configure physics for smooth, stable dark fintech layout
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "color": "#F8FAFC",
          "size": 13,
          "face": "Inter, sans-serif"
        }
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "none",
          "roundness": 0.25
        },
        "font": {
          "color": "#94A3B8",
          "size": 10,
          "align": "middle",
          "background": "#0E1117"
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.75 }
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -4500,
          "centralGravity": 0.35,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Active shock context
    origin_id = propagation_result.origin_id if propagation_result else None
    active_node_ids: Set[str] = set()
    active_edge_keys: Set[tuple] = set()
    inferred_edge_keys: Set[tuple] = set()

    if propagation_result:
        active_node_ids.add(origin_id)
        for impact in propagation_result.affected_assets:
            active_node_ids.add(impact.asset_id)
            for path in impact.all_paths:
                for edge in path.edges:
                    active_node_ids.add(edge.source)
                    active_node_ids.add(edge.target)
                    active_edge_keys.add((edge.source, edge.target))
                    if path.hops > 1:
                        inferred_edge_keys.add((edge.source, edge.target))

    # Add Nodes
    # If a propagation result is provided, focus primarily on the active shock subgraph for clarity
    nodes_to_render = active_node_ids if active_node_ids else set(kg.nodes_data.keys())

    for node_id in nodes_to_render:
        node = kg.get_node(node_id)
        if not node:
            continue

        is_origin = (node_id == origin_id)
        n_type = node.type.value
        n_ticker_str = f"Ticker: {node.ticker}" if node.ticker else "Not publicly traded"

        tooltip = f"""
        <div style="background-color: #1E293B; color: #F8FAFC; padding: 8px 12px; border-radius: 6px; font-family: sans-serif; border: 1px solid #334155;">
          <strong style="font-size: 14px; color: #38BDF8;">{node.name}</strong><br/>
          <span style="font-size: 11px; color: #94A3B8;">ID: {node.id} | Type: {n_type}</span><br/>
          <span style="font-size: 11px; color: #FCD34D;">{n_ticker_str}</span><br/>
          <p style="font-size: 12px; margin-top: 5px; color: #E2E8F0;">{node.description}</p>
        </div>
        """

        # Node color and shape styling
        if is_origin:
            color = {"background": "#F59E0B", "border": "#FBBF24", "highlight": {"background": "#D97706", "border": "#FFFFFF"}}
            shape = "star"
            size = 30
        elif n_type == "equity":
            color = {"background": "#0284C7", "border": "#38BDF8", "highlight": {"background": "#0369A1", "border": "#FFFFFF"}}
            shape = "dot"
            size = 20
        elif n_type == "sector":
            color = {"background": "#4F46E5", "border": "#818CF8", "highlight": {"background": "#4338CA", "border": "#FFFFFF"}}
            shape = "box"
            size = 18
        elif n_type == "market_index":
            color = {"background": "#DB2777", "border": "#F472B6", "highlight": {"background": "#BE185D", "border": "#FFFFFF"}}
            shape = "hexagon"
            size = 22
        elif n_type == "commodity":
            color = {"background": "#D97706", "border": "#FBBF24", "highlight": {"background": "#B45309", "border": "#FFFFFF"}}
            shape = "diamond"
            size = 24
        elif n_type == "currency":
            color = {"background": "#059669", "border": "#34D399", "highlight": {"background": "#047857", "border": "#FFFFFF"}}
            shape = "diamond"
            size = 24
        else:
            color = {"background": "#64748B", "border": "#94A3B8", "highlight": {"background": "#475569", "border": "#FFFFFF"}}
            shape = "dot"
            size = 16

        net.add_node(
            node.id,
            label=node.name,
            title=tooltip,
            color=color,
            shape=shape,
            size=size,
        )

    # Add Edges
    for edge in kg.edges_data:
        if edge.source not in nodes_to_render or edge.target not in nodes_to_render:
            continue

        edge_key = (edge.source, edge.target)
        is_direct_shock = (edge.source == origin_id)
        is_active_path = edge_key in active_edge_keys

        # Sign Color
        if edge.sign.value == "positive":
            edge_color = "#10B981"  # Emerald Green
        elif edge.sign.value == "negative":
            edge_color = "#EF4444"  # Crimson Red
        else:
            edge_color = "#F59E0B"  # Amber

        # Line Style encoding
        # - Direct shock: Solid, thick line (width=4)
        # - Curated structural prior: Dashed line (dashes=[8, 4], width=2)
        # - Inferred multi-hop: Dotted line (dashes=[3, 3], width=2)
        if is_direct_shock:
            dashes = False
            width = 4
            edge_style_label = "Observed Direct Shock"
        elif edge_key in inferred_edge_keys:
            dashes = [3, 3]  # Dotted
            width = 2
            edge_style_label = "Inferred Multi-Hop Path"
        else:
            dashes = [8, 4]  # Dashed
            width = 2
            edge_style_label = "Curated Structural Prior"

        evidence_str = "".join([f"<li>{e}</li>" for e in edge.evidence_refs]) if edge.evidence_refs else "<li>None cited</li>"

        edge_tooltip = f"""
        <div style="background-color: #1E293B; color: #F8FAFC; padding: 8px 12px; border-radius: 6px; font-family: sans-serif; border: 1px solid #334155;">
          <strong style="color: {edge_color};">{edge.source} ➔ {edge.target}</strong><br/>
          <span style="font-size: 11px; color: #94A3B8;">Style: {edge_style_label} | Provenance: {edge.provenance.value}</span><br/>
          <span style="font-size: 11px;">Sign: <b>{edge.sign.value.upper()}</b> | Weight: <b>{edge.weight:.2f}</b> | Lag: <b>{edge.lag_days}d</b></span><br/>
          <p style="font-size: 12px; margin: 4px 0; color: #E2E8F0;"><b>Rationale:</b> {edge.rationale}</p>
          <ul style="font-size: 11px; margin: 0; padding-left: 15px; color: #CBD5E1;">{evidence_str}</ul>
        </div>
        """

        net.add_edge(
            edge.source,
            edge.target,
            title=edge_tooltip,
            color=edge_color,
            width=width,
            dashes=dashes,
            label=f"{edge.sign.value[0].upper()} ({edge.weight:.2f})",
        )

    return net.generate_html()


def render_graph_legend():
    """Render interactive HTML legend explaining edge styles, line dashes, and color codes."""
    legend_html = """
    <div style="background-color: #1A1D24; padding: 12px 18px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px; font-family: sans-serif;">
      <div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: center; justify-content: space-between;">
        
        <div>
          <span style="color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Edge Transmission Style:</span>
          <div style="display: flex; gap: 15px; margin-top: 4px; font-size: 12px; color: #E2E8F0;">
            <span><strong style="color: #F8FAFC;">━━━</strong> <b>Solid:</b> Observed Direct Shock</span>
            <span><strong style="color: #F8FAFC;">- - -</strong> <b>Dashed:</b> Curated Structural Prior</span>
            <span><strong style="color: #F8FAFC;">····</strong> <b>Dotted:</b> Inferred Multi-Hop Path</span>
          </div>
        </div>

        <div>
          <span style="color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Hypothesized Sign:</span>
          <div style="display: flex; gap: 15px; margin-top: 4px; font-size: 12px;">
            <span style="color: #10B981; font-weight: 600;">● Positive (+1)</span>
            <span style="color: #EF4444; font-weight: 600;">● Negative (-1)</span>
            <span style="color: #F59E0B; font-weight: 600;">● Ambiguous (0)</span>
          </div>
        </div>

        <div>
          <span style="color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Node Categories:</span>
          <div style="display: flex; gap: 12px; margin-top: 4px; font-size: 12px; color: #E2E8F0;">
            <span>⭐ Origin Shock</span>
            <span>🔷 Commodity/FX</span>
            <span>📦 Sector</span>
            <span>🔵 Equity</span>
            <span>🛑 Benchmark Index</span>
          </div>
        </div>

      </div>
    </div>
    """
    return legend_html
