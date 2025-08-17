#!/usr/bin/env python3
"""
Genome diagram visualization using Plotly.
Creates interactive genome diagrams with gene annotations and PFAM domains.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import colorsys
import re

# Color schemes
COLORS = {
    'core_aligned': 'rgba(255, 99, 71, 0.9)',     # Bright tomato - core syntenic
    'syntenic': 'rgba(255, 99, 71, 0.8)',         # Tomato - legacy support
    'boundary': 'rgba(255, 165, 0, 0.8)',         # Orange - boundary genes
    'context': 'rgba(135, 206, 235, 0.6)',        # Light sky blue - context
    'non_syntenic': 'rgba(135, 206, 235, 0.8)',   # Sky blue - legacy support
    'forward_strand': '#2E8B57',                   # Sea green
    'reverse_strand': '#FF6347',                   # Tomato
    'scale_bar': '#2F4F4F',                        # Dark slate gray
    'background': '#F8F9FA'                        # Light gray
}

def generate_domain_color(domain: str) -> str:
    """Generate consistent color for PFAM domain based on hash."""
    # Use hash to generate consistent color
    hash_val = hash(domain) % 360
    # Convert HSV to RGB with fixed saturation and value for consistency
    rgb = colorsys.hsv_to_rgb(hash_val / 360, 0.7, 0.9)
    return f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)'

def create_gene_arrow_path(gene_data: Dict, y_center: float = 1.0, 
                          height: float = 0.3, arrow_size: float = 0.1) -> str:
    """
    Create SVG path for gene arrow shape based on strand direction.
    
    Args:
        gene_data: Dict with 'start_pos', 'end_pos', 'strand'
        y_center: Y coordinate for gene center
        height: Gene height
        arrow_size: Relative size of arrow head
    """
    start = gene_data['start_pos']
    end = gene_data['end_pos']
    strand = gene_data['strand']
    
    y_top = y_center + height / 2
    y_bottom = y_center - height / 2
    
    gene_length = end - start
    arrow_length = min(gene_length * arrow_size, gene_length * 0.5)  # Max 50% of gene
    
    if strand == 1:  # Forward strand (arrow points right)
        arrow_start = end - arrow_length
        path = f"M {start},{y_bottom} L {arrow_start},{y_bottom} L {end},{y_center} L {arrow_start},{y_top} L {start},{y_top} Z"
    else:  # Reverse strand (arrow points left)
        arrow_end = start + arrow_length
        path = f"M {start},{y_center} L {arrow_end},{y_top} L {end},{y_top} L {end},{y_bottom} L {arrow_end},{y_bottom} Z"
    
    return path

def create_scale_bar(start_pos: int, end_pos: int, num_ticks: int = 5) -> Tuple[List, List, List]:
    """Create scale bar with tick marks and labels."""
    positions = np.linspace(start_pos, end_pos, num_ticks)
    
    # Create human-readable labels
    labels = []
    for pos in positions:
        if pos >= 1_000_000:
            labels.append(f"{pos/1_000_000:.1f}M")
        elif pos >= 1_000:
            labels.append(f"{pos/1_000:.0f}k")
        else:
            labels.append(f"{pos:.0f}")
    
    return positions.tolist(), [1] * len(positions), labels

def create_genome_diagram(genes_df: pd.DataFrame, locus_id: str, 
                         syntenic_blocks: Optional[List[Dict]] = None,
                         width: int = 1200, height: int = 500) -> go.Figure:
    """
    Create interactive genome diagram with genes and PFAM domains.
    
    Args:
        genes_df: DataFrame with gene information
        locus_id: Locus identifier for title
        syntenic_blocks: Optional list of syntenic block regions to highlight
        width: Plot width
        height: Plot height
    
    Returns:
        Plotly figure object
    """
    if genes_df.empty:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text="No genes found", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get genomic coordinates - focus on syntenic region + flanking context only
    if 'synteny_role' in genes_df.columns:
        # Filter to core syntenic region + flanking context (3 genes on each side)
        focused_genes = genes_df[genes_df['synteny_role'].isin(['core_aligned', 'boundary', 'context'])]
        if not focused_genes.empty:
            start_pos = focused_genes['start_pos'].min()
            end_pos = focused_genes['end_pos'].max()
        else:
            # Fallback if no synteny role data
            start_pos = genes_df['start_pos'].min()
            end_pos = genes_df['end_pos'].max()
    else:
        # Fallback for genes without synteny role information
        start_pos = genes_df['start_pos'].min()
        end_pos = genes_df['end_pos'].max()
    
    total_length = end_pos - start_pos
    
    # Create subplot with 4 tracks: scale, genes, pfam domains, legend
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.15, 0.6, 0.25],  # Scale, genes, domains
        vertical_spacing=0.05,
        subplot_titles=["Scale", f"Genes: {locus_id}", "PFAM Domains"]
    )
    
    # Track 1: Scale bar
    scale_positions, scale_y, scale_labels = create_scale_bar(start_pos, end_pos)
    
    fig.add_trace(
        go.Scatter(
            x=[start_pos, end_pos], 
            y=[1, 1], 
            mode='lines',
            line=dict(color=COLORS['scale_bar'], width=3),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Add scale ticks
    for pos, label in zip(scale_positions, scale_labels):
        fig.add_trace(
            go.Scatter(
                x=[pos], y=[1], 
                mode='markers+text',
                marker=dict(symbol='line-ns', size=10, color=COLORS['scale_bar']),
                text=[label],
                textposition='bottom center',
                textfont=dict(color='black'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # Track 2: Genes
    gene_shapes = []
    gene_hovers = []
    
    # Determine if genes are in syntenic blocks
    syntenic_regions = set()
    if syntenic_blocks:
        for block in syntenic_blocks:
            # This would need actual coordinate mapping in practice
            syntenic_regions.update(range(block.get('start', 0), block.get('end', 0)))
    
    # Find boundary positions for aligned region markers
    core_aligned_genes = genes_df[genes_df.get('synteny_role') == 'core_aligned'] if 'synteny_role' in genes_df.columns else pd.DataFrame()
    aligned_start_pos = aligned_end_pos = None
    
    if not core_aligned_genes.empty:
        aligned_start_pos = core_aligned_genes['start_pos'].min()
        aligned_end_pos = core_aligned_genes['end_pos'].max()
    
    for _, gene in genes_df.iterrows():
        # Determine gene color and properties based on synteny role
        if 'synteny_role' in gene and pd.notna(gene['synteny_role']):
            role = gene['synteny_role']
            if role == 'core_aligned':
                gene_color = COLORS['core_aligned']
                line_width = 2  # Thicker border for core aligned
            elif role == 'boundary':
                gene_color = COLORS['boundary']
                line_width = 1.5
            elif role == 'context':
                gene_color = COLORS['context']
                line_width = 1
            elif role == 'syntenic':  # Legacy support
                gene_color = COLORS['syntenic']
                line_width = 1.5
            else:
                gene_color = '#CCCCCC'  # Unknown/gray
                line_width = 1
        else:
            # Fallback to old syntenic regions logic
            is_syntenic = bool(syntenic_regions.intersection(range(gene['start_pos'], gene['end_pos'])))
            gene_color = COLORS['syntenic'] if is_syntenic else COLORS['non_syntenic']
            line_width = 1
        
        # Determine y position based on strand
        y_center = 1.3 if gene['strand'] == 1 else 0.7
        
        # Create gene arrow path
        arrow_path = create_gene_arrow_path(gene, y_center, height=0.25)
        
        # Add gene shape with variable line width
        fig.add_shape(
            type="path",
            path=arrow_path,
            fillcolor=gene_color,
            line=dict(color="black", width=line_width),
            row=2, col=1
        )
        
        # Add invisible scatter point for hover
        fig.add_trace(
            go.Scatter(
                x=[(gene['start_pos'] + gene['end_pos']) / 2],
                y=[y_center],
                mode='markers',
                marker=dict(size=1, opacity=0),
                hovertemplate=(
                    f"<b>{gene['gene_id']}</b><br>"
                    f"Position: {gene['start_pos']:,}-{gene['end_pos']:,} bp<br>"
                    f"Strand: {'+' if gene['strand'] == 1 else '-'}<br>"
                    f"Length: {gene['gene_length']:,} bp<br>"
                    f"Block Role: {gene.get('position_in_block', gene.get('synteny_role', 'unknown'))}<br>"
                    f"Gene Index: {gene.get('gene_index', 'N/A')}<br>"
                    f"PFAM: {gene['pfam_domains'] if gene['pfam_domains'] else 'None'}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=gene['gene_id']
            ),
            row=2, col=1
        )
    
    # Add vertical boundary lines for aligned region
    if aligned_start_pos is not None and aligned_end_pos is not None:
        # Start boundary line
        fig.add_shape(
            type="line",
            x0=aligned_start_pos, x1=aligned_start_pos,
            y0=0.2, y1=1.8,
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=1
        )
        
        # End boundary line
        fig.add_shape(
            type="line",
            x0=aligned_end_pos, x1=aligned_end_pos,
            y0=0.2, y1=1.8,
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=1
        )
        
        # Add boundary labels
        fig.add_annotation(
            x=aligned_start_pos,
            y=1.9,
            text="Aligned Start",
            showarrow=False,
            font=dict(size=10, color="red"),
            row=2, col=1
        )
        
        fig.add_annotation(
            x=aligned_end_pos,
            y=1.9,
            text="Aligned End",
            showarrow=False,
            font=dict(size=10, color="red"),
            row=2, col=1
        )
    
    # Track 3: PFAM Domains
    domain_y_pos = 1.0
    domain_height = 0.4
    unique_domains = set()
    
    for _, gene in genes_df.iterrows():
        if not gene['pfam_domains']:
            continue
        
        domains = gene['pfam_domains'].split(';')
        gene_length = gene['end_pos'] - gene['start_pos']
        
        # Distribute domains evenly across gene length
        domain_width = gene_length / len(domains)
        
        for i, domain in enumerate(domains):
            unique_domains.add(domain)
            domain_color = generate_domain_color(domain)
            
            domain_start = gene['start_pos'] + (i * domain_width)
            domain_end = domain_start + domain_width
            
            # Add domain rectangle
            fig.add_shape(
                type="rect",
                x0=domain_start, x1=domain_end,
                y0=domain_y_pos - domain_height/2,
                y1=domain_y_pos + domain_height/2,
                fillcolor=domain_color,
                line=dict(color="black", width=1),
                row=3, col=1
            )
            
            # Add domain hover
            fig.add_trace(
                go.Scatter(
                    x=[(domain_start + domain_end) / 2],
                    y=[domain_y_pos],
                    mode='markers',
                    marker=dict(size=1, opacity=0),
                    hovertemplate=(
                        f"<b>{domain}</b><br>"
                        f"Gene: {gene['gene_id']}<br>"
                        f"Position: {domain_start:.0f}-{domain_end:.0f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name=domain
                ),
                row=3, col=1
            )
    
    # Configure layout
    fig.update_layout(
        title={
            'text': f"Genome Diagram: {locus_id}",
            'x': 0.5,
            'font': {'size': 16}
        },
        height=height,
        width=width,
        showlegend=False,
        dragmode='pan',
        plot_bgcolor=COLORS['background'],
        hovermode='closest'
    )
    
    # Configure axes
    for i in range(1, 4):
        # X-axes: show coordinates on bottom track only
        fig.update_xaxes(
            showgrid=True if i == 3 else False,
            title="Position (bp)" if i == 3 else "",
            range=[start_pos - total_length * 0.05, end_pos + total_length * 0.05],
            row=i, col=1
        )
        
        # Y-axes: hide ticks, adjust ranges
        y_ranges = {1: [0.5, 1.5], 2: [0.2, 1.8], 3: [0.3, 1.7]}
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            range=y_ranges[i],
            row=i, col=1
        )
    
    # Add legend as annotation
    if len(unique_domains) > 0:
        legend_text = f"PFAM Domains ({len(unique_domains)} unique): " + "; ".join(sorted(list(unique_domains))[:10])
        if len(unique_domains) > 10:
            legend_text += f" + {len(unique_domains) - 10} more"
        
        fig.add_annotation(
            text=legend_text,
            xref="paper", yref="paper",
            x=0, y=-0.1, xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig

def create_comparative_genome_view(query_genes: pd.DataFrame, target_genes: pd.DataFrame,
                                  query_locus: str, target_locus: str,
                                  syntenic_connections: Optional[List[Dict]] = None,
                                  width: int = 1200, height: int = 600) -> go.Figure:
    """
    Create side-by-side comparison of two genomic loci.
    
    Args:
        query_genes: DataFrame with query locus genes
        target_genes: DataFrame with target locus genes  
        query_locus: Query locus identifier
        target_locus: Target locus identifier
        syntenic_connections: Optional list of connections between loci
        width: Plot width
        height: Plot height
    
    Returns:
        Plotly figure with comparative view
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.45, 0.45],
        vertical_spacing=0.1,
        subplot_titles=[f"Query: {query_locus}", f"Target: {target_locus}"]
    )
    
    # Plot query locus (top)
    if not query_genes.empty:
        _add_genes_to_subplot(fig, query_genes, row=1, y_offset=1.0)
    
    # Plot target locus (bottom)  
    if not target_genes.empty:
        _add_genes_to_subplot(fig, target_genes, row=2, y_offset=1.0)
    
    # Add syntenic connections if provided
    if syntenic_connections and not query_genes.empty and not target_genes.empty:
        _add_syntenic_connections(fig, syntenic_connections, 
                                query_genes, target_genes)
    
    # Configure layout
    fig.update_layout(
        title=f"Comparative View: {query_locus} vs {target_locus}",
        height=height,
        width=width,
        showlegend=False,
        dragmode='pan'
    )
    
    # Configure axes
    for i in [1, 2]:
        fig.update_xaxes(title="Position (bp)", row=i, col=1)
        fig.update_yaxes(showticklabels=False, row=i, col=1)
    
    return fig

def _add_genes_to_subplot(fig: go.Figure, genes_df: pd.DataFrame, 
                         row: int, y_offset: float = 1.0):
    """Helper function to add genes to a subplot."""
    for _, gene in genes_df.iterrows():
        # Determine gene color and y position
        gene_color = COLORS['forward_strand'] if gene['strand'] == 1 else COLORS['reverse_strand']
        y_center = y_offset + (0.2 if gene['strand'] == 1 else -0.2)
        
        # Create gene arrow
        arrow_path = create_gene_arrow_path(gene, y_center, height=0.2)
        
        fig.add_shape(
            type="path",
            path=arrow_path,
            fillcolor=gene_color,
            line=dict(color="black", width=1),
            row=row, col=1
        )
        
        # Add hover
        fig.add_trace(
            go.Scatter(
                x=[(gene['start_pos'] + gene['end_pos']) / 2],
                y=[y_center],
                mode='markers',
                marker=dict(size=1, opacity=0),
                hovertemplate=(
                    f"<b>{gene['gene_id']}</b><br>"
                    f"Position: {gene['start_pos']:,}-{gene['end_pos']:,}<br>"
                    f"PFAM: {gene['pfam_domains'] if gene['pfam_domains'] else 'None'}"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=row, col=1
        )

def _add_syntenic_connections(fig: go.Figure, connections: List[Dict],
                            query_genes: pd.DataFrame, target_genes: pd.DataFrame):
    """Helper function to add syntenic connection lines between loci."""
    # This would implement bezier curves or straight lines connecting
    # syntenic genes between the two tracks
    # Implementation depends on the specific connection data format
    pass

def create_cluster_overview(cluster_data: Dict, member_blocks: List[Dict],
                          width: int = 1000, height: int = 400) -> go.Figure:
    """
    Create overview visualization for a syntenic cluster.
    
    Args:
        cluster_data: Dictionary with cluster metadata
        member_blocks: List of syntenic blocks in the cluster
        width: Plot width  
        height: Plot height
    
    Returns:
        Plotly figure with cluster overview
    """
    if not member_blocks:
        fig = go.Figure()
        fig.add_annotation(text="No blocks in cluster", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create scatter plot of blocks by length vs identity
    lengths = [block['length'] for block in member_blocks]
    identities = [block['identity'] for block in member_blocks]
    scores = [block['score'] for block in member_blocks]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lengths,
        y=identities,
        mode='markers',
        marker=dict(
            size=[np.log10(score) * 5 for score in scores],  # Size by score
            color=scores,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        text=[f"Block {b['block_id']}" for b in member_blocks],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Length: %{x:,} gene windows<br>"
            "Embedding similarity: %{y:.3f}<br>"
            "Score: %{marker.color:.1f}"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=f"Cluster {cluster_data['cluster_id']} Overview ({len(member_blocks)} blocks)",
        xaxis_title="Block Length (gene windows)",
        yaxis_title="Embedding similarity",
        width=width,
        height=height,
        xaxis_type="log"
    )
    
    return fig
