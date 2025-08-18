#!/usr/bin/env python3
"""
ELSA Genome Browser - Streamlit Application

Interactive genome browser for exploring syntenic blocks with PFAM domain annotations.
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from types import SimpleNamespace
import math
import time

# Import genome visualization functions
from cluster_analyzer import ClusterAnalyzer
from visualization.genome_plots import create_genome_diagram, create_comparative_genome_view
from elsa.analyze.shingles import srp_tokens, block_shingles

# Configure page
st.set_page_config(
    page_title="ELSA Genome Browser",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = Path("genome_browser.db")
COLORS = {
    'syntenic': 'rgba(255, 99, 71, 0.8)',      # Tomato
    'non_syntenic': 'rgba(135, 206, 235, 0.8)', # Sky blue
    'forward_strand': 'rgba(34, 139, 34, 0.8)',  # Forest green
    'reverse_strand': 'rgba(255, 140, 0, 0.8)',   # Dark orange
    'pfam_domain': 'rgba(147, 112, 219, 0.8)',   # Medium slate blue
}

@st.cache_resource
def get_database_connection():
    """Get cached database connection."""
    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.info("Please run the database setup and data ingestion scripts first.")
        st.stop()
    
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_resource
def get_cluster_analyzer():
    """Get cached cluster analyzer instance."""
    return ClusterAnalyzer(DB_PATH)

@st.cache_data
def load_genomes() -> pd.DataFrame:
    """Load genome information."""
    conn = get_database_connection()
    return pd.read_sql_query("SELECT * FROM genomes ORDER BY genome_id", conn)

@st.cache_data
def load_syntenic_blocks(limit: int = 1000, offset: int = 0,
                        genome_filter: Optional[List[str]] = None,
                        size_range: Optional[Tuple[int, int]] = None,
                        identity_threshold: float = 0.0,
                        block_type: Optional[str] = None) -> pd.DataFrame:
    """Load syntenic blocks with filters and pagination."""
    conn = get_database_connection()
    
    query = """
        SELECT sb.*, 
               g1.organism_name as query_organism,
               g2.organism_name as target_organism
        FROM syntenic_blocks sb
        LEFT JOIN genomes g1 ON sb.query_genome_id = g1.genome_id
        LEFT JOIN genomes g2 ON sb.target_genome_id = g2.genome_id
        WHERE 1=1
    """
    params = []
    
    # Apply filters
    if genome_filter:
        placeholders = ','.join(['?' for _ in genome_filter])
        query += f" AND (sb.query_genome_id IN ({placeholders}) OR sb.target_genome_id IN ({placeholders}))"
        params.extend(genome_filter * 2)
    
    if size_range:
        query += " AND sb.length BETWEEN ? AND ?"
        params.extend(size_range)
    
    if identity_threshold > 0:
        query += " AND sb.identity >= ?"
        params.append(identity_threshold)
    
    if block_type:
        query += " AND sb.block_type = ?"
        params.append(block_type)
    
    query += " ORDER BY sb.identity DESC, sb.block_id"  # Show high-quality alignments in a more representative order
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    return pd.read_sql_query(query, conn, params=params)

def load_genes_for_locus(locus_id: str, block_id: Optional[int] = None, locus_role: Optional[str] = None, extended_context: bool = False) -> pd.DataFrame:
    """Load genes for a specific locus, optionally filtered to aligned regions."""
    conn = get_database_connection()
    
    # Parse locus ID format: "1313.30775:1313.30775_accn|1313.30775.con.0001"
    # Need to extract contig_id to match genes table format: "accn|1313.30775.con.0001"
    if ':' in locus_id:
        genome_part, contig_part = locus_id.split(':', 1)
        # Extract the actual contig_id from the second part
        # "1313.30775_accn|1313.30775.con.0001" -> "accn|1313.30775.con.0001"
        if '_' in contig_part:
            contig_id = contig_part.split('_', 1)[1]  # Take everything after first '_'
        else:
            contig_id = contig_part
    else:
        # Fallback: treat entire string as contig
        contig_id = locus_id
    
    # If we have block info, load only the aligned region + context
    if block_id is not None and locus_role is not None:
        return load_aligned_genes_for_block(conn, contig_id, block_id, locus_role, extended_context)
    
    # Default: load all genes for the locus
    query = """
        SELECT g.*, c.length as contig_length
        FROM genes g
        JOIN contigs c ON g.contig_id = c.contig_id
        WHERE g.contig_id = ?
        ORDER BY g.start_pos
    """
    
    return pd.read_sql_query(query, conn, params=[contig_id])

def load_genes_for_region(contig_id: str, start_bp: int, end_bp: int, extended_context: bool = False) -> pd.DataFrame:
    """Load genes for a contig focused on a bp interval, with flanking context and synteny roles.

    This is region-centric (not tied to a single block) and works even on single-contig genomes.
    """
    conn = get_database_connection()

    # Load all genes on this contig in order
    all_genes = pd.read_sql_query(
        """
        SELECT g.*, c.length as contig_length
        FROM genes g
        JOIN contigs c ON g.contig_id = c.contig_id
        WHERE g.contig_id = ?
        ORDER BY g.start_pos
        """,
        conn,
        params=[contig_id],
    )

    if all_genes.empty:
        return all_genes

    # Compute gene_index 0-based
    all_genes = all_genes.copy()
    all_genes["gene_index"] = range(len(all_genes))

    # Determine indices spanning the interval
    # First gene whose end covers the start, last gene whose start is before the end
    core_start_candidates = all_genes.index[all_genes["end_pos"] >= int(start_bp)].tolist()
    core_end_candidates = all_genes.index[all_genes["start_pos"] <= int(end_bp)].tolist()
    if not core_start_candidates or not core_end_candidates:
        # No overlap; return a small slice around the nearest region
        mid = len(all_genes) // 2
        slice_df = all_genes.iloc[max(0, mid - 25) : min(len(all_genes), mid + 25)].copy()
        slice_df["synteny_role"] = "context"
        slice_df["position_in_block"] = "Flanking Region"
        return slice_df

    core_start_idx = core_start_candidates[0]
    core_end_idx = core_end_candidates[-1]

    buffer = 10 if extended_context else 3
    start_idx = max(0, core_start_idx - buffer)
    end_idx = min(len(all_genes) - 1, core_end_idx + buffer)

    genes_df = all_genes.iloc[start_idx : end_idx + 1].copy()

    # Assign roles
    roles = []
    positions = []
    for gi, idx in enumerate(genes_df["gene_index"]):
        if core_start_idx <= idx <= core_end_idx:
            role = "core_aligned"
            pos_label = "Conserved Block"
        elif idx == core_start_idx - 1 or idx == core_end_idx + 1:
            role = "boundary"
            pos_label = "Block Edge"
        else:
            role = "context"
            pos_label = "Flanking Region"
        roles.append(role)
        positions.append(pos_label)

    genes_df["synteny_role"] = roles
    genes_df["position_in_block"] = positions

    return genes_df

def load_aligned_genes_for_block(_conn, contig_id: str, block_id: int, locus_role: str, extended_context: bool = False) -> pd.DataFrame:
    """Load only the genes that are part of the aligned region for a specific block."""
    
    # Convert numpy integer to regular Python integer to avoid SQLite issues
    block_id = int(block_id)
    
    # Get fresh database connection instead of using potentially stale _conn
    fresh_conn = get_database_connection()
    cursor = fresh_conn.cursor()
    cursor.execute("""
        SELECT query_window_start, query_window_end, target_window_start, target_window_end
        FROM syntenic_blocks 
        WHERE block_id = ?
    """, (block_id,))
    
    result = cursor.fetchone()
    if not result:
        # Fallback to all genes if no alignment info
        return load_genes_for_locus(contig_id)
    
    query_start, query_end, target_start, target_end = result
    
    # Determine which window range to use based on locus role
    if locus_role == 'query' and query_start is not None and query_end is not None:
        window_start, window_end = query_start, query_end
    elif locus_role == 'target' and target_start is not None and target_end is not None:
        window_start, window_end = target_start, target_end
    else:
        # No alignment info available, load all genes
        query = """
            SELECT g.*, c.length as contig_length, 'unknown' as synteny_role
            FROM genes g
            JOIN contigs c ON g.contig_id = c.contig_id
            WHERE g.contig_id = ?
            ORDER BY g.start_pos
        """
        return pd.read_sql_query(query, conn, params=[contig_id])
    
    # Convert window indices to gene ranges
    # Each window contains 5 genes with stride 1: window N covers genes N to N+4
    gene_start_idx = window_start  # Gene index of first gene in first window
    gene_end_idx = window_end + 4  # Gene index of last gene in last window (window contains 5 genes)
    
    # Add context genes around the aligned region (amount depends on extended_context setting)
    context_buffer = 10 if extended_context else 3  # More context if extended_context is True
    gene_start_idx = max(0, gene_start_idx - context_buffer)
    gene_end_idx = gene_end_idx + context_buffer
    
    # Load genes in the range, using ROW_NUMBER to get positional indices
    query = """
        SELECT g.*, c.length as contig_length,
               ROW_NUMBER() OVER (ORDER BY g.start_pos) - 1 as gene_index,
               CASE 
                   WHEN (ROW_NUMBER() OVER (ORDER BY g.start_pos) - 1) BETWEEN ? AND ? 
                   THEN 'core_aligned'
                   WHEN (ROW_NUMBER() OVER (ORDER BY g.start_pos) - 1) BETWEEN ? AND ?
                   THEN 'boundary'
                   ELSE 'context'
               END as synteny_role,
               CASE 
                   WHEN (ROW_NUMBER() OVER (ORDER BY g.start_pos) - 1) BETWEEN ? AND ? 
                   THEN 'Conserved Block'
                   WHEN (ROW_NUMBER() OVER (ORDER BY g.start_pos) - 1) BETWEEN ? AND ?
                   THEN 'Block Edge'
                   ELSE 'Flanking Region'
               END as position_in_block
        FROM genes g
        JOIN contigs c ON g.contig_id = c.contig_id
        WHERE g.contig_id = ?
        ORDER BY g.start_pos
        LIMIT ? OFFSET ?
    """
    
    # Calculate the actual aligned gene range (without context)
    core_aligned_start = window_start
    core_aligned_end = window_end + 4
    
    # Boundary genes are those within the windows but not in the tightest alignment
    boundary_start = max(0, window_start - 1)  
    boundary_end = window_end + 5
    
    limit = gene_end_idx - gene_start_idx + 1
    offset = gene_start_idx
    
    return pd.read_sql_query(query, fresh_conn, params=[
        core_aligned_start, core_aligned_end,  # Core aligned range
        boundary_start, boundary_end,          # Boundary range  
        core_aligned_start, core_aligned_end,  # For display label
        boundary_start, boundary_end,          # For display label
        contig_id, limit, offset
    ])

@st.cache_data
def get_block_count(genome_filter: Optional[List[str]] = None,
                   size_range: Optional[Tuple[int, int]] = None,
                   identity_threshold: float = 0.0,
                   block_type: Optional[str] = None) -> int:
    """Get total count of blocks matching filters."""
    conn = get_database_connection()
    
    query = "SELECT COUNT(*) FROM syntenic_blocks WHERE 1=1"
    params = []
    
    if genome_filter:
        placeholders = ','.join(['?' for _ in genome_filter])
        query += f" AND (query_genome_id IN ({placeholders}) OR target_genome_id IN ({placeholders}))"
        params.extend(genome_filter * 2)
    
    if size_range:
        query += " AND length BETWEEN ? AND ?"
        params.extend(size_range)
    
    if identity_threshold > 0:
        query += " AND identity >= ?"
        params.append(identity_threshold)
    
    if block_type:
        query += " AND block_type = ?"
        params.append(block_type)
    
    result = conn.execute(query, params).fetchone()
    return result[0] if result else 0

def create_sidebar_filters() -> Dict:
    """Create sidebar filters and return filter values."""
    st.sidebar.header("🔍 Filters")
    
    # Load genomes for filter
    genomes_df = load_genomes()
    
    # Genome filter
    selected_genomes = st.sidebar.multiselect(
        "Select Genomes",
        options=genomes_df['genome_id'].tolist(),
        default=genomes_df['genome_id'].tolist()[:3],  # Default to first 3
        help="Select genomes to include in analysis"
    )
    
    # Block size filter (gene windows)
    st.sidebar.subheader("Block Size Range (Gene Windows)")
    size_min = st.sidebar.number_input("Min size (gene windows)", value=1, min_value=1, step=1)
    size_max = st.sidebar.number_input("Max size (gene windows)", value=100, min_value=1, step=1)
    
    # Block type filter
    block_type = st.sidebar.selectbox(
        "Block Type",
        options=[None, 'small', 'medium', 'large'],
        format_func=lambda x: 'All' if x is None else x.title(),
        help="Filter by block size category"
    )
    
    # Identity threshold
    identity_threshold = st.sidebar.slider(
        "Min Embedding Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum cosine similarity in ESM2 embedding space"
    )
    
    # Results per page
    page_size = st.sidebar.selectbox(
        "Results per page",
        options=[25, 50, 100, 200],
        index=1,
        help="Number of blocks to show per page"
    )
    
    # PFAM domain search
    st.sidebar.subheader("🎯 PFAM Search")
    domain_search = st.sidebar.text_input(
        "Search domains",
        placeholder="e.g., PF00001, kinase, transferase",
        help="Search for specific PFAM domains or keywords"
    )
    
    return {
        'genomes': selected_genomes,
        'size_range': (size_min, size_max),
        'identity_threshold': identity_threshold,
        'block_type': block_type,
        'page_size': page_size,
        'domain_search': domain_search
    }

def display_dashboard():
    """Display overview dashboard."""
    st.header("📊 ELSA Analysis Dashboard")
    
    # Load summary statistics
    conn = get_database_connection()
    
    # Get basic stats
    stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM genomes) as total_genomes,
            (SELECT COUNT(*) FROM syntenic_blocks) as total_blocks,
            (SELECT COUNT(*) FROM clusters) as total_clusters,
            (SELECT COUNT(*) FROM genes) as total_genes,
            (SELECT COUNT(*) FROM genes WHERE pfam_domains != '') as annotated_genes
    """
    
    stats = conn.execute(stats_query).fetchone()
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🧬 Genomes", f"{stats[0]:,}")
    with col2:
        st.metric("🔗 Syntenic Blocks", f"{stats[1]:,}")
    with col3:
        st.metric("🎯 Clusters", f"{stats[2]:,}")
    with col4:
        st.metric("🧬 Total Genes", f"{stats[3]:,}")
    with col5:
        annotation_pct = (stats[4] / stats[3] * 100) if stats[3] > 0 else 0
        st.metric("📝 Annotated", f"{annotation_pct:.1f}%")
    
    # Block size distribution
    st.subheader("Block Size Distribution")
    
    size_dist_query = """
        SELECT block_type, COUNT(*) as count
        FROM syntenic_blocks 
        GROUP BY block_type
        ORDER BY 
            CASE block_type 
                WHEN 'small' THEN 1 
                WHEN 'medium' THEN 2 
                WHEN 'large' THEN 3 
            END
    """
    
    size_dist = pd.read_sql_query(size_dist_query, conn)
    
    if not size_dist.empty:
        fig = px.bar(size_dist, x='block_type', y='count', 
                    title="Syntenic Block Size Distribution",
                    labels={'block_type': 'Block Type', 'count': 'Number of Blocks'},
                    color='block_type')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Genome comparison matrix
    st.subheader("Genome Comparison Matrix")
    
    matrix_query = """
        SELECT query_genome_id, target_genome_id, COUNT(*) as block_count
        FROM syntenic_blocks
        GROUP BY query_genome_id, target_genome_id
        ORDER BY block_count DESC
    """
    
    matrix_data = pd.read_sql_query(matrix_query, conn)
    
    if not matrix_data.empty:
        # Create pivot table
        matrix_pivot = matrix_data.pivot(index='query_genome_id', 
                                       columns='target_genome_id', 
                                       values='block_count').fillna(0)
        
        fig = px.imshow(matrix_pivot, 
                       title="Syntenic Block Counts Between Genomes",
                       labels=dict(x="Target Genome", y="Query Genome", color="Blocks"),
                       aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def display_block_explorer(filters: Dict):
    """Display syntenic block explorer with pagination."""
    st.header("🔍 Syntenic Block Explorer")
    
    # Get total count for pagination
    total_blocks = get_block_count(
        genome_filter=filters['genomes'],
        size_range=filters['size_range'],
        identity_threshold=filters['identity_threshold'],
        block_type=filters['block_type']
    )
    
    if total_blocks == 0:
        st.warning("No blocks found matching the current filters.")
        return
    
    # Pagination controls
    page_size = filters['page_size']
    total_pages = math.ceil(total_blocks / page_size)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        current_page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
    
    with col3:
        st.info(f"Total: {total_blocks:,} blocks")
    
    # Load blocks for current page
    offset = (current_page - 1) * page_size
    blocks_df = load_syntenic_blocks(
        limit=page_size,
        offset=offset,
        genome_filter=filters['genomes'],
        size_range=filters['size_range'],
        identity_threshold=filters['identity_threshold'],
        block_type=filters['block_type']
    )
    
    if blocks_df.empty:
        st.warning("No blocks found for current page.")
        return
    
    # Display blocks table
    st.subheader(f"Blocks {offset + 1}-{offset + len(blocks_df)} of {total_blocks:,}")
    
    # Format display columns - keep length as numeric for proper sorting
    display_df = blocks_df[['block_id', 'query_locus', 'target_locus', 
                           'length', 'identity', 'score', 'block_type']].copy()
    
    # Format other columns but keep length numeric
    display_df['identity'] = display_df['identity'].apply(lambda x: f"{x:.3f}")
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1f}")
    
    # Rename columns to show proper units and clarify data types
    display_df = display_df.rename(columns={
        'length': 'length (gene windows)',
        'identity': 'embedding_similarity'
    })
    
    # Add selection
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Show detailed view for selected block
    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_block = blocks_df.iloc[selected_idx]
        
        st.subheader(f"Block Details: {selected_block['block_id']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Query Locus:**", selected_block['query_locus'])
            st.write("**Length:**", f"{selected_block['length']:,} gene windows")
            st.write("**Query Windows:**", f"{selected_block['n_query_windows']:,}")
            
            # Show window range information if available
            if selected_block.get('query_window_start') is not None:
                window_range = f"Windows {selected_block['query_window_start']}-{selected_block['query_window_end']}"
                gene_range = f"Genes {selected_block['query_window_start']}-{selected_block['query_window_end'] + 4}"
                st.write("**Query Range:**", f"{window_range} ({gene_range})")
        
        with col2:
            st.write("**Target Locus:**", selected_block['target_locus'])
            st.write("**Embedding Similarity:**", f"{selected_block['identity']:.3f}")
            st.write("**Target Windows:**", f"{selected_block['n_target_windows']:,}")
            
            # Show window range information if available
            if selected_block.get('target_window_start') is not None:
                window_range = f"Windows {selected_block['target_window_start']}-{selected_block['target_window_end']}"
                gene_range = f"Genes {selected_block['target_window_start']}-{selected_block['target_window_end'] + 4}"
                st.write("**Target Range:**", f"{window_range} ({gene_range})")
        
        # Add debugging section for window-level analysis
        if st.checkbox("🔍 Show Detailed Window Analysis", key=f"debug_{selected_block['block_id']}"):
            st.subheader("Window-Level Similarity Analysis")
            
            # Parse window information from the block
            query_windows_json = selected_block.get('query_windows_json')
            target_windows_json = selected_block.get('target_windows_json')
            
            if query_windows_json and target_windows_json:
                import json
                
                try:
                    query_windows = json.loads(query_windows_json)
                    target_windows = json.loads(target_windows_json)
                    
                    st.write(f"**Total Matched Windows:** {len(query_windows)}")
                    
                    # Create window mapping table
                    window_debug_data = []
                    for i, (q_win, t_win) in enumerate(zip(query_windows, target_windows)):
                        # Extract window indices
                        q_idx = q_win.split('_')[-1] if '_' in q_win else 'N/A'
                        t_idx = t_win.split('_')[-1] if '_' in t_win else 'N/A'
                        
                        # Calculate gene ranges (window N = genes N to N+4)
                        try:
                            q_gene_start = int(q_idx)
                            q_gene_end = q_gene_start + 4
                            q_gene_range = f"{q_gene_start}-{q_gene_end}"
                        except:
                            q_gene_range = "N/A"
                        
                        try:
                            t_gene_start = int(t_idx)
                            t_gene_end = t_gene_start + 4
                            t_gene_range = f"{t_gene_start}-{t_gene_end}"
                        except:
                            t_gene_range = "N/A"
                        
                        window_debug_data.append({
                            'Match #': i + 1,
                            'Query Window': q_win,
                            'Query Genes': q_gene_range,
                            'Target Window': t_win,
                            'Target Genes': t_gene_range,
                            'Position Offset': abs(int(q_idx) - int(t_idx)) if q_idx.isdigit() and t_idx.isdigit() else 'N/A'
                        })
                    
                    debug_df = pd.DataFrame(window_debug_data)
                    st.dataframe(debug_df, hide_index=True)
                    
                    # Analysis insights
                    st.subheader("🧠 Algorithm Insights")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("**Window Matches**", len(query_windows))
                        st.metric("**Embedding Similarity**", f"{selected_block['identity']:.3f}")
                    
                    with col2:
                        # Calculate positional conservation
                        if debug_df['Position Offset'].dtype != 'object':
                            avg_offset = debug_df['Position Offset'].mean()
                            st.metric("**Avg Position Offset**", f"{avg_offset:.1f}")
                        
                        # Show block size ratio
                        size_ratio = selected_block['n_target_windows'] / selected_block['n_query_windows']
                        st.metric("**Size Ratio (T/Q)**", f"{size_ratio:.2f}")
                    
                    # Highlight potential issues
                    issues = []
                    if len(query_windows) < 3:
                        issues.append("⚠️ Few window matches - may be spurious")
                    if debug_df['Position Offset'].dtype != 'object' and debug_df['Position Offset'].max() > 5:
                        issues.append("⚠️ Large positional offsets - poor gene order conservation") 
                    if abs(size_ratio - 1.0) > 0.5:
                        issues.append("⚠️ Significant size difference between blocks")
                    
                    if issues:
                        st.warning("**Potential Algorithm Issues:**")
                        for issue in issues:
                            st.write(issue)
                    else:
                        st.success("✅ Block appears to have good syntenic characteristics")
                        
                except json.JSONDecodeError:
                    st.error("Could not parse window information")
            else:
                st.info("No detailed window information available for this block")
        
        # Option to view genome diagram
        if st.button("🧬 View Genome Diagram", key=f"view_{selected_block['block_id']}"):
            st.session_state.selected_block = selected_block.to_dict()  # Convert Series to dict
            st.session_state.current_page = 'genome_viewer'
            st.rerun()

def display_genome_viewer():
    """Display genome diagram viewer."""
    if 'selected_block' not in st.session_state:
        st.warning("Please select a syntenic block first.")
        if st.button("← Back to Block Explorer"):
            st.session_state.current_page = 'block_explorer'
            st.rerun()
        return
    
    block = st.session_state.selected_block
    
    st.header(f"🧬 Genome Viewer - Block {block['block_id']}")
    
    # Back button
    if st.button("← Back to Block Explorer"):
        st.session_state.current_page = 'block_explorer'
        st.rerun()
    
    # GPT-5 Analysis Section (full width at top)
    st.divider()
    st.subheader("🤖 GPT-5 Functional Analysis")
    
    # Initialize GPT analysis state
    gpt_state_key = f"gpt_analysis_{block['block_id']}"
    if gpt_state_key not in st.session_state:
        st.session_state[gpt_state_key] = {"analyzed": False, "data": None, "report": None}
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🤖 Analyze with GPT-5", key="gpt_analysis_btn"):
            with st.spinner("Analyzing syntenic block with GPT-5..."):
                try:
                    from gpt5_analyzer import analyze_syntenic_block
                    analysis_data, gpt_report = analyze_syntenic_block(block['block_id'])
                    
                    if analysis_data and gpt_report:
                        st.session_state[gpt_state_key] = {
                            "analyzed": True,
                            "data": analysis_data,
                            "report": gpt_report
                        }
                        st.success("✅ GPT-5 analysis completed!")
                    else:
                        st.error("❌ Failed to generate GPT-5 analysis")
                        if gpt_report:
                            st.write(f"Error: {gpt_report}")
                
                except ImportError:
                    st.error("❌ GPT-5 analyzer module not found")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"GPT analysis error: {e}")
    
    with col2:
        if st.session_state[gpt_state_key]["analyzed"]:
            st.info("GPT-5 analysis available below")
        else:
            st.info("Click 'Analyze with GPT-5' to generate functional analysis")
    
    # Display GPT analysis if available
    if st.session_state[gpt_state_key]["analyzed"]:
        analysis_data = st.session_state[gpt_state_key]["data"]
        gpt_report = st.session_state[gpt_state_key]["report"]
        
        # Show block summary
        with st.expander("📋 **Block Summary**", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Similarity", f"{analysis_data.identity:.3f}")
            with col2:
                st.metric("Score", f"{analysis_data.score:.2f}")
            with col3:
                st.metric("Length", f"{analysis_data.block_length} windows")
            
            st.write(f"**Query**: {analysis_data.query_locus.organism_name} ({analysis_data.query_locus.gene_count} genes)")
            st.write(f"**Target**: {analysis_data.target_locus.organism_name} ({analysis_data.target_locus.gene_count} genes)")
        
        # Display GPT analysis
        with st.expander("🧠 **GPT-5 Functional Analysis Report**", expanded=True):
            st.markdown(gpt_report)
        
        # Download button
        st.download_button(
            label="📄 Download Analysis Report",
            data=f"# GPT-5 Analysis Report - Block {block['block_id']}\n\n{gpt_report}",
            file_name=f"gpt5_analysis_block_{block['block_id']}.md",
            mime="text/markdown"
        )
    
    st.divider()
    
    # Debug info
    st.write(f"**Debug**: Loading genes for query locus: {block['query_locus']}")
    
    # Load genes for both loci - now showing only aligned regions + context
    with st.spinner("Loading aligned gene regions..."):
        query_genes = load_genes_for_locus(block['query_locus'], block['block_id'], 'query')
        target_genes = load_genes_for_locus(block['target_locus'], block['block_id'], 'target')
    
    st.write(f"**Debug**: Query genes loaded: {len(query_genes)}, Target genes loaded: {len(target_genes)}")
    
    # Add comprehensive biological legend
    with st.expander("📖 **Understanding Syntenic Block Visualization**", expanded=True):
        st.markdown("""
        **Syntenic blocks** represent genomic regions with **conserved gene order** between different bacterial strains or species, 
        indicating evolutionary relatedness and functional importance.
        
        **Gene Classification:**
        - 🔴 **Conserved Block** (thick border): Core genes in the syntenic region showing evolutionary conservation
        - 🟠 **Block Edge**: Genes at the boundaries of the conserved region  
        - 🔵 **Flanking Region**: Neighboring genes outside the conserved block, shown for genomic context
        
        **Gene Orientation:**
        - ➡️ **Forward strand** (+): Gene transcribed left-to-right
        - ⬅️ **Reverse strand** (−): Gene transcribed right-to-left
        
        **Vertical Red Lines**: Mark the precise boundaries of the conserved syntenic block
        
        **PFAM Domains**: Protein family annotations showing functional conservation across the alignment
        """)
    
    st.markdown("---")
    
    # Display genome diagrams using proper visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Query: {block['query_locus']}")
        if not query_genes.empty:
            st.info(f"{len(query_genes)} genes found")
            
            # Create professional genome diagram showing aligned region
            with st.spinner("Generating genome diagram..."):
                query_fig = create_genome_diagram(
                    query_genes,  # Now pre-filtered to aligned region + context
                    f"{block['query_locus']} (Aligned Region)",
                    width=600,
                    height=400
                )
                st.plotly_chart(query_fig, use_container_width=True)
            
            # Show summary data table with enhanced alignment information
            st.write("**Gene Summary:**")
            display_cols = ['gene_id', 'start_pos', 'end_pos', 'strand']
            
            # Add alignment-specific columns if available
            if 'position_in_block' in query_genes.columns:
                display_cols.append('position_in_block')
            if 'gene_index' in query_genes.columns:
                display_cols.append('gene_index')
            if 'synteny_role' in query_genes.columns:
                display_cols.append('synteny_role')
                
            display_cols.append('pfam_domains')
            
            # Create a more informative summary
            summary_df = query_genes[display_cols].copy()
            
            # Color-code the dataframe based on alignment role
            if 'position_in_block' in summary_df.columns:
                def highlight_alignment_role(row):
                    role = row.get('position_in_block', '')
                    if role == 'Conserved Block':
                        return ['background-color: #8B0000; color: white'] * len(row)  # Dark red with white text
                    elif role == 'Block Edge':
                        return ['background-color: #CC6600; color: white'] * len(row)  # Dark orange with white text
                    elif role == 'Flanking Region':
                        return ['background-color: #000080; color: white'] * len(row)  # Dark blue with white text
                    return [''] * len(row)
                
                styled_df = summary_df.style.apply(highlight_alignment_role, axis=1)
                st.dataframe(styled_df, hide_index=True)
            else:
                st.dataframe(summary_df, hide_index=True)
        else:
            st.warning("No genes found for query locus")
    
    with col2:
        st.subheader(f"Target: {block['target_locus']}")
        if not target_genes.empty:
            st.info(f"{len(target_genes)} genes found")
            
            # Create professional genome diagram showing aligned region
            with st.spinner("Generating genome diagram..."):
                target_fig = create_genome_diagram(
                    target_genes,  # Now pre-filtered to aligned region + context
                    f"{block['target_locus']} (Aligned Region)",
                    width=600,
                    height=400
                )
                st.plotly_chart(target_fig, use_container_width=True)
            
            # Show summary data table with enhanced alignment information
            st.write("**Gene Summary:**")
            display_cols = ['gene_id', 'start_pos', 'end_pos', 'strand']
            
            # Add alignment-specific columns if available
            if 'position_in_block' in target_genes.columns:
                display_cols.append('position_in_block')
            if 'gene_index' in target_genes.columns:
                display_cols.append('gene_index')
            if 'synteny_role' in target_genes.columns:
                display_cols.append('synteny_role')
                
            display_cols.append('pfam_domains')
            
            # Create a more informative summary
            summary_df = target_genes[display_cols].copy()
            
            # Color-code the dataframe based on alignment role
            if 'position_in_block' in summary_df.columns:
                def highlight_alignment_role(row):
                    role = row.get('position_in_block', '')
                    if role == 'Conserved Block':
                        return ['background-color: #8B0000; color: white'] * len(row)  # Dark red with white text
                    elif role == 'Block Edge':
                        return ['background-color: #CC6600; color: white'] * len(row)  # Dark orange with white text
                    elif role == 'Flanking Region':
                        return ['background-color: #000080; color: white'] * len(row)  # Dark blue with white text
                    return [''] * len(row)
                
                styled_df = summary_df.style.apply(highlight_alignment_role, axis=1)
                st.dataframe(styled_df, hide_index=True)
            else:
                st.dataframe(summary_df, hide_index=True)
        else:
            st.warning("No genes found for target locus")
    
    # Add comparative view option
    st.divider()
    if not query_genes.empty and not target_genes.empty:
        if st.button("📊 Show Comparative Genome View", key="comparative_view"):
            st.subheader("Comparative Genome Analysis")
            with st.spinner("Generating comparative view..."):
                try:
                    comp_data = _build_comparative_json(block['block_id'], query_genes, target_genes)
                    _render_comparative_d3(comp_data, width=1200, height=500)
                except Exception as e:
                    st.error(f"Comparative view failed: {e}")

def load_cluster_stats():
    """Load precomputed cluster statistics."""
    try:
        from cluster_analyzer import get_all_cluster_stats
        return get_all_cluster_stats()
    except Exception as e:
        logger.error(f"Error loading cluster stats: {e}")
        return []

@st.cache_data
def generate_cluster_summaries(cluster_stats):
    """Generate GPT-4.1-mini summaries for all clusters."""
    try:
        from cluster_analyzer import ClusterAnalyzer
        analyzer = ClusterAnalyzer(Path("genome_browser.db"))
        
        summaries = {}
        for stats in cluster_stats:
            summary = analyzer.generate_cluster_summary(stats)
            summaries[stats.cluster_id] = summary
        
        return summaries
    except Exception as e:
        logger.error(f"Error generating cluster summaries: {e}")
        return {}

def display_cluster_explorer():
    """Display cluster explorer interface."""
    st.header("🧩 Syntenic Block Clusters")
    st.markdown("*Explore clusters of related syntenic blocks with AI-generated functional summaries*")
    
    # Load cluster data
    with st.spinner("Loading cluster analysis..."):
        cluster_stats = load_cluster_stats()
    
    if not cluster_stats:
        st.error("No cluster data available. Please run cluster analysis first.")
        return
    
    st.success(f"Found {len(cluster_stats)} clusters")
    
    # Cluster overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_blocks = sum(stats.size for stats in cluster_stats)
        st.metric("Total Blocks", f"{total_blocks:,}")
    with col2:
        avg_cluster_size = np.mean([stats.size for stats in cluster_stats])
        st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
    with col3:
        total_organisms = len(set(org for stats in cluster_stats for org in stats.organisms))
        st.metric("Organisms Involved", total_organisms)
    with col4:
        large_clusters = len([s for s in cluster_stats if s.size >= 100])
        st.metric("Large Clusters (≥100)", large_clusters)
    
    # Filter controls
    st.subheader("🔍 Filter Clusters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        size_filter = st.selectbox(
            "Cluster Size",
            ["All", "Large (≥100)", "Medium (10-99)", "Small (<10)"],
            index=0
        )
    
    with col2:
        type_filter = st.selectbox(
            "Cluster Type", 
            ["All"] + list(set(stats.cluster_type for stats in cluster_stats)),
            index=0
        )
    
    with col3:
        organism_filter = st.selectbox(
            "Organism",
            ["All"] + sorted(set(org for stats in cluster_stats for org in stats.organisms)),
            index=0
        )
    
    # Apply filters
    filtered_stats = cluster_stats
    if size_filter != "All":
        if size_filter == "Large (≥100)":
            filtered_stats = [s for s in filtered_stats if s.size >= 100]
        elif size_filter == "Medium (10-99)":
            filtered_stats = [s for s in filtered_stats if 10 <= s.size < 100]
        elif size_filter == "Small (<10)":
            filtered_stats = [s for s in filtered_stats if s.size < 10]
    
    if type_filter != "All":
        filtered_stats = [s for s in filtered_stats if s.cluster_type == type_filter]
    
    if organism_filter != "All":
        filtered_stats = [s for s in filtered_stats if organism_filter in s.organisms]
    
    st.info(f"Showing {len(filtered_stats)} clusters")
    
    # Display clusters immediately with plots, generate AI summaries individually
    if filtered_stats:
        st.subheader("📋 Cluster Overview")
        
        # Display clusters in a grid
        for i in range(0, len(filtered_stats), 2):
            col1, col2 = st.columns(2)
            
            # First cluster card
            stats = filtered_stats[i]
            with col1:
                display_cluster_card_with_async_summary(stats)
            
            # Second cluster card (if exists)
            if i + 1 < len(filtered_stats):
                stats = filtered_stats[i + 1]
                with col2:
                    display_cluster_card_with_async_summary(stats)

def _merge_intervals(intervals: List[Dict], gap_bp: int = 1000) -> List[Dict]:
    """Merge overlapping/nearby intervals and aggregate block support.

    intervals: list of dicts with keys: start_bp, end_bp, block_id
    Returns merged list with keys: start_bp, end_bp, blocks (set)
    """
    if not intervals:
        return []
    # Sort by start
    sorted_ints = sorted(intervals, key=lambda x: (x["start_bp"], x["end_bp"]))
    merged = []
    curr = {
        "start_bp": sorted_ints[0]["start_bp"],
        "end_bp": sorted_ints[0]["end_bp"],
        "blocks": {sorted_ints[0]["block_id"]},
    }

    for iv in sorted_ints[1:]:
        if iv["start_bp"] <= curr["end_bp"] + gap_bp:
            curr["end_bp"] = max(curr["end_bp"], iv["end_bp"])
            curr["blocks"].add(iv["block_id"])
        else:
            merged.append(curr)
            curr = {"start_bp": iv["start_bp"], "end_bp": iv["end_bp"], "blocks": {iv["block_id"]}}
    merged.append(curr)
    return merged

def compute_display_regions_for_cluster(cluster_id: int, gap_bp: int = 1000, min_support: int = 1) -> List[Dict]:
    """Compute block-supported display regions for a cluster, robust to single-contig genomes.

    Returns list of dicts: {genome_id, contig_id, organism_name, start_bp, end_bp, support, blocks}
    """
    conn = get_database_connection()
    try:
        # Per-block per-role intervals (in bp) for this cluster
        per_block_query = """
            SELECT gbm.block_id, gbm.block_role, g.genome_id, g.contig_id,
                   MIN(g.start_pos) AS start_bp, MAX(g.end_pos) AS end_bp
            FROM gene_block_mappings gbm
            JOIN genes g ON gbm.gene_id = g.gene_id
            JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
            WHERE sb.cluster_id = ?
            GROUP BY gbm.block_id, gbm.block_role, g.genome_id, g.contig_id
        """
        per_block = pd.read_sql_query(per_block_query, conn, params=[int(cluster_id)])
        if per_block.empty:
            return []

        # Map genome_id -> organism_name
        genomes_df = pd.read_sql_query("SELECT genome_id, organism_name FROM genomes", conn)
        org_map = dict(zip(genomes_df["genome_id"], genomes_df["organism_name"]))

        regions = []
        # Group by (genome_id, contig_id)
        for (genome_id, contig_id), group in per_block.groupby(["genome_id", "contig_id"]):
            intervals = [
                {"start_bp": int(row.start_bp), "end_bp": int(row.end_bp), "block_id": int(row.block_id)}
                for _, row in group.iterrows()
            ]
            merged = _merge_intervals(intervals, gap_bp=gap_bp)
            for iv in merged:
                support = len(iv["blocks"])  # number of distinct blocks supporting
                if support >= min_support:
                    regions.append(
                        {
                            "genome_id": genome_id,
                            "contig_id": contig_id,
                            "organism_name": org_map.get(genome_id, "Unknown organism"),
                            "start_bp": iv["start_bp"],
                            "end_bp": iv["end_bp"],
                            "support": support,
                            "blocks": iv["blocks"],
                        }
                    )
        # Order deterministic: by genome_id, contig_id, start
        regions.sort(key=lambda r: (r["genome_id"], r["contig_id"], r["start_bp"]))
        return regions
    except Exception as e:
        logger.error(f"Error computing display regions: {e}")
        return []

def find_representative_block_for_cluster(stats):
    """Find a representative syntenic block for the cluster."""
    try:
        conn = sqlite3.connect("genome_browser.db")
        
        # Try to find a block matching the representative loci
        if stats.representative_query and stats.representative_target:
            # Search for blocks involving the representative loci
            block_query = """
                SELECT block_id, query_locus, target_locus, identity, score, length
                FROM syntenic_blocks 
                WHERE (query_locus LIKE ? OR target_locus LIKE ?)
                   OR (query_locus LIKE ? OR target_locus LIKE ?)
                ORDER BY score DESC 
                LIMIT 1
            """
            cursor = conn.execute(block_query, (
                f"%{stats.representative_query}%", f"%{stats.representative_query}%",
                f"%{stats.representative_target}%", f"%{stats.representative_target}%"
            ))
            block_row = cursor.fetchone()
            
            if block_row:
                # Create a block dict for the genome viewer
                selected_block = {
                    'block_id': block_row[0],
                    'query_locus': block_row[1],
                    'target_locus': block_row[2],
                    'identity': block_row[3],
                    'score': block_row[4],
                    'length': block_row[5]
                }
                conn.close()
                return selected_block
        
        # Fallback: find any high-scoring block
        fallback_query = """
            SELECT block_id, query_locus, target_locus, identity, score, length
            FROM syntenic_blocks 
            ORDER BY score DESC 
            LIMIT 1
            OFFSET ?
        """
        offset = stats.cluster_id % 100  # Vary by cluster
        cursor = conn.execute(fallback_query, (offset,))
        block_row = cursor.fetchone()
        
        if block_row:
            selected_block = {
                'block_id': block_row[0],
                'query_locus': block_row[1],
                'target_locus': block_row[2],
                'identity': block_row[3],
                'score': block_row[4],
                'length': block_row[5]
            }
            conn.close()
            return selected_block
            
        conn.close()
        return None
        
    except Exception as e:
        logger.error(f"Error finding representative block: {e}")
        return None

def create_domain_frequency_chart(domain_counts, cluster_id):
    """Create a mini domain frequency chart for cluster cards."""
    if not domain_counts:
        return None
    
    # Take top 8 domains for the mini chart
    top_domain_data = domain_counts[:8]
    domains = [item[0] for item in top_domain_data]
    counts = [item[1] for item in top_domain_data]
    
    # Reverse order so most common domains appear at the top
    domains.reverse()
    counts.reverse()
    
    # Create a horizontal bar chart with dark colors and actual counts
    fig = go.Figure(go.Bar(
        x=counts,
        y=domains,
        orientation='h',
        marker=dict(
            color='#2E5CAF',  # Dark blue for contrast
            line=dict(color='#1E3A6F', width=1)
        ),
        showlegend=False,
        text=counts,  # Show counts on bars
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=120, r=10, t=10, b=20),  # Increased left margin for bigger labels
        xaxis=dict(showticklabels=True, showgrid=True, gridcolor='lightgray', 
                   tickfont=dict(size=9, color='#333333')),
        yaxis=dict(tickfont=dict(size=12, color='#333333', family='Arial Black')),  # Bigger, bold labels
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def display_cluster_card_with_async_summary(stats):
    """Display a cluster card with immediate plot rendering and async AI summary."""
    with st.container():
        # Header
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h4 style="color: #2E5CAF; margin: 0 0 10px 0;">🧩 Cluster {stats.cluster_id}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Multi-dimensional stats summary
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Alignments:** {stats.total_alignments:,} blocks")
            st.write(f"**Genomic Scope:** {stats.unique_contigs} contigs, {stats.unique_genes:,} genes")
        with col2:
            st.write(f"**Length:** {stats.consensus_length} windows (avg)")
            st.write(f"**Identity:** {stats.avg_identity:.1%} ± {stats.diversity:.3f}")
            st.write(f"**Organisms:** {stats.organism_count} ({', '.join(stats.organisms[:2])}{'...' if len(stats.organisms) > 2 else ''})")
        
        # Domain frequency mini-chart (renders immediately)
        if hasattr(stats, 'domain_counts') and stats.domain_counts:
            st.markdown("**Top PFAM Domains:**")
            chart = create_domain_frequency_chart(stats.domain_counts, stats.cluster_id)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False}, key=f"domain_chart_{stats.cluster_id}")
        
        # AI Summary (generated on demand with button)
        summary_key = f"cluster_summary_{stats.cluster_id}"
        
        if summary_key not in st.session_state:
            st.session_state[summary_key] = {"generated": False, "content": ""}
        
        # Show button to generate AI summary or loading state
        loading_key = f"cluster_loading_{stats.cluster_id}"
        
        if not st.session_state[summary_key]["generated"]:
            if st.session_state.get(loading_key, False):
                # Show loading state and perform API call
                with st.status("🤖 Generating AI analysis...", expanded=False) as status:
                    try:
                        from cluster_analyzer import ClusterAnalyzer
                        analyzer = ClusterAnalyzer(Path("genome_browser.db"))
                        summary = analyzer.generate_cluster_summary(stats)
                        
                        # Update session state
                        st.session_state[summary_key] = {"generated": True, "content": summary}
                        st.session_state[loading_key] = False
                        
                        # Update status
                        status.update(label="✅ Analysis complete!", state="complete")
                        
                    except Exception as e:
                        st.session_state[summary_key] = {"generated": True, "content": f"Error generating summary: {e}"}
                        st.session_state[loading_key] = False
                        status.update(label="❌ Analysis failed!", state="error")
                        
                # Display content immediately after generation (whether success or error)
                if st.session_state[summary_key]["generated"]:
                    with st.expander("🤖 **AI Functional Summary**", expanded=True):
                        content = st.session_state[summary_key]["content"]
                        if content:
                            st.markdown(content)
                        else:
                            st.error("No content generated!")
            else:
                # Show button to start generation
                if st.button(f"🤖 Generate AI Analysis", key=f"ai_button_{stats.cluster_id}"):
                    st.session_state[loading_key] = True
                    st.rerun()
        else:
            # Show the generated summary (for when page reloads after generation)
            with st.expander("🤖 **AI Functional Summary**", expanded=True):
                content = st.session_state[summary_key]["content"]
                if content:
                    st.markdown(content)
                else:
                    st.error("No content in session state!")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📊 Explore Cluster {stats.cluster_id}", key=f"explore_{stats.cluster_id}", type="secondary"):
                with st.spinner("🔄 Loading cluster detail view..."):
                    # Go to dedicated cluster detail view with genome diagrams
                    st.session_state.selected_cluster = stats.cluster_id
                    st.session_state.current_page = 'cluster_detail'
                    st.success(f"✅ Navigating to Cluster {stats.cluster_id} detail view...")
                    st.rerun()
        
        with col2:
            if st.button(f"🧬 View Representative", key=f"repr_{stats.cluster_id}"):
                # Find a representative block ID and jump to genome viewer
                # This is simplified - would need to query the actual block
                st.info(f"Representative: {stats.representative_query} ↔ {stats.representative_target}")

def display_cluster_card(stats, summary):
    """Legacy function - Display a single cluster card with pre-generated summary."""
    with st.container():
        # Header
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h4 style="color: #2E5CAF; margin: 0 0 10px 0;">🧩 Cluster {stats.cluster_id}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Multi-dimensional stats summary
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Alignments:** {stats.total_alignments:,} blocks")
            st.write(f"**Genomic Scope:** {stats.unique_contigs} contigs, {stats.unique_genes:,} genes")
        with col2:
            st.write(f"**Length:** {stats.consensus_length} windows (avg)")
            st.write(f"**Identity:** {stats.avg_identity:.1%} ± {stats.diversity:.3f}")
            st.write(f"**Organisms:** {stats.organism_count} ({', '.join(stats.organisms[:2])}{'...' if len(stats.organisms) > 2 else ''})")
        
        # Domain frequency mini-chart
        if hasattr(stats, 'domain_counts') and stats.domain_counts:
            st.markdown("**Top PFAM Domains:**")
            chart = create_domain_frequency_chart(stats.domain_counts, stats.cluster_id)
            if chart:
                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False}, key=f"domain_chart_{stats.cluster_id}")
        
        # AI Summary
        with st.expander("🤖 **AI Functional Summary**", expanded=False):
            st.markdown(summary)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📊 Explore Cluster {stats.cluster_id}", key=f"explore_{stats.cluster_id}", type="secondary"):
                with st.spinner("🔄 Loading cluster detail view..."):
                    # Go to dedicated cluster detail view with genome diagrams
                    st.session_state.selected_cluster = stats.cluster_id
                    st.session_state.current_page = 'cluster_detail'
                    st.success(f"✅ Navigating to Cluster {stats.cluster_id} detail view...")
                    st.rerun()
        
        with col2:
            if st.button(f"🧬 View Representative", key=f"repr_{stats.cluster_id}"):
                # Find a representative block ID and jump to genome viewer
                # This is simplified - would need to query the actual block
                st.info(f"Representative: {stats.representative_query} ↔ {stats.representative_target}")

@st.cache_data
def load_cluster_blocks(cluster_id: int) -> pd.DataFrame:
    """Load all syntenic blocks belonging to a specific cluster."""
    conn = get_database_connection()
    
    # Primary query: Use cluster_id column directly
    query = """
        SELECT sb.*, 
               g1.organism_name as query_organism,
               g2.organism_name as target_organism
        FROM syntenic_blocks sb
        LEFT JOIN genomes g1 ON sb.query_genome_id = g1.genome_id
        LEFT JOIN genomes g2 ON sb.target_genome_id = g2.genome_id
        WHERE sb.cluster_id = ?
        ORDER BY sb.score DESC, sb.block_id
    """
    
    result = pd.read_sql_query(query, conn, params=[cluster_id])
    
    # Fallback query: Use cluster_assignments table if cluster_id column is empty
    if result.empty and cluster_id > 0:
        fallback_query = """
            SELECT sb.*, 
                   g1.organism_name as query_organism,
                   g2.organism_name as target_organism
            FROM syntenic_blocks sb
            LEFT JOIN genomes g1 ON sb.query_genome_id = g1.genome_id
            LEFT JOIN genomes g2 ON sb.target_genome_id = g2.genome_id
            INNER JOIN cluster_assignments ca ON sb.block_id = ca.block_id
            WHERE ca.cluster_id = ?
            ORDER BY sb.score DESC, sb.block_id
        """
        result = pd.read_sql_query(fallback_query, conn, params=[cluster_id])
    
    return result

def extract_unique_loci_from_cluster(cluster_blocks: pd.DataFrame) -> List[str]:
    """Extract all unique loci involved in a cluster."""
    all_loci = set()
    
    for _, block in cluster_blocks.iterrows():
        all_loci.add(block['query_locus'])
        all_loci.add(block['target_locus'])
    
    return sorted(list(all_loci))

def display_cluster_detail():
    """Display detailed view of a specific cluster with genome diagrams for each locus."""
    if 'selected_cluster' not in st.session_state:
        st.error("❌ No cluster selected")
        if st.button("← Back to Cluster Explorer"):
            st.session_state.current_page = 'cluster_explorer'
            st.rerun()
        return
    
    cluster_id = st.session_state.selected_cluster
    
    # Header with navigation breadcrumb
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"🧩 Cluster {cluster_id} - Genome Diagrams View")
        st.caption("🔍 Cluster Explorer → Cluster Detail")
    with col2:
        if st.button("← Back to Cluster Explorer", type="primary"):
            st.session_state.current_page = 'cluster_explorer'
            if 'selected_cluster' in st.session_state:
                del st.session_state.selected_cluster
            st.rerun()
    
    # Load cluster blocks and display regions
    with st.spinner("Loading cluster data..."):
        try:
            # Load blocks in this cluster
            cluster_blocks = load_cluster_blocks(cluster_id)
            
            if cluster_blocks.empty:
                if cluster_id == 0:
                    st.warning(f"🔄 Cluster 0 is the **sink cluster** containing non-robust or singleton blocks.")
                    st.info("💡 Try exploring a cluster with ID > 0 for meaningful syntenic relationships.")
                else:
                    st.error(f"❌ No blocks found for cluster {cluster_id}")
                st.info("This could mean:")
                st.write("• The clustering analysis hasn't been run yet")
                st.write("• The cluster ID doesn't exist")  
                st.write("• Database connectivity issues")
                return
            
            # Compute display regions; if unavailable, fall back to unique loci
            regions = compute_display_regions_for_cluster(cluster_id, gap_bp=1000, min_support=1)
            use_regions = len(regions) > 0
            if not use_regions:
                unique_loci = extract_unique_loci_from_cluster(cluster_blocks)
            
            # Load cluster stats if available
            try:
                from cluster_analyzer import ClusterAnalyzer
                analyzer = ClusterAnalyzer(Path("genome_browser.db"))
                stats = analyzer.get_cluster_stats(cluster_id)
            except:
                stats = None
                
        except Exception as e:
            st.error(f"Error loading cluster data: {e}")
            return
    
    # Cluster overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Blocks", f"{len(cluster_blocks):,}")
    with col2:
        if 'regions' in locals() and regions:
            st.metric("🧬 Display Regions", f"{len(regions):,}")
        else:
            st.metric("🧬 Unique Loci", f"{len(unique_loci):,}")
    with col3:
        avg_identity = cluster_blocks['identity'].mean()
        st.metric("📈 Avg Identity", f"{avg_identity:.1%}")
    with col4:
        avg_length = cluster_blocks['length'].mean()
        st.metric("📏 Avg Length", f"{avg_length:.1f}")
    
    # Display summary info
    if stats:
        st.info(f"**Organisms:** {', '.join(stats.organisms[:3])}{'...' if len(stats.organisms) > 3 else ''}")
    
    # Show cluster composition table
    with st.expander("📋 **Cluster Block Details**", expanded=False):
        display_cols = ['block_id', 'query_locus', 'target_locus', 'length', 'identity', 'score']
        display_df = cluster_blocks[display_cols].copy()
        display_df['identity'] = display_df['identity'].apply(lambda x: f"{x:.3f}")
        display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # AI Summary (if available)
    if stats:
        detail_summary_key = f"cluster_detail_summary_{cluster_id}"
        
        if detail_summary_key not in st.session_state:
            st.session_state[detail_summary_key] = {"generated": False, "content": ""}
        
        detail_loading_key = f"cluster_detail_loading_{cluster_id}"
        
        with st.expander("🤖 **AI Functional Analysis**", expanded=False):
            if not st.session_state[detail_summary_key]["generated"]:
                if st.session_state.get(detail_loading_key, False):
                    with st.status("🤖 Generating analysis...", expanded=False) as status:
                        try:
                            summary = get_cluster_analyzer().generate_cluster_summary(stats)
                            st.session_state[detail_summary_key] = {"generated": True, "content": summary}
                            st.session_state[detail_loading_key] = False
                            status.update(label="✅ Analysis complete!", state="complete")
                            st.rerun()
                        except Exception as e:
                            st.session_state[detail_summary_key] = {"generated": True, "content": f"Error: {e}"}
                            st.session_state[detail_loading_key] = False
                            status.update(label="❌ Analysis failed!", state="error")
                else:
                    if st.button("🤖 Generate AI Analysis", key=f"ai_detail_button_{cluster_id}"):
                        st.session_state[detail_loading_key] = True
                        st.rerun()
            else:
                st.markdown(st.session_state[detail_summary_key]["content"])
    
    # Main section: Genome diagrams for each locus
    st.divider()
    if use_regions:
        st.subheader("🧬 Genome Diagrams by Display Region")
        st.markdown(f"*Showing genome diagrams for all {len(regions)} block-supported regions in this cluster*")
    else:
        st.subheader("🧬 Genome Diagrams by Locus")
        st.markdown(f"*Showing genome diagrams for all {len(unique_loci)} loci involved in this cluster*")
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        show_extended_context = st.checkbox("Show extended context", value=False, 
                                           help="Show additional flanking genes beyond the syntenic region")
    with col2:
        max_loci_per_page = st.selectbox("Loci per page", options=[5, 10, 20, 50], index=1,
                                        help="Number of loci to display per page")
    
    # Pagination for loci
    total_items = len(regions) if use_regions else len(unique_loci)
    total_pages = math.ceil(total_items / max_loci_per_page)
    if total_pages > 1:
        current_page = st.number_input(f"Page (1-{total_pages})", 
                                     min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (current_page - 1) * max_loci_per_page
        end_idx = min(start_idx + max_loci_per_page, total_items)
        if use_regions:
            page_regions = regions[start_idx:end_idx]
            st.info(f"Showing regions {start_idx + 1}-{end_idx} of {total_items}")
        else:
            page_loci = unique_loci[start_idx:end_idx]
            st.info(f"Showing loci {start_idx + 1}-{end_idx} of {total_items}")
    else:
        if use_regions:
            page_regions = regions
        else:
            page_loci = unique_loci
    
    # Display genome diagram for each locus
    if use_regions:
        items_iter = enumerate(page_regions)
    else:
        items_iter = enumerate(page_loci)

    for i, item in items_iter:
        st.markdown("---")
        if use_regions:
            region = item
            # Header
            org = region.get("organism_name", "Unknown organism")
            contig_id = region["contig_id"]
            genome_id = region["genome_id"]
            start_bp = region["start_bp"]
            end_bp = region["end_bp"]
            support = region["support"]
            display_title = f"{genome_id}:{contig_id} [{start_bp:,}-{end_bp:,}]"
            st.subheader(f"🧬 Region {i+1}: {display_title} ({org})")

            # Determine representative block among supporting blocks
            rep_block = None
            if region.get("blocks"):
                blocks_subset = cluster_blocks[cluster_blocks["block_id"].isin(list(region["blocks"]))]
                if not blocks_subset.empty:
                    rep_block = blocks_subset.loc[blocks_subset["score"].idxmax()]

            if rep_block is not None:
                st.caption(
                    f"Supported by {support} block(s) | Representative block: {int(rep_block['block_id'])} (score: {rep_block['score']:.1f})"
                )
            else:
                st.caption(f"Supported by {support} block(s)")
        else:
            locus_id = item
            display_title = locus_id
            # Locus header with organism info
            organism_info = ""
            sample_blocks = cluster_blocks[
                (cluster_blocks['query_locus'] == locus_id) | 
                (cluster_blocks['target_locus'] == locus_id)
            ]
            if not sample_blocks.empty:
                first_block = sample_blocks.iloc[0]
                if first_block['query_locus'] == locus_id:
                    organism_info = f" ({first_block.get('query_organism', 'Unknown organism')})"
                else:
                    organism_info = f" ({first_block.get('target_organism', 'Unknown organism')})"
            st.subheader(f"🧬 Locus {i+1}: {locus_id}{organism_info}")
            # Count blocks involving this locus
            involving_blocks = len(sample_blocks)
            if involving_blocks > 0:
                best_block = sample_blocks.loc[sample_blocks['score'].idxmax()]
                st.caption(f"Participates in {involving_blocks} syntenic block(s) | Representative block: {best_block['block_id']} (score: {best_block['score']:.1f})")
            else:
                st.caption(f"No syntenic blocks found for this locus")
        
        # Load genes for this locus
        with st.spinner("Loading genes..."):
            try:
                if use_regions:
                    genes_df = load_genes_for_region(contig_id, start_bp, end_bp, extended_context=show_extended_context)
                    st.caption(f"🎯 Showing block-supported region ({len(genes_df)} genes)")
                else:
                    # Original behavior (per-locus)
                    representative_block = None
                    locus_role = None
                    sample_blocks = cluster_blocks[
                        (cluster_blocks['query_locus'] == locus_id) | 
                        (cluster_blocks['target_locus'] == locus_id)
                    ]
                    if not sample_blocks.empty:
                        best_block = sample_blocks.loc[sample_blocks['score'].idxmax()]
                        if best_block['query_locus'] == locus_id:
                            locus_role = 'query'
                            representative_block = best_block
                        elif best_block['target_locus'] == locus_id:
                            locus_role = 'target'
                            representative_block = best_block
                    if representative_block is not None and locus_role is not None:
                        genes_df = load_genes_for_locus(locus_id, representative_block['block_id'], locus_role)
                        st.caption(f"🎯 Showing focused syntenic region ({len(genes_df)} genes)")
                    else:
                        # Fallback paths
                        fallback_query = """
                            SELECT block_id, query_locus, target_locus, score
                            FROM syntenic_blocks 
                            WHERE query_locus = ? OR target_locus = ?
                            ORDER BY score DESC 
                            LIMIT 1
                        """
                        conn = get_database_connection()
                        fallback_result = pd.read_sql_query(fallback_query, conn, params=[locus_id, locus_id])
                        if not fallback_result.empty:
                            fallback_block = fallback_result.iloc[0]
                            fallback_role = 'query' if fallback_block['query_locus'] == locus_id else 'target'
                            genes_df = load_genes_for_locus(locus_id, fallback_block['block_id'], fallback_role)
                            st.caption(f"🎯 Using fallback syntenic block {fallback_block['block_id']} ({len(genes_df)} genes)")
                            st.info("ℹ️ This locus doesn't participate in blocks within this cluster, showing representative syntenic region from another cluster")
                        else:
                            genes_df = load_genes_for_locus(locus_id)
                            if len(genes_df) > 50:
                                mid_point = len(genes_df) // 2
                                s_idx = max(0, mid_point - 25)
                                e_idx = min(len(genes_df), mid_point + 25)
                                genes_df = genes_df.iloc[s_idx:e_idx].copy()
                                st.caption(f"⚠️ Showing center region of locus ({len(genes_df)} genes)")
                                st.warning("No syntenic blocks found for this locus - showing representative center region")
                            else:
                                st.caption(f"⚠️ Showing entire small locus ({len(genes_df)} genes)")
                                st.warning("No syntenic blocks found for this locus")
                
                if genes_df.empty:
                    st.warning(f"No genes found for {display_title}")
                    continue
                
            except Exception as e:
                st.error(f"Error loading genes for {display_title}: {e}")
                continue
        
        # Create and display genome diagram
        try:
            with st.spinner("Generating genome diagram..."):
                title = f"{display_title} - Cluster {cluster_id}"
                fig = create_genome_diagram(genes_df, title, width=1000, height=300)
                st.plotly_chart(fig, use_container_width=True, key=f"genome_{cluster_id}_{i}")
            
            # Expandable gene annotation table
            with st.expander(f"📋 **Gene Annotations for {display_title}**", expanded=False):
                st.caption(f"Showing {len(genes_df)} genes")
                
                # Prepare display columns
                display_cols = ['gene_id', 'start_pos', 'end_pos', 'strand', 'pfam_domains']
                if 'position_in_block' in genes_df.columns:
                    display_cols.insert(-1, 'position_in_block')
                
                annotation_df = genes_df[display_cols].copy()
                
                # Truncate long PFAM domain lists for display
                if 'pfam_domains' in annotation_df.columns:
                    annotation_df['pfam_domains'] = annotation_df['pfam_domains'].apply(
                        lambda x: x[:100] + '...' if isinstance(x, str) and len(x) > 100 else x
                    )
                
                # Color-code by syntenic role if available
                if 'position_in_block' in annotation_df.columns:
                    def highlight_syntenic_role(row):
                        role = row.get('position_in_block', '')
                        if role == 'Conserved Block':
                            return ['background-color: #ffebee; color: #c62828'] * len(row)  # Light red
                        elif role == 'Block Edge':
                            return ['background-color: #fff3e0; color: #ef6c00'] * len(row)  # Light orange
                        elif role == 'Flanking Region':
                            return ['background-color: #e3f2fd; color: #1565c0'] * len(row)  # Light blue
                        return [''] * len(row)
                    
                    styled_df = annotation_df.style.apply(highlight_syntenic_role, axis=1)
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                else:
                    st.dataframe(annotation_df, hide_index=True, use_container_width=True)
                
                # Additional gene statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    forward_genes = len(genes_df[genes_df['strand'] == '+'])
                    st.metric("➡️ Forward strand", forward_genes)
                with col2:
                    reverse_genes = len(genes_df[genes_df['strand'] == '-'])
                    st.metric("⬅️ Reverse strand", reverse_genes)
                with col3:
                    annotated_genes = len(genes_df[genes_df['pfam_domains'].notna() & (genes_df['pfam_domains'] != '')])
                    st.metric("🏷️ PFAM annotated", annotated_genes)
        
        except Exception as e:
            st.error(f"Error creating genome diagram for {display_title}: {e}")
            logger.error(f"Genome diagram error for {display_title}: {e}")
    
    # Navigation footer
    if total_pages > 1:
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if use_regions:
                st.info(f"Page {current_page} of {total_pages} | {len(regions)} total regions in cluster")
            else:
                st.info(f"Page {current_page} of {total_pages} | {len(unique_loci)} total loci in cluster")

def main():
    """Main Streamlit application."""
    # Title and navigation
    st.title("🧬 ELSA Genome Browser")
    st.markdown("*Syntenic Block Explorer with PFAM Domain Annotations*")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    # Debug session state (can be removed later)
    if st.sidebar.checkbox("🔧 Debug Mode", value=False):
        st.sidebar.write("**Session State:**")
        st.sidebar.json({
            "current_page": st.session_state.get('current_page'),
            "selected_cluster": st.session_state.get('selected_cluster'),
            "selected_block": bool(st.session_state.get('selected_block'))
        })
    
    # Handle special pages that override tab behavior
    if st.session_state.get('current_page') == 'cluster_detail':
        display_cluster_detail()
        return
    elif st.session_state.get('current_page') == 'genome_viewer' and 'selected_block' in st.session_state:
        display_genome_viewer()
        return
    
    # Standard tab navigation for main pages
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Block Explorer", "🧬 Genome Viewer", "🧩 Cluster Explorer"])
    
    with tab1:
        st.session_state.current_page = 'dashboard'
        display_dashboard()
    
    with tab2:
        st.session_state.current_page = 'block_explorer'
        filters = create_sidebar_filters()
        display_block_explorer(filters)
    
    with tab3:
        st.session_state.current_page = 'genome_viewer'
        display_genome_viewer()
    
    with tab4:
        st.session_state.current_page = 'cluster_explorer'
        display_cluster_explorer()
    
    # Optional fifth tab: clustering tuner
    if st.sidebar.checkbox("🛠️ Show Clustering Tuner", value=False, help="Tune clustering parameters and re-cluster without re-running all-vs-all"):
        st.session_state.current_page = 'clustering_tuner'
        display_clustering_tuner()


def display_clustering_tuner():
    """Interactive interface to re-run clustering only with adjustable parameters.
    This uses existing syntenic_blocks.csv and window embeddings to recompute cluster assignments,
    then updates the genome browser database and reloads the app.
    """
    st.header("🛠️ Clustering Tuner")
    st.markdown("Tune clustering parameters and re-run clustering without re-doing all-vs-all.")

    # Locate analysis outputs (app runs from genome_browser/ by default)
    default_blocks = Path("../syntenic_analysis/syntenic_blocks.csv")
    default_config = Path("../elsa.config.yaml")

    col1, col2 = st.columns(2)
    with col1:
        blocks_csv = st.text_input("Path to syntenic_blocks.csv", value=str(default_blocks))
        config_yaml = st.text_input("Path to ELSA config (to find windows parquet)", value=str(default_config))
    with col2:
        db_path = st.text_input("Genome browser DB path", value=str(DB_PATH))
        method = st.selectbox("Clustering method", ["mutual_jaccard"], index=0)
        windows_override = st.text_input("Windows parquet override (optional)", value="", help="Full path to *windows*.parquet if manifest/work_dir lookup fails")

    st.subheader("Parameters")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        jaccard_tau = st.number_input("jaccard_tau", value=0.75, min_value=0.0, max_value=1.0, step=0.05)
        df_max = st.number_input("df_max (max DF for shingles)", value=30, min_value=1, step=1)
        min_low_df_anchors = st.number_input("min_low_df_anchors", value=3, min_value=0, step=1)
        idf_mean_min = st.number_input("idf_mean_min", value=1.0, min_value=0.0, step=0.1)
        mutual_k = st.number_input("mutual_k", value=3, min_value=1, step=1)
        max_df_percentile = st.text_input("max_df_percentile (optional, e.g., 0.9)", value="")
    with pcol2:
        v_mad_max_genes = st.number_input("v_mad_max_genes", value=0.5, min_value=0.0, step=0.1)
        enable_cassette_mode = st.checkbox("enable_cassette_mode", value=True)
        cassette_max_len = st.number_input("cassette_max_len", value=4, min_value=2, step=1)
        degree_cap = st.number_input("degree_cap (per-node top edges)", value=10, min_value=0, step=1)
        k_core_min_degree = st.number_input("k_core_min_degree", value=3, min_value=0, step=1)
        triangle_support_min = st.number_input("triangle_support_min", value=1, min_value=0, step=1)
    with pcol3:
        use_weighted_jaccard = st.checkbox("use_weighted_jaccard", value=True)
        use_community_detection = st.checkbox("use_community_detection", value=True)
        community_method = st.selectbox("community_method", ["greedy"], index=0)
        shingle_k = st.number_input("shingle_k (SRP k-gram)", value=3, min_value=2, step=1)
        srp_bits = st.number_input("srp_bits", value=256, min_value=32, step=32)
        srp_bands = st.number_input("srp_bands", value=32, min_value=1, step=1)

    st.subheader("Expansion (optional)")
    e1, e2, e3 = st.columns(3)
    with e1:
        enable_expansion = st.checkbox("Enable expansion", value=True)
        tau_expand = st.number_input("tau_expand (medoid)", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
        tau_peer = st.number_input("tau_peer", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
    with e2:
        m_peers = st.number_input("min peer support", value=2, min_value=0, step=1)
        M_expand = st.number_input("min low-DF anchors", value=2, min_value=0, step=1)
        max_added_per_cluster = st.number_input("max add/cluster", value=10, min_value=0, step=1)
    with e3:
        allow_degree_relax = st.checkbox("Relax degree cap for adds", value=True)
        waves = st.number_input("Expansion waves", value=1, min_value=1, step=1)

    if st.button("Re-run clustering", type="primary"):
        try:
            # Validate inputs
            blocks_path = Path(blocks_csv)
            if not blocks_path.exists():
                st.error(f"Blocks CSV not found: {blocks_path}")
                return
            cfg_path = Path(config_yaml)
            if not cfg_path.exists():
                st.error(f"Config file not found: {cfg_path}")
                return

            # Build config namespace for clustering
            cfg = SimpleNamespace(
                jaccard_tau=float(jaccard_tau),
                df_max=int(df_max),
                min_low_df_anchors=int(min_low_df_anchors),
                idf_mean_min=float(idf_mean_min),
                mutual_k=int(mutual_k),
                max_df_percentile=(float(max_df_percentile) if max_df_percentile.strip() else None),
                v_mad_max_genes=float(v_mad_max_genes),
                enable_cassette_mode=bool(enable_cassette_mode),
                cassette_max_len=int(cassette_max_len),
                degree_cap=int(degree_cap),
                k_core_min_degree=int(k_core_min_degree),
                triangle_support_min=int(triangle_support_min),
                use_weighted_jaccard=bool(use_weighted_jaccard),
                use_community_detection=bool(use_community_detection),
                community_method=community_method,
                srp_bits=int(srp_bits), srp_bands=int(srp_bands), srp_band_bits=8, srp_seed=1337,
                shingle_k=int(shingle_k),
                min_anchors=4, min_span_genes=8,
                size_ratio_min=0.5, size_ratio_max=2.0,
                keep_singletons=False, sink_label=0
            )

            with st.status("Re-clustering blocks...", expanded=True) as status:
                st.write("Loading blocks and window embeddings...")
                logger.info("[TUNER] Loading blocks CSV: %s", blocks_path)
                blocks = _load_blocks_from_csv(blocks_path)
                st.write(f"✓ Loaded {len(blocks)} blocks from CSV")
                logger.info("[TUNER] Loaded %d blocks", len(blocks))

                # Stitch adjacent blocks to recover split operons/cassettes
                pre_n = len(blocks)
                blocks = _stitch_blocks(blocks, max_gap=1)
                st.write(f"✓ Stitching: {pre_n} → {len(blocks)} blocks after merging adjacents")
                logger.info("[TUNER] Stitching reduced blocks: %d -> %d", pre_n, len(blocks))

                override_path = Path(windows_override) if windows_override.strip() else None
                window_lookup, lookup_meta = _create_window_lookup_from_config(cfg_path, override_path)
                if window_lookup is None:
                    st.error("Failed to create window embedding lookup. Check the windows parquet path.")
                    return
                st.write(f"✓ Windows parquet: {lookup_meta.get('path','?')} | rows={lookup_meta.get('n_windows',0)} | emb_dim={lookup_meta.get('emb_dim',0)}")
                logger.info("[TUNER] Windows parquet: %s rows=%s emb_dim=%s", lookup_meta.get('path'), lookup_meta.get('n_windows'), lookup_meta.get('emb_dim'))

                # Quick coverage check over a sample of window IDs from blocks
                sample_ids = []
                for b in blocks:
                    sample_ids.extend(b.query_windows[:2])
                    sample_ids.extend(b.target_windows[:2])
                    if len(sample_ids) >= 400:
                        break
                found = sum(1 for wid in sample_ids if window_lookup(wid) is not None)
                st.write(f"Coverage check: {found}/{len(sample_ids)} sampled window IDs found in embeddings")
                logger.info("[TUNER] Coverage check: %d/%d", found, len(sample_ids))

                st.write(f"Clustering {len(blocks)} blocks with updated parameters...")
                from elsa.analyze.cluster_mutual_jaccard import cluster_blocks_jaccard
                t0 = time.time()
                assignments = cluster_blocks_jaccard(blocks, window_lookup, cfg)
                dt = time.time() - t0
                st.write(f"✓ Clustering finished in {dt:.2f}s")
                logger.info("[TUNER] clustering finished in %.2fs", dt)

                if not assignments:
                    st.error("Re-clustering produced no assignments.")
                    return

                from collections import Counter
                ctr = Counter([cl for cl in assignments.values() if cl and cl > 0])
                st.write(f"Found {len(ctr)} clusters (non-sink). Top sizes: {sorted(ctr.values(), reverse=True)[:10]}")
                logger.info("[TUNER] clusters=%d top_sizes=%s", len(ctr), sorted(ctr.values(), reverse=True)[:10])

                # Optional expansion
                if enable_expansion:
                    st.write("Running expansion phase...")
                    t1 = time.time()
                    assignments = _expand_clusters(
                        blocks, assignments, window_lookup,
                        df_max=int(df_max), shingle_k=int(shingle_k),
                        srp_bits=int(srp_bits), srp_bands=int(srp_bands), srp_band_bits=8, srp_seed=1337,
                        tau_expand=float(tau_expand), tau_peer=float(tau_peer),
                        m_peers=int(m_peers), M_expand=int(M_expand),
                        max_added_per_cluster=int(max_added_per_cluster),
                        allow_degree_relax=bool(allow_degree_relax), waves=int(waves)
                    )
                    dt1 = time.time() - t1
                    ctr2 = Counter([cl for cl in assignments.values() if cl and cl > 0])
                    st.write(f"✓ Expansion finished in {dt1:.2f}s → clusters: {len(ctr2)} (Top sizes: {sorted(ctr2.values(), reverse=True)[:10]})")
                    logger.info("[TUNER] expansion finished in %.2fs clusters=%d", dt1, len(ctr2))

                st.write("Updating genome browser database with new cluster assignments...")
                _apply_cluster_assignments_to_db(assignments, Path(db_path))

                status.update(label="✅ Re-clustering complete", state="complete")
                st.success("Clusters updated. Reloading app...")
                st.rerun()
        except Exception as e:
            st.error(f"Re-clustering failed: {e}")
            logger.exception("Re-clustering error")


def _load_blocks_from_csv(blocks_csv_path: Path):
    """Load blocks from syntenic_blocks.csv and construct minimal block objects suitable for clustering.
    Reconstruct matches by pairing query/target window IDs in order.
    """
    import pandas as pd
    df = pd.read_csv(blocks_csv_path)
    blocks = []

    class _Match:
        __slots__ = ("query_window_id", "target_window_id")
        def __init__(self, q, t):
            self.query_window_id = q
            self.target_window_id = t

    class _Block:
        def __init__(self, idx, row):
            self.id = int(row['block_id']) if 'block_id' in row else idx
            qws = str(row.get('query_windows_json', '') or '')
            tws = str(row.get('target_windows_json', '') or '')
            self.query_windows = [w for w in qws.split(';') if w]
            self.target_windows = [w for w in tws.split(';') if w]
            n = min(len(self.query_windows), len(self.target_windows))
            self.matches = [_Match(self.query_windows[i], self.target_windows[i]) for i in range(n)]
            self.alignment_length = n
            self.identity = float(row.get('identity', 0.0) or 0.0)
            self.chain_score = float(row.get('score', 0.0) or 0.0)
            # Optional strand if available
            self.strand = 1
            # For stitching
            self.query_locus = row.get('query_locus', '')
            self.target_locus = row.get('target_locus', '')
            self.q_start = int(row.get('query_window_start')) if not pd.isna(row.get('query_window_start')) else None
            self.q_end = int(row.get('query_window_end')) if not pd.isna(row.get('query_window_end')) else None
            self.t_start = int(row.get('target_window_start')) if not pd.isna(row.get('target_window_start')) else None
            self.t_end = int(row.get('target_window_end')) if not pd.isna(row.get('target_window_end')) else None

    for idx, row in df.iterrows():
        blocks.append(_Block(idx, row))
    return blocks


def _stitch_blocks(blocks: List, max_gap: int = 1) -> List:
    """Stitch adjacent blocks from the same query/target locus pair if window ranges are contiguous/overlapping.

    Args:
        blocks: list of _Block objects loaded from CSV
        max_gap: maximum allowed gap (in window indices) between blocks to stitch
    Returns:
        New list of stitched _Block objects
    """
    # Group by (query_locus, target_locus)
    from collections import defaultdict
    groups = defaultdict(list)
    for b in blocks:
        key = (b.query_locus, b.target_locus)
        groups[key].append(b)

    stitched = []
    for key, blist in groups.items():
        # Sort by query start (fallback to window index from first query window)
        def start_key(b):
            return b.q_start if b.q_start is not None else (int(b.query_windows[0].split('_')[-1]) if b.query_windows else 0)
        blist.sort(key=start_key)

        current = None
        for b in blist:
            if current is None:
                current = b
                continue
            # Determine adjacency on both query and target ranges
            q_adj = (current.q_end is not None and b.q_start is not None and b.q_start <= (current.q_end + max_gap))
            t_adj = (current.t_end is not None and b.t_start is not None and b.t_start <= (current.t_end + max_gap))
            if q_adj and t_adj:
                # Stitch: merge windows and ranges
                current.query_windows = list(dict.fromkeys(current.query_windows + b.query_windows))
                current.target_windows = list(dict.fromkeys(current.target_windows + b.target_windows))
                n = min(len(current.query_windows), len(current.target_windows))
                current.matches = [
                    _Match(current.query_windows[i], current.target_windows[i]) for i in range(n)
                ]
                current.alignment_length = n
                # Update ranges
                if current.q_start is not None and b.q_start is not None:
                    current.q_start = min(current.q_start, b.q_start)
                else:
                    current.q_start = current.q_start or b.q_start
                if current.q_end is not None and b.q_end is not None:
                    current.q_end = max(current.q_end, b.q_end)
                else:
                    current.q_end = current.q_end or b.q_end
                if current.t_start is not None and b.t_start is not None:
                    current.t_start = min(current.t_start, b.t_start)
                else:
                    current.t_start = current.t_start or b.t_start
                if current.t_end is not None and b.t_end is not None:
                    current.t_end = max(current.t_end, b.t_end)
                else:
                    current.t_end = current.t_end or b.t_end
                # Combine scores conservatively (take max)
                current.chain_score = max(current.chain_score, b.chain_score)
                current.identity = max(current.identity, b.identity)
            else:
                stitched.append(current)
                current = b
        if current is not None:
            stitched.append(current)

    # Preserve blocks in groups with missing locus info
    anon = [b for b in blocks if (b.query_locus, b.target_locus) not in groups]
    return stitched + anon


def _create_window_lookup_from_config(config_path: Path, windows_override: Optional[Path] = None):
    """Create a window embedding lookup from ELSA config and manifest.
    Returns a callable window_id -> np.ndarray or None on failure.
    """
    try:
        from elsa.params import load_config
        from elsa.manifest import ELSAManifest
        import pandas as pd
        import numpy as np

        # Use explicit path only; default to ../elsa_index/shingles/windows.parquet relative to repo.
        _default = Path(__file__).resolve().parent.parent / "elsa_index/shingles/windows.parquet"
        windows_path = Path(windows_override) if windows_override else _default
        if not windows_path.exists():
            logger.error(f"Windows parquet not found at: {windows_path}")
            return None
        windows_df = pd.read_parquet(windows_path)
        emb_cols = [c for c in windows_df.columns if c.startswith('emb_')]
        lookup = {}
        for _, row in windows_df.iterrows():
            win_id = f"{row['sample_id']}_{row['locus_id']}_{int(row['window_idx'])}"
            lookup[win_id] = np.array([row[c] for c in emb_cols])

        def _lookup_fn(window_id: str):
            return lookup.get(window_id)

        meta = {"path": str(windows_path), "n_windows": len(windows_df), "emb_dim": len(emb_cols)}
        return _lookup_fn, meta
    except Exception as e:
        logger.error(f"Failed to create window lookup: {e}")
        return None


def _apply_cluster_assignments_to_db(assignments: Dict[int, int], db_path: Path):
    """Apply new cluster assignments to the genome_browser.db without re-ingesting all data."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Reset cluster assignments and clusters
        cur.execute("DELETE FROM cluster_assignments")
        cur.execute("DELETE FROM clusters")

        # Build cluster sizes
        cluster_sizes: Dict[int, int] = {}
        for block_id, cl in assignments.items():
            if cl is None or cl == 0:
                continue
            cluster_sizes[cl] = cluster_sizes.get(cl, 0) + 1

        # Insert clusters with minimal info
        for cl, size in sorted(cluster_sizes.items()):
            cur.execute(
                """
                INSERT INTO clusters (cluster_id, size, consensus_length, consensus_score, diversity,
                                      representative_query, representative_target, cluster_type)
                VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, 'unknown')
                """,
                (int(cl), int(size))
            )

        # Insert cluster assignments
        rows = [(int(bid), int(cl)) for bid, cl in assignments.items() if cl and cl > 0]
        if rows:
            cur.executemany("INSERT INTO cluster_assignments (block_id, cluster_id) VALUES (?, ?)", rows)

        # Update syntenic_blocks.cluster_id based on assignments
        cur.execute(
            """
            UPDATE syntenic_blocks
            SET cluster_id = COALESCE((SELECT ca.cluster_id FROM cluster_assignments ca WHERE ca.block_id = syntenic_blocks.block_id), 0)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _build_comparative_json(block_id: int, query_genes_df: pd.DataFrame, target_genes_df: pd.DataFrame) -> Dict:
    """Build clinker-like JSON for comparative view using DB gene_block_mappings order.
    Links are derived by relative_position order within the block (no edit distance).
    """
    conn = get_database_connection()
    # Map DataFrame order to indices
    q_ids = query_genes_df['gene_id'].tolist()
    t_ids = target_genes_df['gene_id'].tolist()
    q_index = {gid: i for i, gid in enumerate(q_ids)}
    t_index = {gid: i for i, gid in enumerate(t_ids)}

    # Fetch ordered genes by relative_position in this block
    q_rows = pd.read_sql_query(
        """
        SELECT gbm.gene_id, gbm.relative_position
        FROM gene_block_mappings gbm
        WHERE gbm.block_id = ? AND gbm.block_role = 'query'
        ORDER BY gbm.relative_position
        """,
        conn, params=[int(block_id)]
    )
    t_rows = pd.read_sql_query(
        """
        SELECT gbm.gene_id, gbm.relative_position
        FROM gene_block_mappings gbm
        WHERE gbm.block_id = ? AND gbm.block_role = 'target'
        ORDER BY gbm.relative_position
        """,
        conn, params=[int(block_id)]
    )

    # Build gene arrays for rendering
    def _genes_payload(df: pd.DataFrame) -> List[Dict]:
        out = []
        for _, row in df.iterrows():
            out.append({
                'id': row['gene_id'],
                'start': int(row['start_pos']),
                'end': int(row['end_pos']),
                'strand': int(1 if row['strand'] in (1, '+') else -1),
                'pfam': row.get('pfam_domains', '') or ''
            })
        return out

    q_genes = _genes_payload(query_genes_df)
    t_genes = _genes_payload(target_genes_df)

    # Pair by rank (relative_position order), restricted to genes present in our view
    n = min(len(q_rows), len(t_rows), len(q_genes), len(t_genes))
    edges = []
    for i in range(n):
        qg = q_rows.iloc[i]['gene_id']
        tg = t_rows.iloc[i]['gene_id']
        if qg in q_index and tg in t_index:
            edges.append({
                'source': q_index[qg],
                'target': t_index[tg],
                'score': 1.0,  # placeholder weight; can map from block identity
            })

    return {
        'query_locus': q_genes,
        'target_locus': t_genes,
        'edges': edges
    }


def _render_comparative_d3(data: Dict, width: int = 1000, height: int = 500):
    """Render clinker-like comparative view using D3 in a Streamlit component."""
    import streamlit.components.v1 as components
    import json
    payload = json.dumps(data)
    html = f"""
    <div id="cmp" style="width:100%; height:{height}px;"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    const data = {payload};
    const width = {width};
    const height = {height};
    const margin = {{top: 20, right: 20, bottom: 20, left: 20}};
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const trackH = innerH/2 - 40;

    const svg = d3.select('#cmp').append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('background', '#fff');

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales: map gene index to x
    const qN = data.query_locus.length;
    const tN = data.target_locus.length;
    const qX = d3.scaleLinear().domain([0, Math.max(1,qN-1)]).range([0, innerW]);
    const tX = d3.scaleLinear().domain([0, Math.max(1,tN-1)]).range([0, innerW]);

    // Draw query genes (top)
    const qY = 40;
    const geneW = Math.max(6, innerW / Math.max(qN, tN) * 0.6);
    const geneH = 12;

    function arrowPath(x, y, w, h, strand) {{
      if (strand >= 0) {{
        return `M${x},${y} h${w-h/2} l${h/2},${h/2} l-${h/2},${h/2} h-${w-h/2} z`;
      }} else {{
        return `M${x+w},${y} h-${w-h/2} l-${h/2},${h/2} l${h/2},${h/2} h${w-h/2} z`;
      }}
    }}

    const q = g.append('g');
    q.selectAll('path.gene')
      .data(data.query_locus)
      .enter().append('path')
      .attr('class','gene')
      .attr('d', (d,i)=>arrowPath(qX(i), qY, geneW, geneH, d.strand))
      .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
      .attr('stroke','#000').attr('stroke-width',0.5)
      .append('title').text(d=>`${d.id}\nPFAM: ${d.pfam||'None'}`);

    // Draw target genes (bottom)
    const tY = innerH - 40;
    const t = g.append('g');
    t.selectAll('path.gene')
      .data(data.target_locus)
      .enter().append('path')
      .attr('class','gene')
      .attr('d', (d,i)=>arrowPath(tX(i), tY, geneW, geneH, d.strand))
      .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
      .attr('stroke','#000').attr('stroke-width',0.5)
      .append('title').text(d=>`${d.id}\nPFAM: ${d.pfam||'None'}`);

    // Edges
    const edges = g.append('g').attr('opacity',0.7);
    edges.selectAll('path.edge')
      .data(data.edges)
      .enter().append('path')
      .attr('class','edge')
      .attr('fill','none')
      .attr('stroke', d=> d.score>0.8? '#2c7bb6' : d.score>0.5? '#abd9e9':'#fdae61')
      .attr('stroke-width', d=> 1 + 2*(d.score||0.5))
      .attr('d', d=> {{
        const x1 = qX(d.source)+geneW/2, y1=qY+geneH/2;
        const x2 = tX(d.target)+geneW/2, y2=tY+geneH/2;
        const mx = (x1+x2)/2;
        return `M${x1},${y1} C${mx},${y1-60} ${mx},${y2+60} ${x2},${y2}`;
      }})
      .append('title').text(d=>`score=${d.score}`);

    </script>
    """
    components.html(html, height=height+20)


def _compute_shingles_for_block(block, window_lookup, srp_bits, srp_bands, srp_band_bits, srp_seed, shingle_k):
    # Build embedding matrix from query_windows
    win_ids = block.query_windows if getattr(block, 'query_windows', None) else []
    embs = []
    for wid in win_ids:
        emb = window_lookup(wid)
        if emb is not None:
            embs.append(emb)
    if not embs:
        return set()
    import numpy as np
    emb_mat = np.vstack(embs)
    toks = srp_tokens(emb_mat, n_bits=srp_bits, n_bands=srp_bands, band_bits=srp_band_bits, seed=srp_seed)
    return block_shingles(toks, k=shingle_k)


def _expand_clusters(blocks, assignments, window_lookup,
                     df_max, shingle_k, srp_bits, srp_bands, srp_band_bits, srp_seed,
                     tau_expand, tau_peer, m_peers, M_expand,
                     max_added_per_cluster, allow_degree_relax, waves):
    # Build id->block map
    id2block = {int(b.id): b for b in blocks}
    # Build cluster map
    from collections import defaultdict
    clusters = defaultdict(list)
    for bid, cl in assignments.items():
        if cl and cl > 0:
            clusters[int(cl)].append(int(bid))

    # Compute shingles per block and df
    shingle_map = {}
    for b in blocks:
        S = _compute_shingles_for_block(b, window_lookup, srp_bits, srp_bands, srp_band_bits, srp_seed, shingle_k)
        shingle_map[int(b.id)] = S
    # DF across all blocks
    df = {}
    for S in shingle_map.values():
        for s in S:
            df[s] = df.get(s, 0) + 1
    # Filter shingles by df_max and build IDF
    import math
    N = max(1, len(shingle_map))
    idf = {s: math.log(1.0 + (N / max(1, c))) for s, c in df.items()}

    def filtered(S):
        return {s for s in S if df.get(s, 0) <= df_max}

    def wjacc(A, B):
        if not A and not B:
            return 1.0
        inter = A & B
        union = A | B
        if not union:
            return 0.0
        inter_w = sum(idf.get(s, 0.0) for s in inter)
        union_w = sum(idf.get(s, 0.0) for s in union)
        return (inter_w / union_w) if union_w > 0 else 0.0

    # Precompute filtered shingles
    fsh = {bid: filtered(S) for bid, S in shingle_map.items()}

    # One-wave expansion
    added_total = 0
    for cl_id, members in list(clusters.items()):
        if not members:
            continue
        # Compute medoid by max sum of wJaccard
        best_bid = None
        best_sum = -1.0
        for bid in members:
            s = sum(wjacc(fsh[bid], fsh[x]) for x in members if x != bid)
            if s > best_sum:
                best_sum = s
                best_bid = bid
        medoid = best_bid if best_bid is not None else members[0]

        # Candidate pool: unassigned/sink blocks
        unassigned = [int(b.id) for b in blocks if (assignments.get(int(b.id), 0) == 0)]
        accepted = 0
        for bid in unassigned:
            # Medoid score
            mscore = wjacc(fsh[bid], fsh[medoid])
            if mscore < tau_expand:
                continue
            # Low-DF anchors
            low_df_count = sum(1 for s in (fsh[bid] & fsh[medoid]) if df.get(s, 0) <= max(1, df_max // 5))
            if low_df_count < M_expand:
                continue
            # Peer support
            peer_count = 0
            for x in members:
                if wjacc(fsh[bid], fsh[x]) >= tau_peer:
                    peer_count += 1
                    if peer_count >= m_peers:
                        break
            if peer_count < m_peers:
                continue
            # Accept
            assignments[bid] = cl_id
            clusters[cl_id].append(bid)
            accepted += 1
            added_total += 1
            if max_added_per_cluster and accepted >= max_added_per_cluster:
                break
    logger.info("[TUNER] expansion added %d blocks", added_total)
    return assignments

if __name__ == "__main__":
    main()
