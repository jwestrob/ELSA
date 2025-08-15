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
import math

# Import genome visualization functions
from visualization.genome_plots import create_genome_diagram, create_comparative_genome_view

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
    
    query += " ORDER BY sb.length DESC, sb.identity DESC"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    return pd.read_sql_query(query, conn, params=params)

@st.cache_data
def load_genes_for_locus(locus_id: str, block_id: Optional[int] = None, locus_role: Optional[str] = None) -> pd.DataFrame:
    """Load genes for a specific locus, optionally filtered to aligned regions."""
    conn = get_database_connection()
    
    # Parse locus ID format: "1313.30775:1313.30775_accn|1313.30775.con.0001"
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
        return load_aligned_genes_for_block(conn, contig_id, block_id, locus_role)
    
    # Default: load all genes for the locus
    query = """
        SELECT g.*, c.length as contig_length
        FROM genes g
        JOIN contigs c ON g.contig_id = c.contig_id
        WHERE g.contig_id = ?
        ORDER BY g.start_pos
    """
    
    return pd.read_sql_query(query, conn, params=[contig_id])

@st.cache_data  
def load_aligned_genes_for_block(_conn, contig_id: str, block_id: int, locus_role: str) -> pd.DataFrame:
    """Load only the genes that are part of the aligned region for a specific block."""
    
    # Get block alignment info
    cursor = _conn.cursor()
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
    
    # Add context genes (±3 genes around the aligned region)
    context_buffer = 3
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
    
    return pd.read_sql_query(query, _conn, params=[
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
        "Min Identity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum identity score for syntenic blocks"
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
    
    # Format display columns
    display_df = blocks_df[['block_id', 'query_locus', 'target_locus', 
                           'length', 'identity', 'score', 'block_type']].copy()
    
    # Format numbers
    display_df['length'] = display_df['length'].apply(lambda x: f"{x:,} gene windows")
    display_df['identity'] = display_df['identity'].apply(lambda x: f"{x:.3f}")
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1f}")
    
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
            st.write("**Identity:**", f"{selected_block['identity']:.3f}")
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
                        st.metric("**Avg Identity**", f"{selected_block['identity']:.3f}")
                    
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
                st.metric("Identity", f"{analysis_data.identity:.1%}")
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
                comparative_fig = create_comparative_genome_view(
                    query_genes.head(30), 
                    target_genes.head(30),
                    block['query_locus'],
                    block['target_locus'],
                    width=1200,
                    height=600
                )
                st.plotly_chart(comparative_fig, use_container_width=True)

def main():
    """Main Streamlit application."""
    # Title and navigation
    st.title("🧬 ELSA Genome Browser")
    st.markdown("*Syntenic Block Explorer with PFAM Domain Annotations*")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Block Explorer", "🧬 Genome Viewer"])
    
    with tab1:
        if st.session_state.current_page != 'genome_viewer':
            st.session_state.current_page = 'dashboard'
        display_dashboard()
    
    with tab2:
        if st.session_state.current_page != 'genome_viewer':
            st.session_state.current_page = 'block_explorer'
        filters = create_sidebar_filters()
        display_block_explorer(filters)
    
    with tab3:
        st.session_state.current_page = 'genome_viewer'
        display_genome_viewer()

if __name__ == "__main__":
    main()