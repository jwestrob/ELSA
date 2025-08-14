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

# Import AI analysis components
from components.ai_analysis_panel import AIAnalysisPanel
from analysis.data_collector import SyntenicDataCollector
from analysis.gpt5_analyzer import GPT5SyntenicAnalyzer
from analysis.cassette_analyzer import analyze_conserved_cassettes

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

def _find_matching_contig_id(_conn, target_contig_id: str) -> str:
    """Find the actual contig_id in genes table that matches the target format."""
    cursor = _conn.cursor()
    
    # Try exact match first
    cursor.execute("SELECT contig_id FROM genes WHERE contig_id = ? LIMIT 1", (target_contig_id,))
    if cursor.fetchone():
        return target_contig_id
    
    # Try alternative formats
    alternatives = []
    
    # If it has genome prefix, try without: "1313.30775_accn|..." -> "accn|..."
    if '_' in target_contig_id:
        alt = target_contig_id.split('_', 1)[-1]
        alternatives.append(alt)
    
    # If it doesn't have genome prefix, try finding one that does
    else:
        cursor.execute("SELECT DISTINCT contig_id FROM genes WHERE contig_id LIKE ? LIMIT 5", (f"%{target_contig_id}",))
        for (contig_id,) in cursor.fetchall():
            alternatives.append(contig_id)
    
    # Test each alternative
    for alt in alternatives:
        cursor.execute("SELECT contig_id FROM genes WHERE contig_id = ? LIMIT 1", (alt,))
        if cursor.fetchone():
            return alt
    
    # Return original if no match found
    return target_contig_id

@st.cache_data  
def load_aligned_genes_for_block(_conn, contig_id: str, block_id: int, locus_role: str) -> pd.DataFrame:
    """Get core aligned genes for a syntenic block using gene_block_mappings."""
    
    try:
        # Normalize contig_id to match genes table format
        actual_contig_id = _find_matching_contig_id(_conn, contig_id)
        # First try using gene_block_mappings table if it exists
        cursor = _conn.cursor()
        
        # Check if gene_block_mappings table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gene_block_mappings'")
        has_mappings_table = bool(cursor.fetchone())
        
        if has_mappings_table:
            # Use gene_block_mappings to get only genes that are actually in this block
            query = """
            SELECT g.*, gbm.block_role, gbm.relative_position
            FROM genes g
            JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
            WHERE gbm.block_id = ? 
            AND g.contig_id = ?
            AND (gbm.block_role = ? OR gbm.block_role = 'both')
            ORDER BY gbm.relative_position, g.start_pos
            """
            
            aligned_genes = pd.read_sql_query(query, _conn, params=[block_id, actual_contig_id, locus_role])
            
            if not aligned_genes.empty:
                aligned_genes['synteny_role'] = 'core_aligned'
                logger.info(f"Block {block_id} {locus_role}: Found {len(aligned_genes)} genes via gene_block_mappings")
                return aligned_genes
        
        # Fallback 1: Get genes from syntenic_blocks metadata using window information
        cursor.execute("""
        SELECT query_locus, target_locus, length, query_window_start, query_window_end, 
               target_window_start, target_window_end
        FROM syntenic_blocks WHERE block_id = ?
        """, (block_id,))
        block_info = cursor.fetchone()
        
        if not block_info:
            # Check valid block range for debugging
            cursor.execute("SELECT MIN(block_id), MAX(block_id), COUNT(*) FROM syntenic_blocks")
            min_id, max_id, count = cursor.fetchone()
            raise Exception(f"Block {block_id} not found (valid range: {min_id}-{max_id}, total: {count})")
        
        query_locus, target_locus, length, query_window_start, query_window_end, target_window_start, target_window_end = block_info
        
        # Use window indices to estimate gene ranges (assuming ~5 genes per window)
        if locus_role == 'query':
            if query_window_start is not None and query_window_end is not None:
                # Estimate gene range from window indices
                start_gene_idx = query_window_start
                end_gene_idx = query_window_end + 4  # Windows overlap, add some genes
            else:
                start_gene_idx, end_gene_idx = None, None
        else:  # target
            if target_window_start is not None and target_window_end is not None:
                start_gene_idx = target_window_start
                end_gene_idx = target_window_end + 4
            else:
                start_gene_idx, end_gene_idx = None, None
        
        # Get genes from the contig within the estimated gene index range
        if start_gene_idx is not None and end_gene_idx is not None:
            query = """
            SELECT *, ROW_NUMBER() OVER (ORDER BY start_pos) - 1 as gene_index
            FROM genes 
            WHERE contig_id = ?
            ORDER BY start_pos
            """
            all_genes = pd.read_sql_query(query, _conn, params=[actual_contig_id])
            
            if not all_genes.empty:
                # Filter to the gene index range
                aligned_genes = all_genes[
                    (all_genes['gene_index'] >= start_gene_idx) & 
                    (all_genes['gene_index'] <= end_gene_idx)
                ].copy()
                
                if not aligned_genes.empty:
                    aligned_genes.drop(columns=['gene_index'], inplace=True)
            else:
                aligned_genes = pd.DataFrame()
        else:
            # Fallback 2: Get a reasonable subset of genes from the middle of the contig
            # First count total genes in contig
            cursor.execute("SELECT COUNT(*) FROM genes WHERE contig_id = ?", (actual_contig_id,))
            total_genes = cursor.fetchone()[0]
            
            if total_genes > 100:
                # Take genes from the middle portion - this gives us a focused region
                offset = total_genes // 4  # Skip first quarter
                limit = min(80, total_genes // 2)  # Take up to 80 genes from middle
                
                query = """
                SELECT * FROM genes 
                WHERE contig_id = ?
                ORDER BY start_pos
                LIMIT ? OFFSET ?
                """
                aligned_genes = pd.read_sql_query(query, _conn, params=[actual_contig_id, limit, offset])
            else:
                # Small contig, take all genes
                query = """
                SELECT * FROM genes 
                WHERE contig_id = ?
                ORDER BY start_pos
                """
                aligned_genes = pd.read_sql_query(query, _conn, params=[actual_contig_id])
        
        if not aligned_genes.empty:
            aligned_genes['synteny_role'] = 'aligned_region'
            aligned_genes['block_role'] = locus_role
            
            # Ensure span is reasonable (< 50kb for focused view)
            span = aligned_genes['end_pos'].max() - aligned_genes['start_pos'].min()
            if span > 50000:
                # Take consecutive genes from the middle
                mid_idx = len(aligned_genes) // 2
                start_idx = max(0, mid_idx - 40)
                end_idx = min(len(aligned_genes), mid_idx + 40)
                aligned_genes = aligned_genes.iloc[start_idx:end_idx].copy()
            
            logger.info(f"Block {block_id} {locus_role}: Found {len(aligned_genes)} genes (focused region)")
            return aligned_genes
        
        raise Exception(f"No genes found for {locus_role} in block {block_id}")
        
    except Exception as e:
        logger.error(f"Failed to load aligned genes for block {block_id}, {locus_role}: {e}")
        # Return empty DataFrame instead of crashing
        return pd.DataFrame(columns=['gene_id', 'genome_id', 'contig_id', 'start_pos', 'end_pos', 'strand', 
                                   'gene_length', 'protein_id', 'protein_sequence', 'pfam_domains', 
                                   'synteny_role', 'block_role'])

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
    
    # AI Analysis Configuration
    st.sidebar.subheader("🧠 AI Analysis")
    
    # Check for environment variable first
    import os
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    if env_api_key:
        st.sidebar.success("✅ Using OPENAI_API_KEY from environment")
        openai_api_key = env_api_key
    else:
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable",
            placeholder="sk-..."
        )
    
    # Initialize AI analyzer if API key available
    if openai_api_key and not st.session_state.ai_analyzer:
        try:
            st.session_state.ai_analyzer = GPT5SyntenicAnalyzer(api_key=openai_api_key)
            if not env_api_key:
                st.sidebar.success("✅ AI Analysis enabled!")
        except Exception as e:
            st.sidebar.error(f"❌ API key failed: {e}")
            st.session_state.ai_analyzer = None
    
    ai_enabled = bool(st.session_state.ai_analyzer)
    if not ai_enabled and not env_api_key:
        st.sidebar.info("💡 Set OPENAI_API_KEY environment variable or enter your API key above to enable AI analysis")
    
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
    
    # Validate that the selected block still exists (data might have changed)
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM syntenic_blocks WHERE block_id = ?", (block['block_id'],))
    if cursor.fetchone()[0] == 0:
        # Block no longer exists, clear selection
        cursor.execute("SELECT MIN(block_id), MAX(block_id), COUNT(*) FROM syntenic_blocks")
        min_id, max_id, count = cursor.fetchone()
        
        st.error(f"⚠️ Selected block {block['block_id']} no longer exists in database. Valid range: {min_id}-{max_id} ({count} total blocks)")
        st.info("Please select a new block from the Block Explorer.")
        
        # Clear invalid selection
        if 'selected_block' in st.session_state:
            del st.session_state.selected_block
        
        if st.button("← Back to Block Explorer"):
            st.session_state.current_page = 'block_explorer'
            st.rerun()
        return
    
    st.header(f"🧬 Genome Viewer - Block {block['block_id']}")
    
    # Back button
    if st.button("← Back to Block Explorer"):
        st.session_state.current_page = 'block_explorer'
        st.rerun()
    
    # AI Analysis Section
    st.markdown("---")
    ai_enabled = bool(st.session_state.ai_analyzer)
    
    if ai_enabled:
        # Check if we have cached analysis
        block_id = block['block_id']
        cached_analysis = st.session_state.ai_panel.get_cached_analysis(block_id)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("### 🧠 AI Analysis")
        with col2:
            analyze_button = st.button(
                "🤖 Analyze with GPT-5" if not cached_analysis else "🔄 Re-analyze", 
                key=f"ai_analyze_{block_id}",
                help="Get biological insights about this syntenic block"
            )
        with col3:
            if cached_analysis:
                st.success("✅ Analyzed")
            else:
                st.info("💭 Ready")
        
        # Handle analysis trigger
        if analyze_button:
            with st.spinner("🤖 Analyzing with GPT-5..."):
                try:
                    # Collect block data
                    block_data = st.session_state.data_collector.collect_block_data(block_id)
                    
                    if block_data:
                        # Run AI analysis
                        analysis_result = st.session_state.ai_analyzer.analyze_block(block_data)
                        
                        # Store result
                        st.session_state.ai_panel.store_analysis_result(block_id, analysis_result)
                        
                        st.success("✨ Analysis complete!")
                        st.rerun()
                    else:
                        st.error("❌ Could not collect block data for analysis")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    logger.error(f"AI analysis failed for block {block_id}: {e}")
        
        # Display analysis results if available
        if cached_analysis:
            if "error" not in cached_analysis:
                analysis_data = cached_analysis.get("analysis", {})
                
                # Functional Summary (most prominent)
                if "functional_summary" in analysis_data:
                    st.markdown("#### ✨ Summary")
                    st.info(analysis_data["functional_summary"])
                
                # Key insights in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if "key_genes" in analysis_data:
                        st.markdown("**🔬 Key Genes:**")
                        st.write(analysis_data["key_genes"])
                
                with col2:
                    if "biological_significance" in analysis_data:
                        st.markdown("**🎯 Significance:**")
                        st.write(analysis_data["biological_significance"])
                
                # Conservation rationale
                if "conservation_rationale" in analysis_data:
                    with st.expander("🧬 Conservation Analysis", expanded=False):
                        st.write(analysis_data["conservation_rationale"])
                
                # Export option
                metadata = cached_analysis.get("metadata", {})
                if metadata:
                    with st.expander("📊 Analysis Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", metadata.get("model", "unknown"))
                        with col2:
                            st.metric("Genomes", metadata.get("num_genomes", "unknown"))
                        with col3:
                            st.metric("PFAM Domains", metadata.get("num_pfam_domains", "unknown"))
            else:
                st.error(f"❌ Previous analysis failed: {cached_analysis.get('error', 'Unknown error')}")
    else:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            st.info("🧠 **AI Analysis Available** - Set OPENAI_API_KEY environment variable or add your API key in the sidebar to enable GPT-5 powered biological insights.")
        else:
            st.info("🧠 **AI Analysis** - Initializing GPT-5 analyzer...")
    
    st.markdown("---")
    
    # Debug info
    st.write(f"**Debug**: Loading genes for query locus: {block['query_locus']}")
    
    # Load genes for both loci - now showing only aligned regions + context
    with st.spinner("Loading aligned gene regions..."):
        query_genes = load_genes_for_locus(block['query_locus'], block['block_id'], 'query')
        target_genes = load_genes_for_locus(block['target_locus'], block['block_id'], 'target')
    
    st.write(f"**Debug**: Query genes loaded: {len(query_genes)}, Target genes loaded: {len(target_genes)}")
    
    if not query_genes.empty:
        q_span = query_genes['end_pos'].max() - query_genes['start_pos'].min()
        st.write(f"**Debug**: Query span: {query_genes['start_pos'].min():,} - {query_genes['end_pos'].max():,} = {q_span:,} bp")
    
    if not target_genes.empty:
        t_span = target_genes['end_pos'].max() - target_genes['start_pos'].min()
        st.write(f"**Debug**: Target span: {target_genes['start_pos'].min():,} - {target_genes['end_pos'].max():,} = {t_span:,} bp")
    
    # Check if spans are reasonable (should be handled by the filtering function now)
    if not query_genes.empty:
        q_span = query_genes['end_pos'].max() - query_genes['start_pos'].min()
        st.success(f"✅ Query region span: {q_span:,} bp ({len(query_genes)} genes)")
    
    if not target_genes.empty:
        t_span = target_genes['end_pos'].max() - target_genes['start_pos'].min()
        st.success(f"✅ Target region span: {t_span:,} bp ({len(target_genes)} genes)")
    
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
            if 'relative_position' in query_genes.columns:
                display_cols.append('relative_position')
            if 'gene_index' in query_genes.columns:
                display_cols.append('gene_index')
            if 'synteny_role' in query_genes.columns:
                display_cols.append('synteny_role')
                
            display_cols.append('pfam_domains')
            
            # Create a more informative summary
            summary_df = query_genes[display_cols].copy()
            
            # Color-code the dataframe based on alignment role
            if 'relative_position' in summary_df.columns:
                def highlight_alignment_role(row):
                    rel_pos = row.get('relative_position', 0.5)
                    # Color based on position within block (0.0 = start, 1.0 = end)
                    if 0.2 <= rel_pos <= 0.8:
                        return ['background-color: #8B0000; color: white'] * len(row)  # Dark red - core region
                    elif rel_pos < 0.2 or rel_pos > 0.8:
                        return ['background-color: #CC6600; color: white'] * len(row)  # Orange - edges
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
            if 'relative_position' in target_genes.columns:
                display_cols.append('relative_position')
            if 'gene_index' in target_genes.columns:
                display_cols.append('gene_index')
            if 'synteny_role' in target_genes.columns:
                display_cols.append('synteny_role')
                
            display_cols.append('pfam_domains')
            
            # Create a more informative summary
            summary_df = target_genes[display_cols].copy()
            
            # Color-code the dataframe based on alignment role
            if 'relative_position' in summary_df.columns:
                def highlight_alignment_role(row):
                    rel_pos = row.get('relative_position', 0.5)
                    # Color based on position within block (0.0 = start, 1.0 = end)
                    if 0.2 <= rel_pos <= 0.8:
                        return ['background-color: #8B0000; color: white'] * len(row)  # Dark red - core region
                    elif rel_pos < 0.2 or rel_pos > 0.8:
                        return ['background-color: #CC6600; color: white'] * len(row)  # Orange - edges
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
                
                # Collect homology data for this block
                homology_connections = []
                try:
                    # Get block data with homology analysis
                    block_data = st.session_state.data_collector.collect_block_data(block_id)
                    if block_data and block_data.homology_analysis:
                        homology = block_data.homology_analysis
                        
                        # Extract ortholog pairs for visualization
                        for pair in homology.get('ortholog_pairs', []):
                            homology_connections.append({
                                'query_gene': pair.get('query_id'),
                                'target_gene': pair.get('target_id'),
                                'identity': pair.get('identity', 0),
                                'relationship': pair.get('relationship', 'unknown'),
                                'conservation': pair.get('functional_conservation', 'unknown'),
                                'evalue': pair.get('evalue', 1.0)
                            })
                        
                        st.info(f"Found {len(homology_connections)} homologous gene pairs to display")
                    else:
                        st.info("No homology data available for connection visualization")
                        
                except Exception as e:
                    st.warning(f"Could not load homology data: {e}")
                
                # Analyze conserved functional cassettes
                conserved_cassettes = []
                cassette_summary = ""
                try:
                    with st.spinner("Analyzing conserved functional cassettes..."):
                        cassette_analysis = analyze_conserved_cassettes(
                            query_genes, 
                            target_genes
                        )
                        conserved_cassettes = cassette_analysis.get('cassettes', [])
                        cassette_summary = cassette_analysis.get('conservation_summary', 'No analysis available')
                        
                        if conserved_cassettes:
                            st.success(f"🎯 {cassette_summary}")
                        else:
                            st.info("🎯 No conserved functional cassettes detected between these genomic regions")
                            
                except Exception as e:
                    st.warning(f"Could not analyze functional cassettes: {e}")
                
                comparative_fig = create_comparative_genome_view(
                    query_genes, 
                    target_genes,
                    block['query_locus'],
                    block['target_locus'],
                    syntenic_connections=homology_connections,
                    homology_data=block_data.homology_analysis if block_data and block_data.homology_analysis else None,
                    conserved_cassettes=conserved_cassettes,
                    width=1200,
                    height=600
                )
                st.plotly_chart(comparative_fig, use_container_width=True)
                
                # Show connection details if available
                if homology_connections:
                    st.subheader("🔗 Homologous Gene Connections")
                    connections_df = pd.DataFrame(homology_connections)
                    
                    # Add connection strength visualization
                    def connection_strength(row):
                        if row['identity'] >= 0.9:
                            return "🟢 Very Strong"
                        elif row['identity'] >= 0.7:
                            return "🟡 Strong" 
                        elif row['identity'] >= 0.5:
                            return "🟠 Moderate"
                        else:
                            return "🔴 Weak"
                    
                    connections_df['Connection Strength'] = connections_df.apply(connection_strength, axis=1)
                    connections_df['Identity %'] = (connections_df['identity'] * 100).round(1)
                    
                    # Display connection table
                    display_cols = ['query_gene', 'target_gene', 'Identity %', 'Connection Strength', 'relationship', 'conservation']
                    st.dataframe(
                        connections_df[display_cols].rename(columns={
                            'query_gene': 'Query Gene',
                            'target_gene': 'Target Gene', 
                            'relationship': 'Relationship Type',
                            'conservation': 'Functional Conservation'
                        }),
                        hide_index=True
                    )
                
                # Show conserved cassette details if available
                if conserved_cassettes:
                    st.subheader("🎯 Conserved Functional Cassettes")
                    
                    for i, cassette in enumerate(conserved_cassettes[:5]):  # Show top 5 cassettes
                        with st.expander(f"Cassette #{i+1}: {cassette.cassette_type.title()} Conservation ({cassette.domain_conservation_score:.1%})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Query Region**")
                                st.write(f"Position: {cassette.query_start:,} - {cassette.query_end:,} bp")
                                st.write(f"Genes: {', '.join(cassette.query_genes)}")
                            
                            with col2:
                                st.write("**Target Region**")
                                st.write(f"Position: {cassette.target_start:,} - {cassette.target_end:,} bp")
                                st.write(f"Genes: {', '.join(cassette.target_genes)}")
                            
                            st.write("**Shared PFAM Domains**")
                            domains_text = ", ".join(cassette.shared_domains)
                            st.code(domains_text)
                            
                            # Conservation metrics
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Domain Conservation", f"{cassette.domain_conservation_score:.1%}")
                            with metrics_col2:
                                st.metric("Synteny Score", f"{cassette.synteny_score:.1%}")  
                            with metrics_col3:
                                st.metric("Cassette Type", cassette.cassette_type.title())

def main():
    """Main Streamlit application."""
    # Title and navigation
    st.title("🧬 ELSA Genome Browser")
    st.markdown("*Syntenic Block Explorer with PFAM Domain Annotations*")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    
    # Initialize AI analysis components
    if 'ai_panel' not in st.session_state:
        st.session_state.ai_panel = AIAnalysisPanel()
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = SyntenicDataCollector()
    if 'ai_analyzer' not in st.session_state:
        # Will be initialized when user provides API key
        st.session_state.ai_analyzer = None
    
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