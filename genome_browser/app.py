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
from elsa.params import load_config as load_elsa_config
import math
import os
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
# Reduce noisy logs from cluster_analyzer to focus on UI diagnostics
try:
    logging.getLogger('cluster_analyzer').setLevel(logging.WARNING)
except Exception:
    pass

# Constants
ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "genome_browser.db"

def _db_version() -> int:
    try:
        return int(os.path.getmtime(DB_PATH))
    except Exception:
        return 0
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
def load_gene_embeddings_df(parquet_path: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load projected gene embeddings (genes.parquet) if available.

    Tries multiple candidate roots to resolve relative path issues.

    Returns (DataFrame or None, resolved_path or None)
    DataFrame columns include: gene_id, contig_id (if present), start, end, emb_*
    """
    candidates = []
    if parquet_path:
        candidates.append(Path(parquet_path))
    # Common relative locations
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # ELSA/
    candidates.extend([
        Path('elsa_index/ingest/genes.parquet'),
        repo_root / 'elsa_index/ingest/genes.parquet',
        Path.cwd() / 'elsa_index/ingest/genes.parquet',
        Path.cwd().parent / 'elsa_index/ingest/genes.parquet',
    ])
    tried = []
    for path in candidates:
        try:
            p = Path(path)
            tried.append(str(p))
            if p.exists():
                df = pd.read_parquet(p)
                # Keep standard metadata if present
                cols = df.columns.tolist()
                emb_cols = [c for c in cols if c.startswith('emb_')]
                keep = ['gene_id']
                for c in ['contig_id', 'start', 'end']:
                    if c in cols:
                        keep.append(c)
                keep += emb_cols
                return df[keep], str(p)
        except Exception as e:
            logger.error(f"Error reading embeddings parquet {path}: {e}")
            continue
    logger.warning(f"Embeddings parquet not found. Tried: {tried}")
    return None, None

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
                        block_type: Optional[str] = None,
                        contig_search: Optional[str] = None,
                        pfam_search: Optional[str] = None,
                        order_by: Optional[str] = None) -> pd.DataFrame:
    """Load blocks for the explorer, combining macro (syntenic_blocks) and micro pairs when available.

    Applies filters and sorting; pagination is handled after combining sources.
    """
    conn = get_database_connection()
    try:
        logger.info(f"load_syntenic_blocks(limit={limit}, offset={offset}, id_thr={identity_threshold}, block_type={block_type}, size_range={size_range}, contig_search={bool(contig_search)}, pfam_search={bool(pfam_search)}, order_by={order_by})")
    except Exception:
        pass

    # Load macro blocks via SQL (with joins and filters)
    base = """
        SELECT sb.*, 
               g1.organism_name as query_organism,
               g2.organism_name as target_organism,
               bc.consensus_len as consensus_len
        FROM syntenic_blocks sb
        LEFT JOIN genomes g1 ON sb.query_genome_id = g1.genome_id
        LEFT JOIN genomes g2 ON sb.target_genome_id = g2.genome_id
        LEFT JOIN block_consensus bc ON bc.block_id = sb.block_id
        WHERE 1=1 AND sb.block_type != 'micro'
    """
    params: List = []

    if genome_filter:
        placeholders = ','.join(['?' for _ in genome_filter])
        base += f" AND (query_genome_id IN ({placeholders}) OR target_genome_id IN ({placeholders}))"
        params.extend(genome_filter * 2)

    if size_range:
        base += " AND length BETWEEN ? AND ?"
        params.extend(size_range)

    if identity_threshold > 0:
        base += " AND identity >= ?"
        params.append(identity_threshold)

    if block_type == 'macro':
        base += " AND block_type = 'macro'"
    elif block_type == 'micro':
        # skip loading macro here; will load micro below
        macro_df = pd.DataFrame()
    
    if contig_search:
        toks = [t.strip() for t in contig_search.split(',') if t.strip()]
        if toks:
            ors = []
            for t in toks:
                ors.append("query_locus LIKE ?")
                params.append(f"%{t}%")
                ors.append("target_locus LIKE ?")
                params.append(f"%{t}%")
            base += " AND (" + " OR ".join(ors) + ")"

    # We leave pfam_search applied to macro only (micro filtering is done after load if needed)
    if pfam_search and block_type != 'micro':
        toks = [t.strip().lower() for t in pfam_search.split(',') if t.strip()]
        if toks:
            ors = []
            for t in toks:
                ors.append("LOWER(g.pfam_domains) LIKE ?")
                params.append(f"%{t}%")
            sub = "SELECT 1 FROM gene_block_mappings gb JOIN genes g ON gb.gene_id = g.gene_id WHERE gb.block_id = allb.block_id AND (" + " OR ".join(ors) + ")"
            base += f" AND (block_type = 'micro' OR EXISTS ({sub}))"

    if block_type != 'micro':
        try:
            macro_df = pd.read_sql_query(base, conn, params=params)
        except Exception as e:
            logger.warning(f"Block query failed: {e}; trying without consensus join")
            base2 = base.replace("bc.consensus_len as consensus_len", "").replace("LEFT JOIN block_consensus bc ON bc.block_id = sb.block_id", "")
            macro_df = pd.read_sql_query(base2, conn, params=params)
    else:
        macro_df = pd.DataFrame()

    # Load micro pairs directly (DB), with CSV fallback
    micro_df = pd.DataFrame()
    try:
        q = """
            SELECT 
                p.block_id,
                'micro' as block_type,
                p.cluster_id,
                (p.query_genome_id || ':' || p.query_contig_id || ':' || CAST(p.query_start_bp AS TEXT) || '-' || CAST(p.query_end_bp AS TEXT)) AS query_locus,
                (p.target_genome_id || ':' || p.target_contig_id || ':' || CAST(p.target_start_bp AS TEXT) || '-' || CAST(p.target_end_bp AS TEXT)) AS target_locus,
                p.query_genome_id,
                p.target_genome_id,
                p.query_contig_id,
                p.target_contig_id,
                (CASE WHEN (
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'query') >=
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'target')
                ) THEN (
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'query')
                ) ELSE (
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'target')
                ) END) AS length,
                p.identity,
                p.score,
                g1.organism_name as query_organism,
                g2.organism_name as target_organism,
                (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'query') AS n_query_windows,
                (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'target') AS n_target_windows
            FROM micro_block_pairs p
            LEFT JOIN genomes g1 ON p.query_genome_id = g1.genome_id
            LEFT JOIN genomes g2 ON p.target_genome_id = g2.genome_id
        """
        micro_df = pd.read_sql_query(q, conn)
    except Exception:
        micro_df = pd.DataFrame()
    # Fallback to sidecar CSVs if DB table missing or empty
    if micro_df is None or micro_df.empty:
        try:
            # Locate sidecar
            here = Path(__file__).resolve()
            repo_root = here.parents[1]
            candidates = [
                Path('micro_gene/micro_block_pairs.csv'),
                repo_root / 'micro_gene' / 'micro_block_pairs.csv',
                Path.cwd() / 'micro_gene' / 'micro_block_pairs.csv',
            ]
            csv_path = None
            for p in candidates:
                if Path(p).exists():
                    csv_path = Path(p)
                    break
            if csv_path:
                pairs = pd.read_csv(csv_path)
                if not pairs.empty:
                    # Build locus strings and metadata; join organism names from genomes table
                    pairs = pairs.copy()
                    pairs['block_type'] = 'micro'
                    pairs['query_locus'] = pairs['query_genome_id'].astype(str) + ':' + pairs['query_contig_id'].astype(str) + ':' + pairs['query_start_bp'].astype(str) + '-' + pairs['query_end_bp'].astype(str)
                    pairs['target_locus'] = pairs['target_genome_id'].astype(str) + ':' + pairs['target_contig_id'].astype(str) + ':' + pairs['target_start_bp'].astype(str) + '-' + pairs['target_end_bp'].astype(str)
                    pairs['length'] = (pairs['query_end_bp'] - pairs['query_start_bp']).where(
                        (pairs['query_end_bp'] - pairs['query_start_bp']) >= (pairs['target_end_bp'] - pairs['target_start_bp']),
                        (pairs['target_end_bp'] - pairs['target_start_bp'])
                    )
                    genomes_df = pd.read_sql_query("SELECT genome_id, organism_name FROM genomes", conn)
                    gmap = dict(zip(genomes_df['genome_id'], genomes_df['organism_name']))
                    pairs['query_organism'] = pairs['query_genome_id'].map(gmap)
                    pairs['target_organism'] = pairs['target_genome_id'].map(gmap)
                    # Compute length as gene windows using mappings CSV
                    # Load mappings CSV if present to get counts
                    maps_candidates = [
                        Path('micro_gene/micro_gene_pair_mappings.csv'),
                        repo_root / 'micro_gene' / 'micro_gene_pair_mappings.csv',
                        Path.cwd() / 'micro_gene' / 'micro_gene_pair_mappings.csv',
                    ]
                    maps_path = None
                    for mp in maps_candidates:
                        if Path(mp).exists():
                            maps_path = Path(mp)
                            break
                    if maps_path:
                        maps_df = pd.read_csv(maps_path)
                        counts = maps_df.groupby(['block_id','block_role']).size().unstack(fill_value=0)
                        pairs['n_query_windows'] = pairs['block_id'].map(lambda b: int(counts.get('query', pd.Series()).get(int(b), 0)))
                        pairs['n_target_windows'] = pairs['block_id'].map(lambda b: int(counts.get('target', pd.Series()).get(int(b), 0)))
                        pairs['length'] = pairs[['n_query_windows','n_target_windows']].max(axis=1)
                    else:
                        pairs['n_query_windows'] = 0
                        pairs['n_target_windows'] = 0
                        pairs['length'] = 0
                    micro_df = pairs[['block_id','block_type','cluster_id','query_locus','target_locus','query_genome_id','target_genome_id','query_contig_id','target_contig_id','length','identity','score','query_organism','target_organism','n_query_windows','n_target_windows']]
        except Exception:
            micro_df = pd.DataFrame()
    # Normalize length for micro: use gene window count
    if micro_df is not None and not micro_df.empty:
        try:
            micro_df['length'] = micro_df[['n_query_windows','n_target_windows']].max(axis=1)
        except Exception:
            pass
    # Apply filters
    if micro_df is not None and not micro_df.empty:
        if genome_filter:
            micro_df = micro_df[(micro_df['query_genome_id'].isin(genome_filter)) | (micro_df['target_genome_id'].isin(genome_filter))]
        if size_range:
            micro_df = micro_df[(micro_df['length'] >= size_range[0]) & (micro_df['length'] <= size_range[1])]
        if identity_threshold > 0:
            micro_df = micro_df[(micro_df['identity'] >= identity_threshold)]
        if contig_search:
            toks = [t.strip() for t in contig_search.split(',') if t.strip()]
            if toks:
                ms = pd.Series(False, index=micro_df.index)
                for t in toks:
                    ms = ms | micro_df['query_locus'].str.contains(t, na=False) | micro_df['target_locus'].str.contains(t, na=False)
                micro_df = micro_df[ms]
        # pfam_search for micro skipped for now (would require joining mappings→genes)
    else:
        try:
            logger.info("Micro source empty (after DB/CSV + filters)")
        except Exception:
            pass

    # Choose sources per block_type
    if block_type == 'macro':
        combined = macro_df
    elif block_type == 'micro':
        combined = micro_df
    else:
        combined = pd.concat([macro_df, micro_df], ignore_index=True, sort=False)

    try:
        logger.info(f"Explorer sources → macro={0 if macro_df is None else len(macro_df)} micro={0 if micro_df is None else len(micro_df)} combined={len(combined)}")
    except Exception:
        pass

    # Sorting (mimic order_by)
    if order_by:
        if 'identity DESC' in order_by:
            combined = combined.sort_values(['identity','block_id'], ascending=[False, True])
        elif 'consensus_len' in order_by and 'DESC' in order_by.upper():
            if 'consensus_len' in combined.columns:
                combined = combined.sort_values(['consensus_len','block_id'], ascending=[False, True])
        elif 'consensus_len' in order_by and 'ASC' in order_by.upper():
            if 'consensus_len' in combined.columns:
                combined = combined.sort_values(['consensus_len','block_id'], ascending=[True, True])
        elif 'length DESC' in order_by:
            combined = combined.sort_values(['length','block_id'], ascending=[False, True])
        elif 'length ASC' in order_by:
            combined = combined.sort_values(['length','block_id'], ascending=[True, True])
    else:
        combined = combined.sort_values(['identity','block_id'], ascending=[False, True])

    # Pagination
    combined = combined.iloc[offset: offset + limit]
    return combined

def load_genes_for_locus(locus_id: str, block_id: Optional[int] = None, locus_role: Optional[str] = None, extended_context: bool = False) -> pd.DataFrame:
    """Load genes for a specific locus, optionally filtered to aligned regions."""
    conn = get_database_connection()
    
    # Parse locus ID format: "1313.30775:1313.30775_accn|1313.30775.con.0001"
    # Need to extract contig_id to match genes table format: "accn|1313.30775.con.0001"
    if ':' in locus_id:
        genome_part, contig_part = locus_id.split(':', 1)
        # Legacy format: "1313.30775_accn|..." -> after underscore
        if '_' in contig_part:
            contig_id = contig_part.split('_', 1)[1]
        else:
            contig_id = contig_part
        # Strip micro locus suffixes:
        # - Legacy micro: "accn|...#start-end"
        # - Paired micro: "accn|...:start-end"
        if '#' in contig_id:
            contig_id = contig_id.split('#', 1)[0]
        if ':' in contig_id:
            # Only strip if looks like a numeric range
            tail = contig_id.rsplit(':', 1)[1]
            try:
                a, b = tail.split('-')
                int(a); int(b)
                contig_id = contig_id.rsplit(':', 1)[0]
            except Exception:
                pass
    else:
        # Fallback: treat entire string as contig
        contig_id = locus_id
    
    # If we have block info, load only the aligned region + context
    if block_id is not None and locus_role is not None:
        try:
            logger.info(f"load_genes_for_locus: block_id={block_id} role={locus_role} contig={contig_id}")
        except Exception:
            pass
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
    # Try macro first
    cursor.execute("""
        SELECT block_type, query_window_start, query_window_end, target_window_start, target_window_end
        FROM syntenic_blocks 
        WHERE block_id = ?
    """, (block_id,))
    result = cursor.fetchone()
    if result:
        block_type, query_start, query_end, target_start, target_end = result
    else:
        block_type = 'micro'
        query_start = query_end = target_start = target_end = None
    try:
        logger.info(f"aligned_genes: block_id={block_id} role={locus_role} block_type={block_type} macro_windows=({query_start},{query_end})/({target_start},{target_end})")
    except Exception:
        pass

    # Determine which window range to use based on locus role
    if locus_role == 'query' and query_start is not None and query_end is not None:
        window_start, window_end = query_start, query_end
    elif locus_role == 'target' and target_start is not None and target_end is not None:
        window_start, window_end = target_start, target_end
    else:
        # If macro failed, try micro pairs to get precise aligned genes from mappings
        if str(block_type) == 'micro':
            try:
                role = 'query' if str(locus_role) == 'query' else 'target'
                # Fetch mapped gene_ids for this micro pair and role
                rows = fresh_conn.execute(
                    """
                    SELECT m.gene_id, g.start_pos, g.end_pos
                    FROM micro_gene_pair_mappings m
                    JOIN genes g ON g.gene_id = m.gene_id
                    WHERE m.block_id = ? AND m.block_role = ?
                    ORDER BY g.start_pos, g.end_pos
                    """,
                    (int(block_id), role),
                ).fetchall()
                if rows:
                    starts = [int(r[1]) for r in rows]
                    ends = [int(r[2]) for r in rows]
                    s_bp, e_bp = (min(starts), max(ends))
                    # Load region with context
                    region_df = load_genes_for_region(contig_id, s_bp, e_bp, extended_context=extended_context)
                    # Override roles: mark mapped genes as core_aligned
                    mapped = {str(r[0]) for r in rows}
                    if 'gene_id' in region_df.columns:
                        region_df['synteny_role'] = region_df['gene_id'].apply(lambda gid: 'core_aligned' if str(gid) in mapped else 'context')
                        region_df['position_in_block'] = region_df['gene_id'].apply(lambda gid: 'Conserved Block' if str(gid) in mapped else 'Flanking Region')
                    return region_df
                else:
                    # Fallback to bp spans from pairs if mapping rows are missing
                    if locus_role == 'query':
                        row = fresh_conn.execute("SELECT query_start_bp, query_end_bp FROM micro_block_pairs WHERE block_id = ?", (int(block_id),)).fetchone()
                    else:
                        row = fresh_conn.execute("SELECT target_start_bp, target_end_bp FROM micro_block_pairs WHERE block_id = ?", (int(block_id),)).fetchone()
                    if row and row[0] is not None and row[1] is not None:
                        return load_genes_for_region(contig_id, int(row[0]), int(row[1]), extended_context=extended_context)
                    # Sidecar fallback: read micro_gene_pair_mappings.csv
                    try:
                        here = Path(__file__).resolve()
                        repo_root = here.parents[1]
                        candidates = [
                            Path('micro_gene/micro_gene_pair_mappings.csv'),
                            repo_root / 'micro_gene' / 'micro_gene_pair_mappings.csv',
                            Path.cwd() / 'micro_gene' / 'micro_gene_pair_mappings.csv',
                        ]
                        csv_path = None
                        for p in candidates:
                            if Path(p).exists():
                                csv_path = Path(p)
                                break
                        if csv_path:
                            import pandas as _pd
                            mdf = _pd.read_csv(csv_path)
                            mdf = mdf[(mdf['block_id'] == int(block_id)) & (mdf['block_role'] == role)]
                            if not mdf.empty:
                                s_bp = int(mdf['start_pos'].min())
                                e_bp = int(mdf['end_pos'].max())
                                try:
                                    logger.info(f"aligned_genes: CSV fallback block_id={block_id} role={role} span=({s_bp},{e_bp})")
                                except Exception:
                                    pass
                                return load_genes_for_region(contig_id, s_bp, e_bp, extended_context=extended_context)
                    except Exception:
                        pass
            except Exception:
                pass
        # Fallback: load all genes
        query = """
            SELECT g.*, c.length as contig_length, 'unknown' as synteny_role
            FROM genes g
            JOIN contigs c ON g.contig_id = c.contig_id
            WHERE g.contig_id = ?
            ORDER BY g.start_pos
        """
        return pd.read_sql_query(query, fresh_conn, params=[contig_id])
    
    # Convert indices to gene ranges
    if str(block_type) == 'micro':
        # For micro, query_window_* are already gene indices
        gene_start_idx = int(window_start) if window_start is not None else 0
        gene_end_idx = int(window_end) if window_end is not None else gene_start_idx
    else:
        # Macro: window N covers genes N..N+4
        gene_start_idx = int(window_start) if window_start is not None else 0
        gene_end_idx = (int(window_end) + 4) if window_end is not None else (gene_start_idx + 4)
    
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
    
    try:
        logger.info(f"aligned_genes: final SQL windowed fetch block_id={block_id} role={locus_role} start_idx={gene_start_idx} end_idx={gene_end_idx}")
    except Exception:
        pass
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
                    block_type: Optional[str] = None,
                    contig_search: Optional[str] = None,
                    pfam_search: Optional[str] = None) -> int:
    """Get total count of blocks (macro+micro) matching filters."""
    conn = get_database_connection()

    base = """
        WITH sb_main AS (
            SELECT block_id, cluster_id, query_locus, target_locus, query_genome_id, target_genome_id,
                   query_contig_id, target_contig_id, length, identity, block_type
            FROM syntenic_blocks
        ),
        micro_pairs AS (
            SELECT 
                p.block_id, p.cluster_id,
                (p.query_genome_id || ':' || p.query_contig_id || ':' || CAST(p.query_start_bp AS TEXT) || '-' || CAST(p.query_end_bp AS TEXT)) AS query_locus,
                (p.target_genome_id || ':' || p.target_contig_id || ':' || CAST(p.target_start_bp AS TEXT) || '-' || CAST(p.target_end_bp AS TEXT)) AS target_locus,
                p.query_genome_id, p.target_genome_id,
                p.query_contig_id, p.target_contig_id,
                (CASE WHEN (
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'query') >=
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'target')
                ) THEN (
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'query')
                ) ELSE (
                    (SELECT COUNT(*) FROM micro_gene_pair_mappings m WHERE m.block_id = p.block_id AND m.block_role = 'target')
                ) END) AS length,
                p.identity, 'micro' AS block_type
            FROM micro_block_pairs p
        ),
        allb AS (
            SELECT * FROM sb_main WHERE block_type != 'micro'
            UNION ALL
            SELECT * FROM micro_pairs
        )
        SELECT COUNT(*) FROM allb WHERE 1=1
    """
    params: List = []

    if genome_filter:
        placeholders = ','.join(['?' for _ in genome_filter])
        base += f" AND (query_genome_id IN ({placeholders}) OR target_genome_id IN ({placeholders}))"
        params.extend(genome_filter * 2)

    if size_range:
        base += " AND length BETWEEN ? AND ?"
        params.extend(size_range)

    if identity_threshold > 0:
        base += " AND (block_type = 'micro' OR identity >= ?)"
        params.append(identity_threshold)

    if block_type:
        base += " AND block_type = ?"
        params.append(block_type)

    if contig_search:
        toks = [t.strip() for t in contig_search.split(',') if t.strip()]
        if toks:
            ors = []
            for t in toks:
                ors.append("query_locus LIKE ?")
                params.append(f"%{t}%")
                ors.append("target_locus LIKE ?")
                params.append(f"%{t}%")
            base += " AND (" + " OR ".join(ors) + ")"

    if pfam_search:
        toks = [t.strip().lower() for t in pfam_search.split(',') if t.strip()]
        if toks:
            ors = []
            for t in toks:
                ors.append("LOWER(g.pfam_domains) LIKE ?")
                params.append(f"%{t}%")
            sub = "SELECT 1 FROM gene_block_mappings gb JOIN genes g ON gb.gene_id = g.gene_id WHERE gb.block_id = allb.block_id AND (" + " OR ".join(ors) + ")"
            base += f" AND (block_type = 'micro' OR EXISTS ({sub}))"

    try:
        row = conn.execute(base, params).fetchone()
        cnt = int(row[0]) if row else 0
        try:
            logger.info(f"get_block_count: count={cnt} filters={{genomes:{bool(genome_filter)}, size_range:{size_range}, id_thr:{identity_threshold}, block_type:{block_type}, contig:{bool(contig_search)}, pfam:{bool(pfam_search)}}}")
        except Exception:
            pass
        return cnt
    except Exception as e:
        logger.warning(f"Unified count query failed: {e}")
        row = conn.execute("SELECT COUNT(*) FROM syntenic_blocks").fetchone()
        return int(row[0]) if row else 0

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
        value=0.0,
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
    
    # Contig substring filter
    st.sidebar.subheader("🧬 Contig Filter")
    contig_search = st.sidebar.text_input(
        "Contig substrings (comma-separated)",
        placeholder="e.g., accn|JBBKAE010000002, JALJEL000000000",
        help="Filter blocks whose query or target locus contains any of these substrings"
    )

    # PFAM domain search (substring, comma-delimited)
    st.sidebar.subheader("🎯 PFAM Filter")
    pfam_search = st.sidebar.text_input(
        "PFAM substrings (comma-separated)",
        placeholder="e.g., Ribosomal_S10, L3, tRNA-synt",
        help="Show blocks having genes annotated with any of these substrings"
    )
    
    return {
        'genomes': selected_genomes,
        'size_range': (size_min, size_max),
        'identity_threshold': identity_threshold,
        'block_type': block_type,
        'page_size': page_size,
        'contig_search': contig_search,
        'pfam_search': pfam_search
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
    
    # Sorting options
    st.subheader("Sort")
    scol1, scol2 = st.columns(2)
    with scol1:
        sort_option = st.selectbox(
            "Order by",
            [
                "Embedding similarity (desc)",
                "Consensus length (desc)",
                "Consensus length (asc)",
                "Length (desc)",
                "Length (asc)",
            ],
            index=0,
        )
    with scol2:
        pass
    order_by = {
        "Embedding similarity (desc)": "identity DESC, block_id",
        "Consensus length (desc)": "COALESCE(consensus_len, 0) DESC, block_id",
        "Consensus length (asc)": "COALESCE(consensus_len, 0) ASC, block_id",
        "Length (desc)": "length DESC, block_id",
        "Length (asc)": "length ASC, block_id",
    }[sort_option]
    
    # Load blocks for current page
    offset = (current_page - 1) * page_size
    blocks_df = load_syntenic_blocks(
        limit=page_size,
        offset=offset,
        genome_filter=filters['genomes'],
        size_range=filters['size_range'],
        identity_threshold=filters['identity_threshold'],
        block_type=filters['block_type'],
        contig_search=filters.get('contig_search') or None,
        pfam_search=filters.get('pfam_search') or None,
        order_by=order_by,
    )
    
    if blocks_df.empty:
        st.warning("No blocks found for current page.")
        return
    
    # Display blocks table
    st.subheader(f"Blocks {offset + 1}-{offset + len(blocks_df)} of {total_blocks:,}")
    
    # Format display columns - keep length as numeric for proper sorting
    cols = ['block_id', 'query_locus', 'target_locus', 'length', 'identity', 'score', 'block_type']
    if 'consensus_len' in blocks_df.columns:
        cols.insert(4, 'consensus_len')
    display_df = blocks_df[cols].copy()
    
    # Format other columns but keep length numeric
    display_df['identity'] = display_df['identity'].apply(lambda x: f"{x:.3f}")
    if 'consensus_len' in display_df.columns:
        display_df = display_df.rename(columns={'consensus_len': 'consensus_len (100% PFAM tokens)'})
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
            st.write("**Query Locus:**", selected_block.get('query_locus', ''))
            try:
                st.write("**Length:**", f"{int(selected_block.get('length', 0)):,} gene windows")
            except Exception:
                st.write("**Length:**", "N/A")
            qn = selected_block.get('n_query_windows')
            try:
                missing_qn = (qn is None) or (isinstance(qn, float) and math.isnan(qn))
            except Exception:
                missing_qn = (qn is None)
            st.write("**Query Windows:**", f"{int(qn):,}" if not missing_qn else "N/A")
            
            # Show window range information if available
            qws = selected_block.get('query_window_start')
            qwe = selected_block.get('query_window_end')
            try:
                qws_missing = (qws is None) or (isinstance(qws, float) and math.isnan(qws))
                qwe_missing = (qwe is None) or (isinstance(qwe, float) and math.isnan(qwe))
            except Exception:
                qws_missing = (qws is None)
                qwe_missing = (qwe is None)
            if not qws_missing and not qwe_missing:
                window_range = f"Windows {int(qws)}-{int(qwe)}"
                gene_range = f"Genes {int(qws)}-{int(qwe) + 4}"
                st.write("**Query Range:**", f"{window_range} ({gene_range})")
        
        with col2:
            st.write("**Target Locus:**", selected_block.get('target_locus', ''))
            try:
                st.write("**Embedding Similarity:**", f"{float(selected_block.get('identity', 0.0)):.3f}")
            except Exception:
                st.write("**Embedding Similarity:**", "N/A")
            tn = selected_block.get('n_target_windows')
            try:
                missing_tn = (tn is None) or (isinstance(tn, float) and math.isnan(tn))
            except Exception:
                missing_tn = (tn is None)
            st.write("**Target Windows:**", f"{int(tn):,}" if not missing_tn else "N/A")
            
            # Show window range information if available
            tws = selected_block.get('target_window_start')
            twe = selected_block.get('target_window_end')
            try:
                tws_missing = (tws is None) or (isinstance(tws, float) and math.isnan(tws))
                twe_missing = (twe is None) or (isinstance(twe, float) and math.isnan(twe))
            except Exception:
                tws_missing = (tws is None)
                twe_missing = (twe is None)
            if not tws_missing and not twe_missing:
                window_range = f"Windows {int(tws)}-{int(twe)}"
                gene_range = f"Genes {int(tws)}-{int(twe) + 4}"
                st.write("**Target Range:**", f"{window_range} ({gene_range})")

        # Pairwise consensus cassette (PFAM-based) for this block
        with st.expander("🧩 Pairwise Consensus Cassette (PFAM)", expanded=True):
            try:
                import sqlite3
                try:
                    from database.cluster_content import compute_block_pfam_consensus
                except Exception:
                    from genome_browser.database.cluster_content import compute_block_pfam_consensus
                conn = sqlite3.connect(str(DB_PATH))
                payload = compute_block_pfam_consensus(conn, int(selected_block['block_id']), df_percentile_ban=0.9)
                conn.close()
                tokens = payload.get('consensus', []) if isinstance(payload, dict) else []
                pairs = payload.get('pairs', []) if isinstance(payload, dict) else []
                if tokens:
                    _render_consensus_strip(tokens, pairs, height=None, key=f"pair_consprev_{selected_block['block_id']}")
                    # Quick caption on directional consensus
                    sup = [p for p in pairs if p.get('same_frac') is not None and int(p.get('support',0)) >= 1]
                    if sup:
                        import statistics as _st
                        agree = _st.mean([float(p['same_frac']) for p in sup])
                        status = 'strong' if agree >= 0.6 else ('mixed' if agree >= 0.4 else 'weak')
                        st.caption(f"Directional consensus: {status} ({agree:.0%}) across adjacent token pairs")
                else:
                    st.info("No PFAM consensus tokens could be derived for this block.")
            except Exception as e:
                st.warning(f"Consensus unavailable for block {selected_block['block_id']}: {e}")
        
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
        # Load target genes if a target locus is present (micro pairs have target)
        if not block.get('target_locus'):
            import pandas as _pd
            target_genes = _pd.DataFrame(columns=query_genes.columns if hasattr(query_genes, 'columns') else [])
        else:
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
        if not block.get('target_locus'):
            try:
                logger.info(f"genome_viewer: missing target_locus for block_id={block.get('block_id')} block_type={block.get('block_type')}")
            except Exception:
                pass
            st.info("No target locus available for this block")
        elif not target_genes.empty:
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
    
    # Comparative genome view (render above diagrams)
    st.divider()
    if not query_genes.empty and not target_genes.empty:
        st.subheader("Comparative Genome Analysis")
        # Slider to calibrate cosine threshold
        col_thr, col_sp = st.columns([1, 3])
        with col_thr:
            cos_thr = st.slider(
                "Embedding cosine threshold",
                min_value=0.70, max_value=0.99, value=0.90, step=0.01,
                help="Draw cosine homology edges for pairs with similarity ≥ threshold"
            )
        with st.spinner("Rendering comparative view..."):
            try:
                comp_data = _build_comparative_json(block['block_id'], query_genes, target_genes, cos_threshold=cos_thr)
                comp_data['query_name'] = block.get('query_locus', 'Query locus')
                comp_data['target_name'] = block.get('target_locus', 'Target locus')
                # Debug preview: show small summary to confirm data built
                try:
                    _preview = {
                        'q': len(comp_data.get('query_locus', [])),
                        't': len(comp_data.get('target_locus', [])),
                        'e': len(comp_data.get('edges', []))
                    }
                    st.caption(f"Comparative data: q={_preview['q']} t={_preview['t']} edges={_preview['e']}")
                    dbg = comp_data.get('debug', {}) or {}
                    with st.expander("Embedding debug", expanded=False):
                        st.write({
                            'embeddings_loaded': dbg.get('embeddings_loaded'),
                            'embedding_dim': dbg.get('embedding_dim'),
                            'present_q': dbg.get('present_q'),
                            'present_t': dbg.get('present_t'),
                            'cos_threshold': dbg.get('cos_threshold'),
                            'cos_pairs_at_threshold': dbg.get('cos_pairs'),
                            'embeddings_path': dbg.get('emb_path'),
                        })
                        top_list = dbg.get('top_cos_pairs') or []
                        if top_list:
                            import pandas as _pd
                            st.dataframe(_pd.DataFrame(top_list, columns=['query_gene','target_gene','cosine']).head(10))
                        else:
                            st.caption("No top cosine pairs available (embeddings missing or empty subset)")
                except Exception:
                    pass
                _render_comparative_d3_v2(comp_data, height=540)
                st.caption("Legend: green=forward strand, red=reverse strand; thicker/stronger lines = higher homology; hover highlights linked genes and shows PFAM/length/strand.")
            except Exception as e:
                st.error(f"Comparative view failed: {e}")

def load_cluster_stats():
    """Load precomputed cluster statistics."""
    try:
        from cluster_analyzer import get_all_cluster_stats
        return get_all_cluster_stats(DB_PATH)
    except Exception as e:
        logger.error(f"Error loading cluster stats: {e}")
        return []

@st.cache_data
def generate_cluster_summaries(cluster_stats):
    """Generate GPT-4.1-mini summaries for all clusters."""
    try:
        from cluster_analyzer import ClusterAnalyzer
        analyzer = ClusterAnalyzer(DB_PATH)
        
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

    # PFAM substring filter (case-insensitive matches against any domain in the cluster)
    with st.expander("PFAM filter", expanded=False):
        pfam_query = st.text_input(
            "PFAM substrings",
            placeholder="Comma-separated, e.g., ribosomal, AAA, ABC_transporter, PF00005",
            help="Show clusters containing any of the comma-separated substrings (case-insensitive) in PFAM domain names"
        )
    if pfam_query:
        # Support multiple comma-delimited substrings (OR semantics)
        terms = [t.strip().lower() for t in pfam_query.split(',') if t.strip()]
        if terms:
            def _matches(stats):
                try:
                    # Check domain_counts first (list of (name, count))
                    if getattr(stats, 'domain_counts', None):
                        for name, cnt in stats.domain_counts:
                            lname = str(name).lower()
                            if any(term in lname for term in terms):
                                return True
                    # Fallback to dominant_functions (list of names)
                    if getattr(stats, 'dominant_functions', None):
                        for name in stats.dominant_functions:
                            lname = str(name).lower()
                            if any(term in lname for term in terms):
                                return True
                except Exception:
                    pass
                return False
            filtered_stats = [s for s in filtered_stats if _matches(s)]
    
    # Default-limit the number of clusters rendered to keep UI responsive
    if 'cluster_limit' not in st.session_state:
        st.session_state.cluster_limit = 20
    with st.expander("Display options", expanded=False):
        col_l, col_r = st.columns([3,1])
        with col_l:
            new_limit = st.number_input(
                "Max clusters to display",
                min_value=1,
                max_value=max(20, len(filtered_stats)),
                value=int(st.session_state.cluster_limit),
                step=10,
                help="Limit the number of clusters shown to avoid heavy reloads",
            )
            st.session_state.cluster_limit = int(new_limit)
        with col_r:
            if st.button("Show more", help="Increase limit by 20"):
                st.session_state.cluster_limit = min(len(filtered_stats), int(st.session_state.cluster_limit) + 20)
    total_after_filter = len(filtered_stats)
    render_limit = max(1, min(total_after_filter, int(st.session_state.cluster_limit)))
    st.info(f"Showing {render_limit} of {total_after_filter} clusters")
    filtered_stats = filtered_stats[:render_limit]
    
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

    
    # Micro clusters are merged into the main explorer feed via get_all_cluster_stats
@st.cache_data
def _consensus_preview(cluster_id: int, min_core_cov: float = 0.6, df_pct: float = 0.9, max_tok: int = 0, cluster_type: str = 'macro', cache_ver: int = 0):
    """Return consensus tokens/pairs for macro or micro clusters.

    For macro clusters, use cluster_consensus or compute on the fly.
    For micro clusters, use micro_cluster_consensus or compute via micro tables.
    """
    try:
        import sqlite3, json
        conn = sqlite3.connect(str(DB_PATH))
        if str(cluster_type).lower() == 'micro':
            # Prefer precomputed micro consensus
            try:
                # Map display id -> raw micro id
                ceil_id = _macro_id_ceiling(conn)
                raw_id = int(cluster_id) - int(ceil_id)
                row = conn.execute("SELECT consensus_json FROM micro_cluster_consensus WHERE cluster_id = ?", (raw_id,)).fetchone()
                if row and row[0]:
                    payload = json.loads(row[0])
                    if isinstance(payload, dict):
                        cons = payload.get('consensus', [])
                        pairs = payload.get('pairs', [])
                        # Only accept precomputed if it has at least one token; else compute with requested params
                        if isinstance(cons, list) and len(cons) > 0:
                            try:
                                logger.info(f"consensus_preview[micro]: precomputed raw_id={raw_id} tokens={len(cons)}")
                            except Exception:
                                pass
                            return cons, pairs
            except Exception:
                pass
            # Compute on the fly for micro
            try:
                try:
                    from database.cluster_content import compute_micro_cluster_pfam_consensus
                except Exception:
                    from genome_browser.database.cluster_content import compute_micro_cluster_pfam_consensus
                payload = compute_micro_cluster_pfam_consensus(conn, raw_id, float(min_core_cov), float(df_pct), int(max_tok))
                if isinstance(payload, dict):
                    cons = payload.get('consensus', [])
                    prs = payload.get('pairs', [])
                    try:
                        logger.info(f"consensus_preview[micro]: computed raw_id={raw_id} tokens={len(cons)}")
                    except Exception:
                        pass
                    return cons, prs
                else:
                    return payload or [], []
            finally:
                conn.close()
        else:
            # Macro path
            try:
                row = conn.execute("SELECT consensus_json FROM cluster_consensus WHERE cluster_id = ?", (int(cluster_id),)).fetchone()
                if row and row[0]:
                    payload = json.loads(row[0])
                    if isinstance(payload, dict):
                        return payload.get('consensus', []), payload.get('pairs', [])
            except Exception:
                pass
            try:
                from database.cluster_content import compute_cluster_pfam_consensus
            except Exception:
                from genome_browser.database.cluster_content import compute_cluster_pfam_consensus
            payload = compute_cluster_pfam_consensus(conn, int(cluster_id), float(min_core_cov), float(df_pct), int(max_tok))
            conn.close()
            if isinstance(payload, dict):
                return payload.get('consensus', []), payload.get('pairs', [])
            else:
                return payload or [], []
    except Exception as e:
        logger.warning(f"Consensus preview failed for cluster {cluster_id}: {e}")
        return [], []

@st.cache_data
def _cluster_summary_v1(cluster_id: int):
    try:
        import sqlite3
        try:
            from database.cluster_content import compute_cluster_explorer_summary
        except Exception:
            from genome_browser.database.cluster_content import compute_cluster_explorer_summary
        conn = sqlite3.connect(str(DB_PATH))
        summary = compute_cluster_explorer_summary(conn, int(cluster_id), 0.6, 0.9, 0)
        conn.close()
        return summary
    except Exception as e:
        logger.warning(f"Cluster summary v1 failed for cluster {cluster_id}: {e}")
        return None

def _render_consensus_strip(tokens: list, pairs: list, height: int | None = None, key: str = "consensus_preview"):
    if not tokens:
        return
    import plotly.graph_objects as go
    xs = list(range(len(tokens)))
    labels = [c['token'] for c in tokens]
    covs = [c['coverage'] for c in tokens]
    colors = [c['color'] for c in tokens]
    fwd = [c.get('fwd_frac', 0.0) for c in tokens]
    nocc = [c.get('n_occ', 0) for c in tokens]
    fig = go.Figure()
    # Arrow glyphs
    shapes = []
    y_top, y_bot = 0.8, 0.2
    head_frac = 0.35
    for i, (col, ff) in enumerate(zip(colors, fwd)):
        x = xs[i]
        x_left = x - 0.45
        x_right = x + 0.45
        if ff >= 0.5:
            body_end = x_right - head_frac
            path = f"M {x_left},{y_bot} L {body_end},{y_bot} L {x_right},0.5 L {body_end},{y_top} L {x_left},{y_top} Z"
        else:
            body_start = x_left + head_frac
            path = f"M {x_right},{y_bot} L {body_start},{y_bot} L {x_left},0.5 L {body_start},{y_top} L {x_right},{y_top} Z"
        shapes.append(dict(type='path', path=path, fillcolor=col, line=dict(color='rgba(0,0,0,0.2)', width=1)))
    hov = [f"{lab}<br>cov={cov:.0%}<br>fwd={ff:.0%} (n={nn})" for lab, cov, ff, nn in zip(labels, covs, fwd, nocc)]
    fig.add_trace(go.Scatter(x=xs, y=[0.5]*len(xs), mode='markers', marker=dict(size=1, opacity=0), hovertext=hov, hoverinfo='text', showlegend=False))
    # Connectors
    for p in pairs:
        if p.get('same_frac') is None or p.get('support', 0) < 3:
            continue
        i, j = p['i'], p['j']
        frac = p['same_frac']
        col = 'green' if frac >= 0.8 else ('goldenrod' if frac >= 0.6 else 'gray')
        x0 = i + 0.45
        x1 = j - 0.45
        fig.add_annotation(x=x1, y=0.5, ax=x0, ay=0.5, xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor=col,
                           hovertext=f"{labels[i]}→{labels[j]} same {frac:.0%} (n={p['support']})",
                           hoverlabel=dict(bgcolor=col), opacity=0.9)
    # Dynamic height scaling with token count
    if height is None:
        n = max(1, len(tokens))
        height = 70 if n <= 10 else min(140, 70 + (n - 10) * 4)
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=4, b=4), shapes=shapes)
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[-0.5, len(xs)-0.5])
    fig.update_yaxes(visible=False, range=[0, 1.2])
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=key)

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
        # Determine micro via DB instead of trusting stored type
        try:
            is_micro_card = _cluster_is_micro(int(stats.cluster_id))
        except Exception:
            is_micro_card = (str(getattr(stats, 'cluster_type', '')).lower() == 'micro')
        label = 'micro' if is_micro_card else ''
        badge = f" <span style=\"background:#eef;color:#2E5CAF;border-radius:6px;padding:2px 6px;margin-left:6px;font-size:12px;\">{label}</span>" if label else ''
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h4 style="color: #2E5CAF; margin: 0 0 10px 0;">🧩 Cluster {stats.cluster_id}{badge}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Mini consensus preview (fast & cached)
        try:
            # Use a slightly lower coverage threshold for micro to surface small cassettes
            if is_micro_card:
                tokens, pairs = _consensus_preview(int(stats.cluster_id), min_core_cov=0.5, df_pct=0.9, max_tok=0, cluster_type='micro', cache_ver=_db_version())
            else:
                tokens, pairs = _consensus_preview(int(stats.cluster_id), min_core_cov=0.6, df_pct=0.9, max_tok=0, cluster_type='macro', cache_ver=_db_version())
            _render_consensus_strip(tokens, pairs, height=None, key=f"consprev_{stats.cluster_id}")
            # Quick directional consensus badge
            if is_micro_card:
                try:
                    con, pr = _consensus_preview(int(stats.cluster_id), min_core_cov=0.5, df_pct=0.9, max_tok=0, cluster_type='micro', cache_ver=_db_version())
                    supported = [p for p in pr if p.get('same_frac') is not None and int(p.get('support', 0)) >= 3]
                    import statistics as _st
                    agree = _st.mean([float(p['same_frac']) for p in supported]) if supported else 0.0
                    status = 'strong' if agree >= 0.6 else ('mixed' if agree >= 0.4 else 'weak')
                    core = sum(1 for c in con if float(c.get('coverage', 0.0)) >= 0.6)
                    st.caption(f"Directional consensus: {status} ({agree:.0%}) • {core} core tokens")
                except Exception:
                    pass
            else:
                summary = _cluster_summary_v1(int(stats.cluster_id))
                if summary and isinstance(summary, dict):
                    dc = summary.get('directional_consensus', {})
                    agree = dc.get('agree_frac', 0.0)
                    status = dc.get('status', 'weak')
                    caveats = summary.get('caveats', [])
                    hint = summary.get('preview', {}).get('hint', '')
                    st.caption(f"Directional consensus: {status} ({agree:.0%}) • {hint}{' • ' + ', '.join(caveats) if caveats else ''}")
        except Exception as e:
            logger.debug(f"Consensus preview error for cluster {stats.cluster_id}: {e}")

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
                        analyzer = ClusterAnalyzer(DB_PATH)
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
                    # Go to dedicated cluster detail view (supports macro+micro)
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

def _macro_id_ceiling(conn) -> int:
    """Return 1 + max macro cluster_id (so micro display IDs won't collide).

    Using COUNT(*) was wrong when macro IDs are sparse. This uses MAX(cluster_id).
    """
    try:
        row = conn.execute("SELECT COALESCE(MAX(cluster_id), 0) FROM clusters WHERE cluster_id > 0").fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0

def _cluster_is_micro(cluster_id: int) -> bool:
    """Decide whether a display cluster_id refers to a micro cluster.

    Rules:
      - If cluster exists in `clusters`, use its `cluster_type` column when present.
      - Otherwise, treat display IDs > max(macro_id) as micro (display_id = ceil_id + raw_micro).
    """
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        # Prefer explicit cluster_type if present in clusters table
        try:
            row = conn.execute(
                "SELECT cluster_type FROM clusters WHERE cluster_id = ?",
                (int(cluster_id),)
            ).fetchone()
            if row is not None and len(row) > 0 and row[0] is not None:
                ctype = str(row[0]).strip().lower()
                conn.close()
                return (ctype == 'micro')
        except Exception:
            pass
        # Fallback: infer by display ID offset relative to max macro cluster id
        ceil_id = _macro_id_ceiling(conn)
        raw_micro = int(cluster_id) - int(ceil_id)
        if raw_micro <= 0:
            conn.close()
            return False
        row = conn.execute("SELECT 1 FROM micro_gene_clusters WHERE cluster_id = ?", (raw_micro,)).fetchone()
        conn.close()
        return bool(row)
    except Exception:
        return False

@st.cache_data
def load_micro_cluster_blocks(cluster_id: int) -> pd.DataFrame:
    """Load micro blocks (paired) belonging to a micro cluster.

    Prefers micro_block_pairs (query+target). Falls back to single-locus micro_gene_blocks
    (query only) if pairs are unavailable.
    """
    conn = get_database_connection()
    ceil_id = _macro_id_ceiling(conn)
    raw_id = int(cluster_id) - int(ceil_id)
    try:
        q1 = """
            SELECT 
                block_id,
                cluster_id,
                (query_genome_id || ':' || query_contig_id || ':' || CAST(query_start_bp AS TEXT) || '-' || CAST(query_end_bp AS TEXT)) AS query_locus,
                (target_genome_id || ':' || target_contig_id || ':' || CAST(target_start_bp AS TEXT) || '-' || CAST(target_end_bp AS TEXT)) AS target_locus,
                query_genome_id AS query_genome_id,
                target_genome_id AS target_genome_id,
                query_contig_id AS query_contig_id,
                target_contig_id AS target_contig_id,
                (CASE WHEN (query_end_bp - query_start_bp) > (target_end_bp - target_start_bp) THEN (query_end_bp - query_start_bp) ELSE (target_end_bp - target_start_bp) END) AS length,
                identity,
                score
            FROM micro_block_pairs
            WHERE cluster_id = ?
            ORDER BY score DESC, block_id
        """
        # Honor session source hint to avoid DB/sidecar divergence
        try:
            src = st.session_state.get(f"micro_pairs_source_{int(raw_id)}")
        except Exception:
            src = None
        if src == 'sidecar':
            raise RuntimeError('force_sidecar')
        df = pd.read_sql_query(q1, conn, params=[raw_id])
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    # Fallback: load from sidecar CSV (pairs) if available
    try:
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        candidates = [
            # Common sidecar locations
            Path('micro_gene/micro_block_pairs.csv'),
            Path('syntenic_analysis/micro_gene/micro_block_pairs.csv'),
            Path.cwd() / 'micro_gene' / 'micro_block_pairs.csv',
            Path.cwd() / 'syntenic_analysis' / 'micro_gene' / 'micro_block_pairs.csv',
            repo_root / 'micro_gene' / 'micro_block_pairs.csv',
            repo_root / 'syntenic_analysis' / 'micro_gene' / 'micro_block_pairs.csv',
        ]
        for p in candidates:
            if Path(p).exists():
                pairs = pd.read_csv(p)
                pairs = pairs[pairs['cluster_id'] == raw_id].copy()
                if not pairs.empty:
                    pairs['query_locus'] = pairs['query_genome_id'].astype(str) + ':' + pairs['query_contig_id'].astype(str) + ':' + pairs['query_start_bp'].astype(str) + '-' + pairs['query_end_bp'].astype(str)
                    pairs['target_locus'] = pairs['target_genome_id'].astype(str) + ':' + pairs['target_contig_id'].astype(str) + ':' + pairs['target_start_bp'].astype(str) + '-' + pairs['target_end_bp'].astype(str)
                    pairs['length'] = (pairs['query_end_bp'] - pairs['query_start_bp']).where(
                        (pairs['query_end_bp'] - pairs['query_start_bp']) >= (pairs['target_end_bp'] - pairs['target_start_bp']),
                        (pairs['target_end_bp'] - pairs['target_start_bp'])
                    )
                    return pairs[['block_id','cluster_id','query_locus','target_locus','query_genome_id','target_genome_id','query_contig_id','target_contig_id','length','identity','score']]
    except Exception:
        pass
    # No legacy fallback: if no paired micro blocks found (DB and sidecar), return empty
    return pd.DataFrame(columns=[
        'block_id','cluster_id','query_locus','target_locus','query_genome_id','target_genome_id','query_contig_id','target_contig_id','length','identity','score'
    ])

def compute_display_regions_for_micro_cluster(cluster_id: int, gap_bp: int = 1000, min_support: int = 1) -> List[Dict]:
    """Compute display regions for a micro cluster using paired spans in micro_block_pairs.

    Returns regions keyed by genome/contig with merged intervals and the supporting micro pair block_ids.
    """
    conn = get_database_connection()
    ceil_id = _macro_id_ceiling(conn)
    raw_id = int(cluster_id) - int(ceil_id)
    try:
        logger.info(f"multiloc: compute regions for micro cluster display_id={cluster_id}")
        pairs = pd.read_sql_query(
            """
            SELECT 
                block_id,
                query_genome_id, query_contig_id, query_start_bp, query_end_bp,
                target_genome_id, target_contig_id, target_start_bp, target_end_bp
            FROM micro_block_pairs
            WHERE cluster_id = ?
            """,
            conn,
            params=[raw_id],
        )
        if pairs is None or pairs.empty:
            # Try sidecar CSVs for pairs before falling back
            try:
                here = Path(__file__).resolve()
                repo_root = here.parents[1]
                candidates = [
                    Path('micro_gene/micro_block_pairs.csv'),
                    Path('syntenic_analysis/micro_gene/micro_block_pairs.csv'),
                    Path.cwd() / 'micro_gene' / 'micro_block_pairs.csv',
                    Path.cwd() / 'syntenic_analysis' / 'micro_gene' / 'micro_block_pairs.csv',
                    repo_root / 'micro_gene' / 'micro_block_pairs.csv',
                    repo_root / 'syntenic_analysis' / 'micro_gene' / 'micro_block_pairs.csv',
                ]
                for p in candidates:
                    if Path(p).exists():
                        pairs_csv = pd.read_csv(p)
                        pairs_csv = pairs_csv[pairs_csv['cluster_id'] == raw_id].copy()
                        if not pairs_csv.empty:
                            pairs = pairs_csv
                            logger.info("multiloc: loaded pairs from sidecar CSV")
                            try:
                                st.session_state[f"micro_pairs_source_{int(raw_id)}"] = 'sidecar'
                            except Exception:
                                pass
                            break
            except Exception:
                pass
        if pairs is None or pairs.empty:
            logger.warning("multiloc: no micro pairs found (DB or sidecar); regions unavailable for this micro cluster")
            return []
        genomes_df = pd.read_sql_query("SELECT genome_id, organism_name FROM genomes", conn)
        org_map = dict(zip(genomes_df["genome_id"], genomes_df["organism_name"]))

        regions = []
        # Determine pair source for UI/logging consistency
        try:
            src_hint = st.session_state.get(f"micro_pairs_source_{int(raw_id)}", 'db')
        except Exception:
            src_hint = 'db'
        # If pairs came from DB, stamp hint
        try:
            if src_hint not in ('db','sidecar'):
                st.session_state[f"micro_pairs_source_{int(raw_id)}"] = 'db'
        except Exception:
            pass
        # Build intervals per side
        for role in ("query", "target"):
            gcol = f"{role}_genome_id"; ccol = f"{role}_contig_id"; scol = f"{role}_start_bp"; ecol = f"{role}_end_bp"
            side = pairs[["block_id", gcol, ccol, scol, ecol]].copy()
            side = side.rename(columns={gcol: "genome_id", ccol: "contig_id", scol: "start_bp", ecol: "end_bp"})
            side = side.dropna(subset=["genome_id", "contig_id", "start_bp", "end_bp"]).astype({"start_bp": int, "end_bp": int})
            # Optional core intervals if present
            core_cols = [f"{role[0]}_core_start_bp", f"{role[0]}_core_end_bp"]
            has_core = all(col in pairs.columns for col in core_cols)
            if has_core:
                side_core = pairs[["block_id", gcol, ccol, core_cols[0], core_cols[1]]].copy()
                side_core = side_core.rename(columns={gcol: "genome_id", ccol: "contig_id", core_cols[0]: "start_bp", core_cols[1]: "end_bp"})
                side_core = side_core.dropna(subset=["genome_id", "contig_id", "start_bp", "end_bp"]).astype({"start_bp": int, "end_bp": int})
            for (genome_id, contig_id), group in side.groupby(["genome_id", "contig_id"]):
                intervals = [
                    {"start_bp": int(r.start_bp), "end_bp": int(r.end_bp), "block_id": int(r.block_id)}
                    for _, r in group.iterrows()
                ]
                merged = _merge_intervals(intervals, gap_bp=gap_bp)
                # Compute core overlay by merging core intervals for this contig
                core_overlay = None
                if has_core:
                    core_rows = side_core[(side_core["genome_id"] == genome_id) & (side_core["contig_id"] == contig_id)]
                    if core_rows is not None and not core_rows.empty:
                        core_intervals = [
                            {"start_bp": int(r.start_bp), "end_bp": int(r.end_bp), "block_id": int(r.block_id)}
                            for _, r in core_rows.iterrows()
                        ]
                        core_merged = _merge_intervals(core_intervals, gap_bp=gap_bp)
                    else:
                        core_merged = []
                for iv in merged:
                    support = len(iv["blocks"])  # number of blocks supporting
                    if support >= min_support:
                        # Derive a simple core range as union of cores overlapping this region
                        core_start = None
                        core_end = None
                        if has_core:
                            for cv in core_merged:
                                if cv["end_bp"] < iv["start_bp"] or cv["start_bp"] > iv["end_bp"]:
                                    continue
                                core_start = min(core_start, cv["start_bp"]) if core_start is not None else cv["start_bp"]
                                core_end = max(core_end, cv["end_bp"]) if core_end is not None else cv["end_bp"]
                        regions.append({
                            "genome_id": genome_id,
                            "contig_id": contig_id,
                            "organism_name": org_map.get(genome_id, "Unknown organism"),
                            "start_bp": iv["start_bp"],
                            "end_bp": iv["end_bp"],
                            "support": support,
                            "blocks": iv["blocks"],
                            "core_start_bp": core_start,
                            "core_end_bp": core_end,
                            "_source": src_hint,
                        })
        regions.sort(key=lambda r: (r["genome_id"], r["contig_id"], r["start_bp"]))
        logger.info(f"multiloc: built {len(regions)} regions for micro cluster")
        return regions
    except Exception as e:
        logger.error(f"Error computing micro display regions: {e}")
        return []

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
    
    # Determine cluster type
    is_micro = _cluster_is_micro(cluster_id)

    # Load cluster blocks and display regions
    with st.spinner("Loading cluster data..."):
        try:
            # Load blocks in this cluster
            cluster_blocks = load_micro_cluster_blocks(cluster_id) if is_micro else load_cluster_blocks(cluster_id)
            
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
            
            # Compute display regions
            regions = compute_display_regions_for_micro_cluster(cluster_id, gap_bp=1000, min_support=1) if is_micro else compute_display_regions_for_cluster(cluster_id, gap_bp=1000, min_support=1)
            use_regions = len(regions) > 0
            if is_micro and not use_regions:
                st.error("No micro pairs available for this micro cluster; regions are unavailable.")
                st.caption("Run tools/diagnose_micro.py and tools/diagnose_explore_regions.py to verify pairs and region↔block linkage.")
                return
            if not is_micro and not use_regions:
                unique_loci = extract_unique_loci_from_cluster(cluster_blocks)
            
            # Load cluster stats if available
            try:
                from cluster_analyzer import ClusterAnalyzer
                analyzer = ClusterAnalyzer(DB_PATH)
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
    # Banner for micro pair source (DB vs sidecar)
    try:
        if is_micro and 'regions' in locals() and regions:
            srcs = {r.get('_source','db') for r in regions}
            if 'sidecar' in srcs and len(srcs) == 1:
                st.caption("Using micro pairs from sidecar CSV for this cluster.")
            elif 'db' in srcs and len(srcs) == 1:
                st.caption("Using micro pairs from database for this cluster.")
    except Exception:
        pass
    
    # Consensus cassette (PFAM-based) controls and render
    with st.expander("🧩 Consensus Cassette (PFAM)", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            min_core_cov = st.slider("Min coverage", min_value=0.3, max_value=0.95, value=0.0, step=0.05, help="Keep PFAMs present in at least this fraction of blocks")
        with c2:
            df_pct = st.slider("Ban top DF percentile (global)", min_value=0.0, max_value=0.99, value=0.9, step=0.05, help="Filter ultra-common PFAMs computed across all blocks (0 to disable)")
        with c3:
            max_tok = st.number_input("Max tokens", min_value=3, max_value=20, value=10, step=1)
        # Use unified preview helper (macro vs micro aware)
        tokens, pairs = _consensus_preview(int(cluster_id), min_core_cov=min_core_cov, df_pct=df_pct, max_tok=int(max_tok), cluster_type=('micro' if is_micro else 'macro'), cache_ver=_db_version())
        if tokens:
            # Render prettier strip with arrow-shaped genes and thin arrows between
            import plotly.graph_objects as go
            xs = list(range(len(tokens)))
            labels = [c['token'] for c in tokens]
            covs = [c['coverage'] for c in tokens]
            colors = [c['color'] for c in tokens]
            fwd = [c.get('fwd_frac', 0.0) for c in tokens]
            nocc = [c.get('n_occ', 0) for c in tokens]

            fig = go.Figure()
            # Build arrow-like shapes for each token
            shapes = []
            y_top, y_bot = 0.8, 0.2
            body_frac, head_frac = 0.6, 0.3  # body and head proportion (total ~0.9)
            for i, (col, ff) in enumerate(zip(colors, fwd)):
                x = xs[i]
                x_left = x - 0.45
                x_right = x + 0.45
                if ff >= 0.5:
                    # Right-pointing arrow
                    body_end = x_right - head_frac
                    path = f"M {x_left},{y_bot} L {body_end},{y_bot} L {x_right},0.5 L {body_end},{y_top} L {x_left},{y_top} Z"
                else:
                    # Left-pointing arrow
                    body_start = x_left + head_frac
                    path = f"M {x_right},{y_bot} L {body_start},{y_bot} L {x_left},0.5 L {body_start},{y_top} L {x_right},{y_top} Z"
                shapes.append(dict(type='path', path=path, fillcolor=col, line=dict(color='rgba(0,0,0,0.2)', width=1)))

            # Invisible markers at centers for hover tooltips
            hov = [f"{lab}<br>coverage={cov:.0%}<br>forward={ff:.0%} (n={nn})" for lab, cov, ff, nn in zip(labels, covs, fwd, nocc)]
            fig.add_trace(go.Scatter(x=xs, y=[0.5]*len(xs), mode='markers', marker=dict(size=1, opacity=0), hovertext=hov, hoverinfo='text', showlegend=False))

            # Add thin arrows between genes to indicate directionality consistency
            for p in pairs:
                if p.get('same_frac') is None or p.get('support', 0) < 3:
                    continue
                i, j = p['i'], p['j']
                frac = p['same_frac']
                if frac >= 0.8:
                    col = 'green'
                elif frac >= 0.6:
                    col = 'goldenrod'
                else:
                    col = 'gray'
                x0 = i + 0.45
                x1 = j - 0.45
                fig.add_annotation(x=x1, y=0.5, ax=x0, ay=0.5, xref='x', yref='y', axref='x', ayref='y',
                                   showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor=col,
                                   hovertext=f"{labels[i]}→{labels[j]} same-strand {frac:.0%} (n={p['support']})",
                                   hoverlabel=dict(bgcolor=col), opacity=0.9)

            fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10), shapes=shapes)
            fig.update_xaxes(showticklabels=False, showgrid=False, range=[-0.5, len(xs)-0.5])
            fig.update_yaxes(visible=False, range=[0, 1.2])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Consensus core (ordered): " + " • ".join([f"{lab} ({cov:.0%})" for lab, cov in zip(labels, covs)]))
        else:
            st.info("No stable PFAM core detected for this cluster with current settings.")

    # Clinker-like multi-locus alignment (always visible)
    st.markdown("---")
    st.subheader("🧷 Cluster-wide Clinker Alignment")
    try:
        # Choose sensible defaults for quick view
        max_tracks = 6
        use_embeddings = True
        data = _build_cluster_multiloc_json(int(cluster_id), is_micro=is_micro, max_tracks=int(max_tracks), use_embeddings=use_embeddings)
        if data and data.get('loci') and len(data['loci']) >= 2:
            # Make the plot larger with more generous vertical spacing
            _render_multiloc_d3(data, height=180 + 100 * len(data.get('loci', [])))
        else:
            st.caption("Not enough comparable regions to render a cluster-wide alignment.")
    except Exception as e:
        st.warning(f"Multi-locus view unavailable: {e}")

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
            try:
                reg_blocks = set(region.get('blocks', []) or [])
                cb_ids = set()
                try:
                    cb_ids = set(int(x) for x in cluster_blocks['block_id'].astype('int64').tolist())
                except Exception:
                    cb_ids = set(cluster_blocks['block_id'].tolist())
                inter = reg_blocks & cb_ids
                src_hint = region.get('_source')
                logger.info(f"region-debug: cluster={cluster_id} contig={contig_id} {start_bp}-{end_bp} src={src_hint} reg_blocks={len(reg_blocks)} cluster_blocks={len(cb_ids)} intersect={len(inter)} sample_reg={(list(reg_blocks)[:5] if reg_blocks else [])}")
                if reg_blocks and not inter:
                    logger.critical(f"region-mismatch: cluster={cluster_id} contig={contig_id} {start_bp}-{end_bp} src={src_hint} region_blocks_do_not_intersect_cluster_blocks; example_region_blocks={list(reg_blocks)[:5]}")
            except Exception:
                pass
            rep_block = None
            if region.get("blocks"):
                # Normalize block_id dtype to numeric to avoid type-mismatch filtering
                try:
                    _tmp_cb = cluster_blocks.copy()
                    import pandas as _pd
                    _tmp_cb['block_id'] = _pd.to_numeric(_tmp_cb['block_id'], errors='coerce')
                    _region_block_ids = list({int(b) for b in region["blocks"]})
                    blocks_subset = _tmp_cb[_tmp_cb["block_id"].isin(_region_block_ids)].copy()
                except Exception:
                    blocks_subset = cluster_blocks[cluster_blocks["block_id"].isin(list(region["blocks"]))].copy()
                # Further restrict to blocks that overlap this region on the same contig
                if not blocks_subset.empty:
                    import re as _re
                    def _parse_range(loc: str):
                        if not isinstance(loc, str) or ':' not in loc:
                            return None, None, None, None
                        try:
                            parts = loc.split(':', 2)
                            g, c, rest = parts[0], parts[1], parts[2]
                            m = _re.search(r"(\\d+)[^-#]*[-](\\d+)$", rest)
                            if not m:
                                return g, c, None, None
                            return g, c, int(m.group(1)), int(m.group(2))
                        except Exception:
                            return None, None, None, None
                    def _overlaps_row(row):
                        gq, cq, qs, qe = _parse_range(str(row.get('query_locus','')))
                        gt, ct, ts, te = _parse_range(str(row.get('target_locus','')))
                        hit = False
                        if gq == genome_id and cq == contig_id and qs is not None and qe is not None:
                            hit = hit or not (qe < start_bp or qs > end_bp)
                        if gt == genome_id and ct == contig_id and ts is not None and te is not None:
                            hit = hit or not (te < start_bp or ts > end_bp)
                        return hit
                    pre_n = len(blocks_subset)
                    blocks_subset = blocks_subset[blocks_subset.apply(_overlaps_row, axis=1)].copy()
                    post_n = len(blocks_subset)
                    try:
                        logger.info(f"supporting-debug: cluster={cluster_id} region={i+1} pre={pre_n} post_overlap={post_n}")
                    except Exception:
                        pass
                    # If strict overlap filtered everything but we had id matches, fall back to id-only subset
                    if post_n == 0 and pre_n > 0:
                        blocks_subset = _tmp_cb[_tmp_cb["block_id"].isin(_region_block_ids)].copy()
                if not blocks_subset.empty:
                    rep_block = blocks_subset.loc[blocks_subset["score"].idxmax()]
                else:
                    # No fallback: if no overlapping supporting blocks, do not select a representative
                    rep_block = None

            if rep_block is not None:
                st.caption(
                    f"Supported by {support} block(s) | Representative block: {int(rep_block['block_id'])} (score: {rep_block['score']:.1f})"
                )
                # Quick action buttons to open genome viewer for blocks in this region
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    if st.button(
                        f"🔍 View Representative Block {int(rep_block['block_id'])}",
                        key=f"view_rep_block_{cluster_id}_{i}",
                        help="Open genome viewer for this region's representative block"
                    ):
                        sel = {
                            'block_id': int(rep_block['block_id']),
                            'query_locus': rep_block['query_locus'],
                            'target_locus': rep_block['target_locus'],
                            'identity': float(rep_block['identity']),
                            'score': float(rep_block['score']),
                            'length': int(rep_block.get('length', 0) or 0),
                            'block_type': ('micro' if is_micro else 'macro'),
                            # Hint to genome viewer which side corresponds to this region
                            'focus_role': ('query' if str(rep_block['query_locus']).startswith(f"{genome_id}:{contig_id}:") else ('target' if str(rep_block['target_locus']).startswith(f"{genome_id}:{contig_id}:") else None)),
                        }
                        st.session_state.selected_block = sel
                        st.session_state.current_page = 'genome_viewer'
                        st.rerun()
                with col_b:
                    with st.expander("View supporting blocks in genome viewer", expanded=False):
                        # Build a list of supporting blocks for this region
                        sup_df = blocks_subset.copy() if 'blocks_subset' in locals() and not blocks_subset.empty else None

                        if sup_df is not None and not sup_df.empty:
                            # Ensure required columns exist; fill if missing
                            for col in ['score','identity','length']:
                                if col not in sup_df.columns:
                                    sup_df[col] = 0.0
                            # Sort by score desc then identity desc
                            if 'score' in sup_df.columns:
                                sup_df = sup_df.sort_values(['score','identity'], ascending=[False, False])
                            else:
                                sup_df = sup_df.sort_values(['identity'], ascending=[False])
                            sup_df = sup_df[['block_id','query_locus','target_locus','score','identity','length']]
                            try:
                                st.caption(f"Supporting blocks matching this region: {len(sup_df)}")
                            except Exception:
                                pass
                            max_list = 8
                            for j, row in sup_df.head(max_list).iterrows():
                                bid = int(row['block_id'])
                                try:
                                    lbl_score = float(row.get('score', 0.0) or 0.0)
                                except Exception:
                                    lbl_score = 0.0
                                try:
                                    lbl_id = float(row.get('identity', 0.0) or 0.0)
                                except Exception:
                                    lbl_id = 0.0
                                try:
                                    lbl_len = int(row.get('length', 0) or 0)
                                except Exception:
                                    lbl_len = 0
                                # Determine which side matches this region
                                focus_role = 'query' if str(row['query_locus']).startswith(f"{genome_id}:{contig_id}:") else ('target' if str(row['target_locus']).startswith(f"{genome_id}:{contig_id}:") else None)
                                side_tag = f" ({focus_role})" if focus_role else ""
                                label = f"Block {bid} · score={lbl_score:.1f} · id={lbl_id:.3f} · len={lbl_len}{side_tag}"
                                if st.button(f"🔗 {label}", key=f"view_block_{cluster_id}_{i}_{bid}"):
                                    sel = {
                                        'block_id': bid,
                                        'query_locus': row['query_locus'],
                                        'target_locus': row['target_locus'],
                                        'identity': float(lbl_id),
                                        'score': float(lbl_score),
                                        'length': int(lbl_len),
                                        'block_type': ('micro' if is_micro else 'macro'),
                                        'focus_role': focus_role,
                                    }
                                    st.session_state.selected_block = sel
                                    st.session_state.current_page = 'genome_viewer'
                                    st.rerun()
                        else:
                            st.caption("No supporting blocks matched this region.")
            else:
                st.caption(f"Supported by {support} block(s)")
                # Even without a representative, show the supporting blocks expander
                with st.expander("View supporting blocks in genome viewer", expanded=False):
                    sup_df = blocks_subset.copy() if 'blocks_subset' in locals() and not blocks_subset.empty else None
                    if sup_df is not None and not sup_df.empty:
                        for col in ['score','identity','length']:
                            if col not in sup_df.columns:
                                sup_df[col] = 0.0
                        sup_df = sup_df.sort_values(['score','identity'], ascending=[False, False])
                        sup_df = sup_df[['block_id','query_locus','target_locus','score','identity','length']]
                        try:
                            st.caption(f"Supporting blocks matching this region: {len(sup_df)}")
                        except Exception:
                            pass
                        max_list = 8
                        for j, row in sup_df.head(max_list).iterrows():
                            bid = int(row['block_id'])
                            lbl_score = float(row.get('score', 0.0) or 0.0)
                            lbl_id = float(row.get('identity', 0.0) or 0.0)
                            lbl_len = int(row.get('length', 0) or 0)
                            focus_role = 'query' if str(row['query_locus']).startswith(f"{genome_id}:{contig_id}:") else ('target' if str(row['target_locus']).startswith(f"{genome_id}:{contig_id}:") else None)
                            side_tag = f" ({focus_role})" if focus_role else ""
                            label = f"Block {bid} · score={lbl_score:.1f} · id={lbl_id:.3f} · len={lbl_len}{side_tag}"
                            if st.button(f"🔗 {label}", key=f"view_block_{cluster_id}_{i}_{bid}"):
                                sel = {
                                    'block_id': bid,
                                    'query_locus': row['query_locus'],
                                    'target_locus': row['target_locus'],
                                    'identity': float(lbl_id),
                                    'score': float(lbl_score),
                                    'length': int(lbl_len),
                                    'block_type': ('micro' if is_micro else 'macro'),
                                    'focus_role': focus_role,
                                }
                                st.session_state.selected_block = sel
                                st.session_state.current_page = 'genome_viewer'
                                st.rerun()
                    else:
                        st.caption("No supporting blocks matched this region.")
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
                # Quick actions to open genome viewer for this locus's blocks
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    bid = int(best_block['block_id'])
                    if st.button(f"🔍 View Representative Block {bid}", key=f"view_locus_rep_{cluster_id}_{i}"):
                        sel = {
                            'block_id': bid,
                            'query_locus': best_block['query_locus'],
                            'target_locus': best_block['target_locus'],
                            'identity': float(best_block['identity']),
                            'score': float(best_block['score']),
                            'length': int(best_block['length']),
                        }
                        st.session_state.selected_block = sel
                        st.session_state.current_page = 'genome_viewer'
                        st.rerun()
                with col_b:
                    with st.expander("View blocks for this locus", expanded=False):
                        sb = sample_blocks.sort_values('score', ascending=False)
                        for j, row in sb.head(10).iterrows():
                            bid2 = int(row['block_id'])
                            label = f"Block {bid2} · score={row['score']:.1f} · id={row['identity']:.3f} · len={int(row['length'])}"
                            if st.button(f"🔗 {label}", key=f"view_locus_block_{cluster_id}_{i}_{bid2}"):
                                sel2 = {
                                    'block_id': bid2,
                                    'query_locus': row['query_locus'],
                                    'target_locus': row['target_locus'],
                                    'identity': float(row['identity']),
                                    'score': float(row['score']),
                                    'length': int(row['length']),
                                }
                                st.session_state.selected_block = sel2
                                st.session_state.current_page = 'genome_viewer'
                                st.rerun()
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
        windows_override = st.text_input("Windows parquet override (optional)", value="", help="Full path to *windows*.parquet if manifest/work_dir lookup fails")

    # Ensure lookup vars exist regardless of branch to avoid closure NameError in generators
    window_lookup = (lambda _wid: None)
    lookup_meta = {}

    st.subheader("Parameters")
    psrc_col = st.columns(1)[0]
    param_source = psrc_col.selectbox("Parameter source", ["Config YAML", "UI Overrides"], index=0, help="Use clustering parameters from the YAML (safer), or override via UI for experimentation")
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

    st.subheader("Adaptive (small-loci) path")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        enable_adaptive_shingles = st.checkbox("enable_adaptive_shingles", value=False, help="Adapt k/pattern by block length for short loci")
    with a2:
        enable_small_path = st.checkbox("enable_small_path", value=False, help="Require triangles for edges touching small blocks")
    with a3:
        small_len_thresh = st.number_input("small_len_thresh", value=6, min_value=1, step=1)
    with a4:
        small_edge_triangle_min = st.number_input("small_edge_triangle_min", value=1, min_value=0, step=1)

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
            if param_source == "Config YAML":
                try:
                    full_cfg = load_elsa_config(cfg_path)
                    cc = full_cfg.analyze.clustering
                except Exception as e:
                    st.error(f"Failed to load clustering params from YAML: {e}")
                    return
                cfg = SimpleNamespace(
                    jaccard_tau=float(getattr(cc, 'jaccard_tau', 0.75)),
                    df_max=int(getattr(cc, 'df_max', 30)),
                    min_low_df_anchors=int(getattr(cc, 'min_low_df_anchors', 3)),
                    idf_mean_min=float(getattr(cc, 'idf_mean_min', 1.0)),
                    mutual_k=int(getattr(cc, 'mutual_k', 3)),
                    max_df_percentile=(float(getattr(cc, 'max_df_percentile', 0.0)) if getattr(cc, 'max_df_percentile', None) else None),
                    v_mad_max_genes=float(getattr(cc, 'v_mad_max_genes', 0.5)),
                    enable_cassette_mode=bool(getattr(cc, 'enable_cassette_mode', True)),
                    cassette_max_len=int(getattr(cc, 'cassette_max_len', 4)),
                    degree_cap=int(getattr(cc, 'degree_cap', 10)),
                    k_core_min_degree=int(getattr(cc, 'k_core_min_degree', 3)),
                    triangle_support_min=int(getattr(cc, 'triangle_support_min', 1)),
                    use_weighted_jaccard=bool(getattr(cc, 'use_weighted_jaccard', True)),
                    use_community_detection=bool(getattr(cc, 'use_community_detection', True)),
                    community_method=str(getattr(cc, 'community_method', 'greedy')),
                    srp_bits=int(getattr(cc, 'srp_bits', 256)),
                    srp_bands=int(getattr(cc, 'srp_bands', 32)),
                    srp_band_bits=int(getattr(cc, 'srp_band_bits', 8)),
                    srp_seed=int(getattr(cc, 'srp_seed', 1337)),
                    shingle_k=int(getattr(cc, 'shingle_k', 3)),
                    shingle_method=str(getattr(cc, 'shingle_method', 'xor')),
                    bands_per_window=int(getattr(cc, 'bands_per_window', 4)),
                    band_stride=int(getattr(cc, 'band_stride', 7)),
                    min_anchors=int(getattr(cc, 'min_anchors', 4)),
                    min_span_genes=int(getattr(cc, 'min_span_genes', 8)),
                    size_ratio_min=float(getattr(cc, 'size_ratio_min', 0.5)),
                    size_ratio_max=float(getattr(cc, 'size_ratio_max', 2.0)),
                    keep_singletons=bool(getattr(cc, 'keep_singletons', False)),
                    sink_label=int(getattr(cc, 'sink_label', 0)),
                    enable_adaptive_shingles=bool(getattr(cc, 'enable_adaptive_shingles', False)),
                    enable_small_path=bool(getattr(cc, 'enable_small_path', False)),
                    small_len_thresh=int(getattr(cc, 'small_len_thresh', 6)),
                    small_edge_triangle_min=int(getattr(cc, 'small_edge_triangle_min', 1)),
                )
            else:
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
                    keep_singletons=False, sink_label=0,
                    enable_adaptive_shingles=bool(enable_adaptive_shingles),
                    enable_small_path=bool(enable_small_path),
                    small_len_thresh=int(small_len_thresh),
                    small_edge_triangle_min=int(small_edge_triangle_min),
                )

            with st.status("Re-clustering blocks...", expanded=True) as status:
                st.write("Loading blocks and window embeddings...")
                logger.info("[TUNER] Loading blocks CSV: %s", blocks_path)
                blocks = _load_blocks_from_csv(blocks_path)
                st.write(f"✓ Loaded {len(blocks)} blocks from CSV")
                logger.info("[TUNER] Loaded %d blocks", len(blocks))

                # SRP mutual-jaccard pipeline (same as elsa analyze)
                pre_n = len(blocks)
                blocks = _stitch_blocks(blocks, max_gap=1)
                st.write(f"✓ Stitching: {pre_n} → {len(blocks)} blocks after merging adjacents")
                logger.info("[TUNER] Stitching reduced blocks: %d -> %d", pre_n, len(blocks))

                override_path = Path(windows_override) if windows_override.strip() else None
                window_lookup, lookup_meta = _create_window_lookup_from_config(cfg_path, override_path)
                if window_lookup is None:
                    st.error("Failed to create window embedding lookup. Set 'Windows parquet override' to your windows.parquet.")
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

                st.write(f"Clustering {len(blocks)} blocks with updated parameters (mutual_jaccard)…")
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

    # Local lightweight match class for reconstructed matches
    class _Match:
        __slots__ = ("query_window_id", "target_window_id")
        def __init__(self, q, t):
            self.query_window_id = q
            self.target_window_id = t

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


def _build_comparative_json(block_id: int, query_genes_df: pd.DataFrame, target_genes_df: pd.DataFrame, cos_threshold: float = 0.95) -> Dict:
    """Build clinker-like JSON for comparative view.

    Preference order for edges:
    - Compute PFAM and cosine edges independently.
    - Tag each edge with type: 'cos', 'pfam', or 'both' (if present in both).
    - If neither set available, fall back to window/rank pairing (type 'fallback').
    """
    conn = get_database_connection()
    # Map DataFrame order to indices
    q_ids = query_genes_df['gene_id'].tolist()
    t_ids = target_genes_df['gene_id'].tolist()
    q_index = {gid: i for i, gid in enumerate(q_ids)}
    t_index = {gid: i for i, gid in enumerate(t_ids)}
    # Optional: map absolute gene_index -> local position
    q_abs_to_local = {}
    t_abs_to_local = {}
    if 'gene_index' in query_genes_df.columns:
        q_abs_to_local = {int(gi): pos for pos, gi in enumerate(query_genes_df['gene_index'].tolist())}
    if 'gene_index' in target_genes_df.columns:
        t_abs_to_local = {int(gi): pos for pos, gi in enumerate(target_genes_df['gene_index'].tolist())}

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

    edges = []
    cos_edges: Dict[Tuple[int,int], float] = {}
    pf_edges: Dict[Tuple[int,int], int] = {}
    debug_info: Dict[str, any] = {
        'cos_threshold': float(cos_threshold),
        'embeddings_loaded': False,
        'embedding_dim': None,
        'present_q': 0,
        'present_t': 0,
        'cos_pairs': 0,
        'top_cos_pairs': []
    }

    # Attempt 0: Embedding-based cosine similarity matching
    # Reads projected embeddings from elsa_index/ingest/genes.parquet
    try:
        import numpy as _np
        emb_df_tuple = load_gene_embeddings_df()
        # Back-compat: function now returns (df, path)
        if isinstance(emb_df_tuple, tuple):
            emb_df, emb_path = emb_df_tuple
        else:
            emb_df, emb_path = emb_df_tuple, None
        if emb_df is not None:
            # Ensure we have metadata columns for fallback mapping
            meta_cols = []
            for c in ['contig_id', 'start', 'end']:
                if c in emb_df.columns:
                    meta_cols.append(c)
            emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
            if emb_cols:
                # Subset for displayed genes
                emb_small = emb_df[['gene_id'] + meta_cols + emb_cols]
                emb_small = emb_small.copy()
                # Direct ID subset
                emb_direct = emb_small[emb_small['gene_id'].isin(q_ids + t_ids)]
                # Fallback mapping by contig_id and nearest start if direct is empty or partial
                missing_q = [gid for gid in q_ids if gid not in set(emb_direct['gene_id'])]
                missing_t = [gid for gid in t_ids if gid not in set(emb_direct['gene_id'])]
                mapped_ids = {}
                if (missing_q or missing_t) and all(x in emb_small.columns for x in ['contig_id','start']):
                    # Build helper map: for faster per-contig filtering
                    # Note: emb_small may be large; but per-locus we filter minimal
                    emb_by_contig = None  # defer, simple filter per gene is fine here
                    # Map query
                    for _, row in query_genes_df.iterrows():
                        gid = row.get('gene_id')
                        if gid in mapped_ids or gid in set(emb_direct['gene_id']):
                            continue
                        cont = row.get('contig_id')
                        sp = row.get('start_pos')
                        if pd.isna(cont) or pd.isna(sp):
                            continue
                        cand = emb_small[(emb_small['contig_id'] == cont)]
                        if cand.empty:
                            continue
                        # Choose nearest start
                        idx = (cand['start'] - int(sp)).abs().astype('int64').idxmin()
                        mapped_ids[gid] = cand.loc[idx, 'gene_id']
                    # Map target
                    for _, row in target_genes_df.iterrows():
                        gid = row.get('gene_id')
                        if gid in mapped_ids or gid in set(emb_direct['gene_id']):
                            continue
                        cont = row.get('contig_id')
                        sp = row.get('start_pos')
                        if pd.isna(cont) or pd.isna(sp):
                            continue
                        cand = emb_small[(emb_small['contig_id'] == cont)]
                        if cand.empty:
                            continue
                        idx = (cand['start'] - int(sp)).abs().astype('int64').idxmin()
                        mapped_ids[gid] = cand.loc[idx, 'gene_id']
                # Build present lists using direct or mapped ids
                emb_indexed = emb_small.set_index('gene_id')
                def resolve_ids(ids):
                    out = []
                    for gid in ids:
                        egid = gid if gid in emb_indexed.index else mapped_ids.get(gid)
                        if egid is not None and egid in emb_indexed.index:
                            out.append(egid)
                    return out
                present_q_ids = resolve_ids(q_ids)
                present_t_ids = resolve_ids(t_ids)
                debug_info.update({
                    'embeddings_loaded': True,
                    'embedding_dim': len(emb_cols),
                    'present_q': len(present_q_ids),
                    'present_t': len(present_t_ids),
                    'mapped_ids': len(mapped_ids),
                    'emb_path': emb_path
                })
                if present_q_ids and present_t_ids:
                    A = emb_indexed.loc[present_q_ids, emb_cols].to_numpy(dtype='float32', copy=False)
                    B = emb_indexed.loc[present_t_ids, emb_cols].to_numpy(dtype='float32', copy=False)
                    # L2-normalize rows
                    def _l2n(X):
                        denom = _np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
                        return X / denom
                    A = _l2n(A)
                    B = _l2n(B)
                    # Cosine similarity matrix
                    S = A @ B.T
                    tau = float(cos_threshold)
                    # Map back to local indices
                    present_q_map = {gid: i for i, gid in enumerate(present_q_ids)}
                    present_t_map = {gid: i for i, gid in enumerate(present_t_ids)}
                    # Build reverse map from embedding gid to local locus index
                    emb_to_local_q = {}
                    for gid in q_ids:
                        egid = gid if gid in present_q_map else mapped_ids.get(gid)
                        if egid in present_q_map:
                            emb_to_local_q[egid] = q_index[gid]
                    emb_to_local_t = {}
                    for gid in t_ids:
                        egid = gid if gid in present_t_map else mapped_ids.get(gid)
                        if egid in present_t_map:
                            emb_to_local_t[egid] = t_index[gid]
                    pairs_count = 0
                    for egq, i in present_q_map.items():
                        sims = S[i]
                        for egt, j in present_t_map.items():
                            sim = float(sims[j])
                            if sim >= tau:
                                # Map to local indices
                                s_loc = emb_to_local_q.get(egq)
                                t_loc = emb_to_local_t.get(egt)
                                if s_loc is not None and t_loc is not None:
                                    cos_edges[(int(s_loc), int(t_loc))] = max(sim, cos_edges.get((int(s_loc), int(t_loc)), 0.0))
                                    pairs_count += 1
                    debug_info['cos_pairs'] = pairs_count
                    # Top cosine pairs regardless of threshold
                    try:
                        flat = []
                        for egq, i in present_q_map.items():
                            for egt, j in present_t_map.items():
                                flat.append((egq, egt, float(S[i, j])))
                        flat.sort(key=lambda x: x[2], reverse=True)
                        debug_info['top_cos_pairs'] = flat[:10]
                    except Exception:
                        pass
                    
    except Exception:
        # If any error occurs, silently fall back to other methods
        pass

    # Attempt 1: PFAM-based matching within displayed loci (greedy)
    try:
        def domset(val: str):
            if not val:
                return set()
            return {d.strip() for d in str(val).split(';') if d.strip()}

        q_pf = [domset(row.get('pfam_domains', '')) for _, row in query_genes_df.iterrows()]
        t_pf = [domset(row.get('pfam_domains', '')) for _, row in target_genes_df.iterrows()]

        # Build all candidate pairs with overlap > 0
        candidates = []
        for qi, qset in enumerate(q_pf):
            if not qset:
                continue
            for ti, tset in enumerate(t_pf):
                if not tset:
                    continue
                inter = qset & tset
                if inter:
                    # Score by overlap size; tie-breaker by proximity in index
                    score = len(inter)
                    candidates.append((score, -abs(qi - ti), qi, ti))
        # Greedy selection by score then proximity
        if candidates:
            candidates.sort(reverse=True)
            used_q = set()
            used_t = set()
            for score, _, qi, ti in candidates:
                if qi in used_q or ti in used_t:
                    continue
                pf_edges[(int(qi), int(ti))] = int(score)
                used_q.add(qi)
                used_t.add(ti)
    except Exception:
        pass

    # Attempt 2: derive edges from window-paired indices if both sets are empty
    try:
        win_row = pd.read_sql_query(
            """
            SELECT query_windows_json, target_windows_json
            FROM syntenic_blocks WHERE block_id = ?
            """,
            conn, params=[int(block_id)]
        )
        if (not cos_edges and not pf_edges) and (not win_row.empty) and ('gene_index' in query_genes_df.columns) and ('gene_index' in target_genes_df.columns):
            import json as _json
            qw = _json.loads(win_row.iloc[0]['query_windows_json'] or '[]') if 'query_windows_json' in win_row.columns else []
            tw = _json.loads(win_row.iloc[0]['target_windows_json'] or '[]') if 'target_windows_json' in win_row.columns else []
            pair_count = min(len(qw), len(tw))
            seen = set()
            for i in range(pair_count):
                # Extract numeric window indices
                try:
                    qwi = int(str(qw[i]).split('_')[-1])
                    twi = int(str(tw[i]).split('_')[-1])
                except Exception:
                    continue
                # Each window covers 5 genes: indices [wi .. wi+4]
                for k in range(5):
                    q_abs = qwi + k
                    t_abs = twi + k
                    if q_abs in q_abs_to_local and t_abs in t_abs_to_local:
                        s = q_abs_to_local[q_abs]
                        t = t_abs_to_local[t_abs]
                        key = (s, t)
                        if key not in seen:
                            pf_edges[key] = max(1, pf_edges.get(key, 0))
                            seen.add(key)
    except Exception:
        pass

    # Fallback: Pair by rank (relative_position order) if no edges found
    if not cos_edges and not pf_edges:
        n = min(len(q_rows), len(t_rows), len(q_genes), len(t_genes))
        for i in range(n):
            qg = q_rows.iloc[i]['gene_id']
            tg = t_rows.iloc[i]['gene_id']
            if qg in q_index and tg in t_index:
                pf_edges[(q_index[qg], t_index[tg])] = pf_edges.get((q_index[qg], t_index[tg]), 0)

    # Merge edge sets and annotate type/metrics
    all_keys = set(cos_edges.keys()) | set(pf_edges.keys())
    for (s, t) in sorted(all_keys):
        c = cos_edges.get((s, t))
        p = pf_edges.get((s, t))
        if c is not None and p is not None:
            edges.append({'source': s, 'target': t, 'type': 'both', 'cos': float(c), 'pf': int(p)})
        elif c is not None:
            edges.append({'source': s, 'target': t, 'type': 'cos', 'cos': float(c)})
        elif p is not None:
            # If pf comes from fallback/rank with zero, tag as 'fallback'
            et = 'pfam' if p > 0 else 'fallback'
            edges.append({'source': s, 'target': t, 'type': et, 'pf': int(p)})

    return {
        'query_locus': q_genes,
        'target_locus': t_genes,
        'edges': edges,
        'debug': debug_info
    }


def _render_comparative_d3(data: Dict, width: int = 0, height: int = 500):
    """Render clinker-like comparative view using D3 in a Streamlit component.

    Notes:
    - Width is responsive to the container (100%). The `width` arg is ignored.
    - X positions are normalized per locus to fill available width while preserving
      intra-locus genomic spacing (bp-proportional within each track).
    """
    import streamlit.components.v1 as components
    import json
    import base64
    payload = json.dumps(data)
    payload_b64 = base64.b64encode(payload.encode('utf-8')).decode('ascii')
    html = """
    <div id=\"cmp\" style=\"width:100%; height:__HEIGHT__px;\" data-payload=\"__PAYLOAD_B64__\"></div>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
    (function(){
      const container = document.getElementById('cmp');
      let data;
      try {
        const payloadB64 = container.dataset.payload || '';
        data = JSON.parse(atob(payloadB64));
      } catch (e) {
        container.innerHTML = '<div style="color:red; font-family:monospace">Failed to parse data: ' + (e && e.message ? e.message : e) + '</div>';
        return;
      }
      const fixedHeight = __HEIGHT__;
      const margin = {top: 20, right: 260, bottom: 100, left: 20};


      function arrowPath(x, y, w, h, strand) {
        if (strand >= 0) {
          return 'M' + x + ',' + y + ' h' + (w - h/2) + ' l' + (h/2) + ',' + (h/2) + ' l-' + (h/2) + ',' + (h/2) + ' h-' + (w - h/2) + ' z';
        } else {
          return 'M' + (x + w) + ',' + y + ' h-' + (w - h/2) + ' l-' + (h/2) + ',' + (h/2) + ' l' + (h/2) + ',' + (h/2) + ' h' + (w - h/2) + ' z';
        }
      }

      function render() {
        const width = container.clientWidth || 800;
        const height = fixedHeight;
        container.innerHTML = '';

        const svg = d3.select(container).append('svg')
          .attr('width', width)
          .attr('height', height)
          .style('background', '#fff');

        const innerW = Math.max(10, width - margin.left - margin.right);
        const innerH = Math.max(10, height - margin.top - margin.bottom);

        const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        // Y anchors (leave space for labels and legend)
        const qY = 90;
        const tY = innerH - 70;
        const geneH = 12;
        const minGeneW = 6; // minimum pixel width for visibility

        // Scales: normalize bp per locus to full width
        const qMin = d3.min(data.query_locus, d => d.start) || 0;
        const qMax = d3.max(data.query_locus, d => d.end) || 1;
        const tMin = d3.min(data.target_locus, d => d.start) || 0;
        const tMax = d3.max(data.target_locus, d => d.end) || 1;
        const qX = d3.scaleLinear().domain([qMin, qMax]).range([0, innerW]);
        const tX = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);

        // Draw query genes (top)
        const q = g.append('g');
        try {
          q.selectAll('path.gene')
            .data(data.query_locus || [])
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = qX(d.start);
              const x1 = qX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1); // handle any reversed coords
              return arrowPath(x, qY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5)
            .append('title').text(d=> (d.id + '\nPFAM: ' + (d.pfam || 'None')));
        } catch (e) {
          q.append('text').attr('x',10).attr('y',qY).attr('fill','red').text('Render error (query): ' + e);
        }

        // Draw target genes (bottom)
        const t = g.append('g');
        try {
          t.selectAll('path.gene')
            .data(data.target_locus || [])
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = tX(d.start);
              const x1 = tX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1);
              return arrowPath(x, tY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5)
            .append('title').text(d=> (d.id + '\nPFAM: ' + (d.pfam || 'None')));
        } catch (e) {
          t.append('text').attr('x',10).attr('y',tY).attr('fill','red').text('Render error (target): ' + e);
        }

        // Edges (curves between centers)
        const edges = g.append('g').attr('opacity',0.7);
        try {
          edges.selectAll('path.edge')
            .data(data.edges || [])
            .enter().append('path')
            .attr('class','edge')
            .attr('fill','none')
            .attr('stroke', d=> d.score>0.8? '#2c7bb6' : d.score>0.5? '#abd9e9':'#fdae61')
            .attr('stroke-width', d=> 1 + 2*(d.score||0.5))
            .attr('d', d=> {
              const qs = data.query_locus[d.source] || {start: qMin, end: qMin};
              const ts = data.target_locus[d.target] || {start: tMin, end: tMin};
              const x1 = (qX(qs.start) + qX(qs.end)) / 2, y1 = qY + geneH/2;
              const x2 = (tX(ts.start) + tX(ts.end)) / 2, y2 = tY + geneH/2;
              const mx = (x1 + x2) / 2;
              return 'M' + x1 + ',' + y1 + ' C' + mx + ',' + (y1-60) + ' ' + mx + ',' + (y2+60) + ' ' + x2 + ',' + y2;
            })
          .append('title').text(d=> ('score=' + ((d.score != null) ? d.score : '')));
        } catch (e) {
          g.append('text').attr('x',10).attr('y',(qY+tY)/2).attr('fill','red').text('Render error (edges): ' + e);
        }
      }

      // Initial render and resize handling
      render();
      if ('ResizeObserver' in window) {
        const ro = new ResizeObserver(() => render());
        ro.observe(container);
      } else {
        window.addEventListener('resize', render);
      }
    })();
    </script>
    """
    html = html.replace('__PAYLOAD_B64__', payload_b64).replace('__HEIGHT__', str(height))
    components.html(html, height=height+20)


def _render_comparative_d3_v2(data: Dict, width: int = 0, height: int = 500):
    """Alternate renderer using inline JSON script tag (safer for parsing)."""
    import streamlit.components.v1 as components
    import json
    payload = json.dumps(data)
    html = """
    <div id=\"cmp\" style=\"width:100%; height:__HEIGHT__px;\"></div>
    <script id=\"cmp-data\" type=\"application/json\">__PAYLOAD_JSON__</script>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
    (function(){
      try {
        const container = document.getElementById('cmp');
        const dataText = document.getElementById('cmp-data').textContent || '{}';
        const data = JSON.parse(dataText);
        const fixedHeight = __HEIGHT__;
        // Add extra bottom margin to make room for tooltips below bottom track
        const margin = {top: 20, right: 20, bottom: 100, left: 20};

        function arrowPath(x, y, w, h, strand) {
          if (strand >= 0) {
            return 'M' + x + ',' + y + ' h' + (w - h/2) + ' l' + (h/2) + ',' + (h/2) + ' l-' + (h/2) + ',' + (h/2) + ' h-' + (w - h/2) + ' z';
          } else {
            return 'M' + (x + w) + ',' + y + ' h-' + (w - h/2) + ' l-' + (h/2) + ',' + (h/2) + ' l' + (h/2) + ',' + (h/2) + ' h' + (w - h/2) + ' z';
          }
        }

        function render() {
          const width = container.clientWidth || 800;
          const height = fixedHeight;
          container.innerHTML = '';

          const svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('background', '#fff');

          const innerW = Math.max(10, width - margin.left - margin.right);
          const innerH = Math.max(10, height - margin.top - margin.bottom);

          const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

          const qY = 40;
          const tY = innerH - 40;
          const geneH = 12;
          const minGeneW = 6;

          const qMin = d3.min(data.query_locus || [], d => d.start) || 0;
          const qMax = d3.max(data.query_locus || [], d => d.end) || 1;
          const tMin = d3.min(data.target_locus || [], d => d.start) || 0;
          const tMax = d3.max(data.target_locus || [], d => d.end) || 1;
          const qX = d3.scaleLinear().domain([qMin, qMax]).range([0, innerW]);
          const tX = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);

          // Tooltip
          container.style.position = 'relative';
          const tooltip = d3.select(container).append('div')
            .attr('class','cmp-tooltip')
            .style('position','absolute')
            .style('pointer-events','none')
            .style('background','#fff')
            .style('border','1px solid #ccc')
            .style('padding','6px 8px')
            .style('font','12px monospace')
            .style('border-radius','4px')
            .style('box-shadow','0 2px 6px rgba(0,0,0,0.15)')
            .style('display','none')
            .style('z-index','10');

          const qData = (data.query_locus || []).map((d,i)=>Object.assign({_idx:i}, d));
          const tData = (data.target_locus || []).map((d,i)=>Object.assign({_idx:i}, d));

          const q = g.append('g');
          const qGenes = q.selectAll('path.gene')
            .data(qData)
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = qX(d.start);
              const x1 = qX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1);
              return arrowPath(x, qY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5);

          qGenes
            .on('mouseover', function(event, d){
              const strand = (d.strand>=0? '+':'-');
              const len = Math.max(0, Math.round(Math.abs(d.end - d.start)));
              // Cosine similarity (max across connected edges)
              let simLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.source === d._idx);
                if (matches.length) {
                  const maxSim = d3.max(matches, ed => (ed.cos != null ? +ed.cos : -Infinity));
                  if (maxSim != null && isFinite(maxSim)) simLine = '<br>Cosine: ' + (+maxSim).toFixed(3);
                }
              } catch (e) {}
              // PFAM overlap (max across connected edges)
              let pfLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.source === d._idx && ed.pf != null);
                if (matches.length) {
                  const maxPf = d3.max(matches, ed => (+ed.pf || 0));
                  if (maxPf != null && isFinite(maxPf) && maxPf > 0) pfLine = '<br>PFAM overlap: ' + maxPf;
                }
              } catch (e) {}
              tooltip.style('display','block')
                     .html('<b>' + d.id + '</b><br>' +
                           'Length: ' + len.toLocaleString() + ' aa<br>' +
                           'Strand: ' + strand + '<br>' +
                           'PFAM: ' + (d.pfam || 'None') + simLine + pfLine);
              d3.select(this).attr('stroke-width',2);
              highlightFor('q', d._idx);
            })
            .on('mousemove', function(event){
              const pt = d3.pointer(event, container);
              // Keep tooltip within container bounds and away from labels
              const cw = container.clientWidth || 0;
              const ch = container.clientHeight || 0;
              const tw = tooltip.node().offsetWidth || 0;
              const th = tooltip.node().offsetHeight || 0;
              let left = pt[0] + 12;
              let top = pt[1] + 12;
              // Avoid overlapping top label band (~30px) and bottom label band (~30px)
              if (top < 30) top = 34;
              if (top > ch - 34 - th) top = ch - 34 - th;
              if (left + tw > cw - 4) left = Math.max(4, pt[0] - 12 - tw);
              if (top + th > ch - 4) top = Math.max(4, pt[1] - 12 - th);
              tooltip.style('left', left + 'px').style('top', top + 'px');
            })
            .on('mouseout', function(){
              tooltip.style('display','none');
              d3.select(this).attr('stroke-width',0.5);
              clearHighlight();
            });

          const t = g.append('g');
          const tGenes = t.selectAll('path.gene')
            .data(tData)
            .enter().append('path')
            .attr('class','gene')
            .attr('d', d => {
              const x0 = tX(d.start);
              const x1 = tX(d.end);
              const w = Math.max(minGeneW, Math.abs(x1 - x0));
              const x = Math.min(x0, x1);
              return arrowPath(x, tY, w, geneH, d.strand);
            })
            .attr('fill', d=> d.strand>=0 ? '#2E8B57' : '#FF6347')
            .attr('stroke','#000').attr('stroke-width',0.5);

          tGenes
            .on('mouseover', function(event, d){
              const strand = (d.strand>=0? '+':'-');
              const len = Math.max(0, Math.round(Math.abs(d.end - d.start)));
              // Cosine similarity (max across connected edges)
              let simLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.target === d._idx);
                if (matches.length) {
                  const maxSim = d3.max(matches, ed => (ed.cos != null ? +ed.cos : -Infinity));
                  if (maxSim != null && isFinite(maxSim)) simLine = '<br>Cosine: ' + (+maxSim).toFixed(3);
                }
              } catch (e) {}
              // PFAM overlap (max across connected edges)
              let pfLine = '';
              try {
                const matches = (data.edges || []).filter(ed => ed.target === d._idx && ed.pf != null);
                if (matches.length) {
                  const maxPf = d3.max(matches, ed => (+ed.pf || 0));
                  if (maxPf != null && isFinite(maxPf) && maxPf > 0) pfLine = '<br>PFAM overlap: ' + maxPf;
                }
              } catch (e) {}
              tooltip.style('display','block')
                     .html('<b>' + d.id + '</b><br>' +
                           'Length: ' + len.toLocaleString() + ' aa<br>' +
                           'Strand: ' + strand + '<br>' +
                           'PFAM: ' + (d.pfam || 'None') + simLine + pfLine);
              d3.select(this).attr('stroke-width',2);
              highlightFor('t', d._idx);
            })
            .on('mousemove', function(event){
              const pt = d3.pointer(event, container);
              // Keep tooltip within container bounds and away from labels
              const cw = container.clientWidth || 0;
              const ch = container.clientHeight || 0;
              const tw = tooltip.node().offsetWidth || 0;
              const th = tooltip.node().offsetHeight || 0;
              let left = pt[0] + 12;
              let top = pt[1] + 12;
              if (top < 30) top = 34;
              if (top > ch - 34 - th) top = ch - 34 - th;
              if (left + tw > cw - 4) left = Math.max(4, pt[0] - 12 - tw);
              if (top + th > ch - 4) top = Math.max(4, pt[1] - 12 - th);
              tooltip.style('left', left + 'px').style('top', top + 'px');
            })
            .on('mouseout', function(){
              tooltip.style('display','none');
              d3.select(this).attr('stroke-width',0.5);
              clearHighlight();
            });

          const edges = g.append('g').attr('opacity',0.7);
          const eSel = edges.selectAll('path.edge')
            .data(data.edges || [])
            .enter().append('path')
            .attr('class','edge')
            .attr('fill','none')
            .attr('stroke', d=> {
              if (d.type === 'both') return '#7b3294';       // purple
              if (d.type === 'cos') return '#2c7bb6';        // blue
              if (d.type === 'pfam') return '#fdae61';       // orange
              return '#999999';                               // fallback/grey
            })
            .attr('stroke-width', d=> {
              if (d.type === 'cos' || d.type === 'both') {
                const c = Math.max(0, Math.min(1, +d.cos || 0.5));
                return 2 + 4*c; // thicker scaling for cosine/both
              }
              if (d.type === 'pfam') {
                const p = Math.max(0, Math.min(3, +d.pf || 1));
                return 2 + 2*(p/3); // thicker PFAM lines
              }
              return 2.0; // fallback thickness
            })
            .attr('stroke-opacity', 0.6)
            .attr('d', d=> {
              const qs = data.query_locus[d.source] || {start: qMin, end: qMin};
              const ts = data.target_locus[d.target] || {start: tMin, end: tMin};
              const x1 = (qX(qs.start) + qX(qs.end)) / 2, y1 = qY + geneH/2;
              const x2 = (tX(ts.start) + tX(ts.end)) / 2, y2 = tY + geneH/2;
              // Straight line connection
              return 'M' + x1 + ',' + y1 + ' L' + x2 + ',' + y2;
            })
            .append('title').text(d=> {
              let parts = [];
              if (d.cos != null) parts.push('cosine=' + (+d.cos).toFixed(3));
              if (d.pf != null) parts.push('pfam_overlap=' + d.pf);
              if (!parts.length) parts.push('fallback');
              return parts.join(', ');
            });

          // Labels for loci
          const topLabel = (data.query_name || 'Query locus');
          const botLabel = (data.target_name || 'Target locus');
          g.append('text')
            .attr('x', 0)
            .attr('y', qY - 20)
            .attr('fill', '#000')
            .attr('font-size', 12)
            .attr('font-family', 'sans-serif')
            .text(topLabel);
          g.append('text')
            .attr('x', 0)
            .attr('y', tY + geneH + 18)
            .attr('fill', '#000')
            .attr('font-size', 12)
            .attr('font-family', 'sans-serif')
            .text(botLabel);

          // Legend (top-right)
          const legend = g.append('g');
          const lx = Math.max(0, innerW - 260), ly = 8;
          legend.append('rect').attr('x', lx-6).attr('y', ly-6).attr('width', 252).attr('height', 60)
                .attr('fill', 'rgba(255,255,255,0.85)').attr('stroke', '#ccc');
          legend.append('rect').attr('x', lx).attr('y', ly).attr('width', 14).attr('height', 14).attr('fill', '#2E8B57').attr('stroke','#000');
          legend.append('rect').attr('x', lx+18).attr('y', ly).attr('width', 14).attr('height', 14).attr('fill', '#FF6347').attr('stroke','#000');
          legend.append('text').attr('x', lx+36).attr('y', ly+11).attr('font-size', 12).text('Gene arrows (strand color)');
          // Cosine edge
          legend.append('line').attr('x1', lx).attr('y1', ly+26).attr('x2', lx+36).attr('y2', ly+26).attr('stroke', '#2c7bb6').attr('stroke-width', 3);
          legend.append('text').attr('x', lx+44).attr('y', ly+29).attr('font-size', 12).text('Cosine homology');
          // PFAM edge
          legend.append('line').attr('x1', lx).attr('y1', ly+42).attr('x2', lx+36).attr('y2', ly+42).attr('stroke', '#fdae61').attr('stroke-width', 3);
          legend.append('text').attr('x', lx+44).attr('y', ly+45).attr('font-size', 12).text('PFAM homology');
          // Both edge
          legend.append('line').attr('x1', lx+130).attr('y1', ly+34).attr('x2', lx+166).attr('y2', ly+34).attr('stroke', '#7b3294').attr('stroke-width', 3);
          legend.append('text').attr('x', lx+174).attr('y', ly+37).attr('font-size', 12).text('Both');

          // Cross-track highlighting
          function clearHighlight() {
            eSel.attr('stroke-opacity', 0.6).attr('stroke-width', d=> 1 + 2*(d.score||0.5));
            qGenes.attr('opacity', 1.0).attr('stroke-width', 0.5);
            tGenes.attr('opacity', 1.0).attr('stroke-width', 0.5);
          }

          function highlightFor(side, idx) {
            // Determine matching edges and counterpart indices
            const matches = [];
            (data.edges || []).forEach((ed, i) => {
              if (side === 'q' && ed.source === idx) matches.push({ed, i});
              if (side === 't' && ed.target === idx) matches.push({ed, i});
            });

            // Dim everything
            eSel.attr('stroke-opacity', 0.15);
            qGenes.attr('opacity', 0.4).attr('stroke-width', 0.5);
            tGenes.attr('opacity', 0.4).attr('stroke-width', 0.5);

            // Highlight hovered gene
            if (side === 'q') {
              qGenes.filter(d => d._idx === idx).attr('opacity', 1.0).attr('stroke-width', 2);
            } else {
              tGenes.filter(d => d._idx === idx).attr('opacity', 1.0).attr('stroke-width', 2);
            }

            // Highlight counterpart genes and edges
            matches.forEach(({ed}) => {
              eSel.filter(d => d.source === ed.source && d.target === ed.target)
                  .attr('stroke-opacity', 0.95)
                  .attr('stroke-width', d=> 2.5 + 2*(d.score||0.5));
              qGenes.filter(d => d._idx === ed.source).attr('opacity', 1.0).attr('stroke-width', 2);
              tGenes.filter(d => d._idx === ed.target).attr('opacity', 1.0).attr('stroke-width', 2);
            });
          }

        }

        render();
        if ('ResizeObserver' in window) {
          const ro = new ResizeObserver(() => render());
          ro.observe(container);
        } else {
          window.addEventListener('resize', render);
        }
      } catch (e) {
        const el = document.getElementById('cmp');
        if (el) {
          el.innerHTML = '<div style=\\"color:red; font-family:monospace\\">Component error: ' + (e && e.message ? e.message : e) + '</div>';
        }
      }
    })();
    </script>
    """
    # Escape closing tag sequences in JSON
    safe_json = payload.replace('</', '<\\/')
    html = html.replace('__PAYLOAD_JSON__', safe_json).replace('__HEIGHT__', str(height))
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

# ----- Multi-locus clinker-like helpers (cluster-level) -----
def _load_genes_for_region(genome_id: str, contig_id: str, start_bp: int, end_bp: int, pad_bp: int = 0) -> pd.DataFrame:
    conn = get_database_connection()
    s = max(0, int(start_bp) - int(pad_bp))
    e = int(end_bp) + int(pad_bp)
    q = """
        SELECT gene_id, contig_id, start_pos, end_pos, strand, pfam_domains
        FROM genes
        WHERE genome_id = ? AND contig_id = ? AND start_pos <= ? AND end_pos >= ?
        ORDER BY start_pos, end_pos
    """
    return pd.read_sql_query(q, conn, params=[str(genome_id), str(contig_id), e, s])


def _build_cluster_multiloc_json(cluster_id: int, is_micro: bool, max_tracks: int = 6, use_embeddings: bool = False) -> Dict:
    import numpy as _np
    # Choose representative regions
    regions = compute_display_regions_for_micro_cluster(cluster_id, gap_bp=1000, min_support=1) if is_micro else compute_display_regions_for_cluster(cluster_id, gap_bp=1000, min_support=1)
    if not regions:
        try:
            logger.info(f"multiloc: no regions available (is_micro={is_micro}) for cluster {cluster_id}")
        except Exception:
            pass
        return {}
    # Prefer regions with highest support, then length
    regions = sorted(regions, key=lambda r: (int(r.get('support', 1)), int(r.get('end_bp', 0)) - int(r.get('start_bp', 0))), reverse=True)[:max_tracks]

    # Prepare embeddings lookup if requested
    emb_df = None
    emb_status = {"enabled": bool(use_embeddings), "mapped": 0}
    if use_embeddings:
        try:
            emb_df, emb_path = load_gene_embeddings_df()
            if emb_df is not None:
                emb_df = emb_df.set_index('gene_id')
                if not any(c.startswith('emb_') for c in emb_df.columns):
                    emb_df = None
                else:
                    emb_status["path"] = emb_path
        except Exception:
            emb_df = None

    loci = []
    pfam_sets = []
    for r in regions:
        df = _load_genes_for_region(r['genome_id'], r['contig_id'], int(r['start_bp']), int(r['end_bp']), pad_bp=0)
        if df.empty:
            try:
                logger.info(f"multiloc: region has no genes {r['genome_id']}:{r['contig_id']} {r['start_bp']}-{r['end_bp']}")
            except Exception:
                pass
            continue
        genes = []
        pfset = set()
        for _, row in df.iterrows():
            aa_len = 0
            try:
                aa_len = max(1, int(round((int(row['end_pos']) - int(row['start_pos']) + 1) / 3.0)))
            except Exception:
                aa_len = 1
            gobj = {
                'id': row['gene_id'],
                'contig': str(row.get('contig_id') or ''),
                'start': int(row['start_pos']),
                'end': int(row['end_pos']),
                'strand': int(1 if row['strand'] in (1, '+') else -1),
                'pfam': row.get('pfam_domains', '') or '',
                'aa': int(aa_len),
            }
            # Attach normalized embedding if available
            if use_embeddings and emb_df is not None and row['gene_id'] in emb_df.index:
                try:
                    import numpy as _np
                    cols = [c for c in emb_df.columns if c.startswith('emb_')]
                    vec = emb_df.loc[row['gene_id'], cols].to_numpy(dtype='float32', copy=False)
                    nv = vec / (_np.linalg.norm(vec) + 1e-8)
                    gobj['emb'] = [float(x) for x in nv.tolist()]
                    emb_status["mapped"] += 1
                except Exception:
                    pass
            elif use_embeddings and emb_df is not None:
                # Fallback: try contig + start coordinate match if parquet includes these columns
                try:
                    cols = set(emb_df.columns)
                    if {'contig_id','start'}.issubset(cols):
                        sub = emb_df[(emb_df['contig_id'] == str(gobj['contig'])) & (emb_df['start'] == int(gobj['start']))]
                        if not sub.empty:
                            import numpy as _np
                            ecols = [c for c in emb_df.columns if c.startswith('emb_')]
                            vec = sub.iloc[0][ecols].to_numpy(dtype='float32', copy=False)
                            nv = vec / (_np.linalg.norm(vec) + 1e-8)
                            gobj['emb'] = [float(x) for x in nv.tolist()]
                            emb_status["mapped"] += 1
                except Exception:
                    pass
            genes.append(gobj)
            if row.get('pfam_domains'):
                for t in str(row['pfam_domains']).split(';'):
                    t = t.strip()
                    if t:
                        pfset.add(t)
        loci.append({
            'name': f"{r['genome_id']}:{r['contig_id']} [{int(r['start_bp'])}-{int(r['end_bp'])}]",
            'genome_id': r['genome_id'],
            'contig_id': r['contig_id'],
            'start': int(r['start_bp']),
            'end': int(r['end_bp']),
            'genes': genes,
        })
        pfam_sets.append(pfset)
    if not loci:
        try:
            logger.info("multiloc: loci empty after loading genes")
        except Exception:
            pass
        return {}

    # Pairwise similarity (PFAM Jaccard; optionally add embeddings cosine matches)
    L = len(loci)
    sim = [[0.0]*L for _ in range(L)]
    # emb_df already prepared above for gene-level embeddings
    def pf_jacc(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        u = a | b
        if not u:
            return 0.0
        return len(a & b) / float(len(u))

    for i in range(L):
        for j in range(i+1, L):
            s = pf_jacc(pfam_sets[i], pfam_sets[j])
            if use_embeddings and emb_df is not None:
                # Add small bonus from cosine matches above 0.92
                a_ids = [g['id'] for g in loci[i]['genes'] if g['id'] in emb_df.index]
                b_ids = [g['id'] for g in loci[j]['genes'] if g['id'] in emb_df.index]
                if a_ids and b_ids:
                    A = emb_df.loc[a_ids, [c for c in emb_df.columns if c.startswith('emb_')]].to_numpy(dtype='float32', copy=False)
                    B = emb_df.loc[b_ids, [c for c in emb_df.columns if c.startswith('emb_')]].to_numpy(dtype='float32', copy=False)
                    # normalize
                    A = A / (_np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                    B = B / (_np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
                    S = A @ B.T
                    matches = int((S >= 0.92).sum())
                    s += min(0.2, 0.02 * matches)  # cap small bonus
            sim[i][j] = sim[j][i] = float(min(1.0, s))

    # Order loci greedily by similarity (path through high-sim neighbors)
    order = list(range(L))
    if L > 2:
        totals = [sum(sim[i]) for i in range(L)]
        start = max(range(L), key=lambda i: totals[i])
        used = {start}
        seq = [start]
        while len(seq) < L:
            last = seq[-1]
            nxt = None
            best = -1.0
            for j in range(L):
                if j in used:
                    continue
                if sim[last][j] > best:
                    best = sim[last][j]
                    nxt = j
            if nxt is None:
                nxt = next(i for i in range(L) if i not in used)
            seq.append(nxt)
            used.add(nxt)
        order = seq

    loci_ordered = [loci[i] for i in order]

    # Build edges between adjacent tracks: one best match per gene (cosine preferred, PFAM fallback)
    edges = []
    for k in range(len(loci_ordered)-1):
        a = loci_ordered[k]
        b = loci_ordered[k+1]
        # pre-split PFAM tokens for speed
        a_pf = [
            [t.strip() for t in str(g.get('pfam','') or '').split(';') if t.strip()]
            for g in a['genes']
        ]
        b_pf = [
            [t.strip() for t in str(g.get('pfam','') or '').split(';') if t.strip()]
            for g in b['genes']
        ]
        a_pf_sets = [set(lst) for lst in a_pf]
        b_pf_sets = [set(lst) for lst in b_pf]

        # Optional cosine helper using embeddings if available
        def gene_cos_pair(ga, gb):
            try:
                import numpy as _np
                # Prefer pre-attached vectors on gene objects (set during locus load)
                va = ga.get('emb', None)
                vb = gb.get('emb', None)
                if isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb) and len(va) > 0:
                    return float(_np.dot(_np.array(va, dtype='float32'), _np.array(vb, dtype='float32')))
                # Fallback: lookup in emb_df by gene_id if present
                if emb_df is None:
                    return None
                ida = str(ga.get('id','')); idb = str(gb.get('id',''))
                if ida not in emb_df.index or idb not in emb_df.index:
                    return None
                cols = [c for c in emb_df.columns if c.startswith('emb_')]
                va = emb_df.loc[ida, cols].to_numpy(dtype='float32', copy=False)
                vb = emb_df.loc[idb, cols].to_numpy(dtype='float32', copy=False)
                va = va / (_np.linalg.norm(va) + 1e-8)
                vb = vb / (_np.linalg.norm(vb) + 1e-8)
                return float(va.dot(vb))
            except Exception:
                return None

        for ai, ga in enumerate(a['genes']):
            best_score = -1e9
            best_bi = None
            best_pf = 0
            best_cos = None
            for bi, gb in enumerate(b['genes']):
                cosv = gene_cos_pair(ga, gb)
                pf_cnt = len(a_pf_sets[ai] & b_pf_sets[bi])
                # scoring: cosine dominates if present; else PFAM overlap count
                score = (cosv if (cosv is not None and _np.isfinite(cosv)) else (0.5 if pf_cnt>0 else -1))
                if score > best_score:
                    best_score = score
                    best_bi = bi
                    best_pf = pf_cnt
                    best_cos = cosv
            if best_bi is not None:
                edges.append({
                    'a': k, 'b': k+1, 'ai': int(ai), 'bi': int(best_bi),
                    'type': 'emb' if (best_cos is not None) else 'pfam',
                    'pf': int(best_pf), 'cos': (float(best_cos) if best_cos is not None else None)
                })

    return {'loci': loci_ordered, 'edges': edges, 'embeddings': emb_status}


def _render_multiloc_d3(data: Dict, width: int = 0, height: int = 500):
    """Render stacked multi-locus clinker-like view connecting adjacent tracks by PFAM matches."""
    import streamlit.components.v1 as components
    import json as _json
    if not data or not data.get('loci'):
        return
    html = """
    <div id=\"cluster-multiloc\" style=\"width:100%; position:relative; padding:8px 16px; box-sizing:border-box;\"></div>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
      const container = document.getElementById('cluster-multiloc');
      const data = __DATA__;
      const margin = {top: 10, right: 20, bottom: 20, left: 20};
      const W = __WIDTH__;
      const H = __HEIGHT__;
      const baseInnerW = (W>0?W:container.clientWidth) - margin.left - margin.right;
      const minZoom = 0.6, maxZoom = 3.0;
      const rowH = 60;
      const rowGap = 20;
      const contentH = (rowH+rowGap)*data.loci.length;
      const svg = d3.select(container).append('svg')
          .attr('width', baseInnerW + margin.left + margin.right)
          .attr('height', margin.top + margin.bottom + contentH)
          .style('background', '#fff');
      const root = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const panG = root.append('g').attr('class','panzoom');

      // Tooltip similar to block explorer
      const tooltip = d3.select(container).append('div')
        .style('position','absolute')
        .style('pointer-events','none')
        .style('background','#fff')
        .style('border','1px solid #ccc')
        .style('padding','6px 8px')
        .style('font','12px monospace')
        .style('border-radius','4px')
        .style('box-shadow','0 2px 6px rgba(0,0,0,0.15)')
        .style('display','none')
        .style('z-index','10');

      function colorPF(p) { let h=0; for (let i=0;i<p.length;i++) { h=(h*31 + p.charCodeAt(i))>>>0; } const r=(h&0xFF), g2=((h>>8)&0xFF), b=((h>>16)&0xFF); return `rgba(${r},${g2},${b},0.9)`; }

      // Controls and state
      const controls = d3.select(container).append('div').style('margin','6px 0');
      let offsets = data.loci.map(() => 0);
      let flips = data.loci.map(() => false);
      let tau = 0.90;
      let view = d3.zoomIdentity.scale(1.25);
      controls.append('button').text('Reset alignment').on('click', () => { offsets = offsets.map(() => 0); redraw(); });
      controls.append('button').text('Reset view').style('margin-left','8px').on('click', () => { svg.transition().duration(150).call(zoomBehavior.transform, d3.zoomIdentity.scale(1.25)); });
      controls.append('button').text('Flip all to forward').style('margin-left','8px').on('click', () => { const refOri=rowOrientation(0); flips = flips.map((f,i)=> rowOrientation(i)===refOri?false:true); redraw(); });
      controls.append('button').text('Zoom -').style('margin-left','8px').on('click', () => { const k = Math.max(minZoom, +(view.k-0.1).toFixed(2)); svg.transition().duration(120).call(zoomBehavior.scaleTo, k); });
      controls.append('button').text('Zoom +').style('margin-left','4px').on('click', () => { const k = Math.min(maxZoom, +(view.k+0.1).toFixed(2)); svg.transition().duration(120).call(zoomBehavior.scaleTo, k); });
      const zoomLbl = controls.append('span').style('margin-left','8px').style('color','#fff').text(() => ` (${(view.k).toFixed(2)}x)`);
      controls.append('span').style('margin-left','12px').style('color','#fff').style('font-weight','600').text('Cosine τ:');
      const tauInput = controls.append('input')
        .attr('type','range').attr('min','0.70').attr('max','0.99').attr('step','0.01').attr('value','0.90')
        .on('input', function(){ tau=+this.value; tauLbl.text(` ${tau.toFixed(2)}`); drawEdges(); });
      const tauLbl = controls.append('span').style('margin-left','6px').style('color','#fff').text(` ${tau.toFixed(2)}`);

      // Legend (edge meaning)
      const legend = d3.select(container).append('div').style('margin','4px 0 8px 0').style('font','12px sans-serif').style('color','#fff');
      function legendItem(color, label){
        const item = legend.append('span').style('display','inline-flex').style('align-items','center').style('margin-right','14px');
        item.append('span').style('display','inline-block').style('width','18px').style('height','3px').style('margin-right','6px').style('background', color).style('border','1px solid rgba(255,255,255,0.5)');
        item.append('span').text(label);
      }
      legendItem('#2c7bb6', 'Cosine-only');
      legendItem('#fdae61', 'PFAM-only');
      legendItem('#7b3294', 'Both');
      try {
        const emb = (data.embeddings || {});
        const msg = (emb.enabled && emb.mapped>0) ? `Embeddings ON (${emb.mapped.toLocaleString()} genes mapped)` : 'Embeddings OFF (PFAM-only)';
        legend.append('span').style('margin-left','14px').style('opacity','0.85').text(msg);
      } catch (e) {}
      legend.append('div')
        .style('margin-top','6px')
        .style('opacity','0.85')
        .text('Tips: drag background to pan • wheel or buttons to zoom • click a gene to center homologs • double-click a gene or click row label to flip • drag a row to nudge • use Cosine τ to filter edges.');

      function rowOrientation(idx){ const genes = data.loci[idx].genes||[]; let s=0; genes.forEach(g=>{ s += (g.strand>=0?1:-1); }); return s>=0?1:-1; }

      const rowsG = [];
      const gapPx = 16; // fixed inter-gene gap in pixels
      const minGenePx = 3; // minimal visible width per gene
      let pxPerAA = 1; // computed from max row extent
      data.loci.forEach((loc, idx) => {
        const y0 = idx*(rowH+rowGap);
        const minX = d3.min(loc.genes, d => d.start) || 0;
        const maxX = d3.max(loc.genes, d => d.end) || 1;
        const baseX = d3.scaleLinear().domain([minX, maxX]).range([0, baseInnerW]);
        panG.append('text').attr('x', 0).attr('y', y0-4).text(loc.name).attr('font-size','10px').attr('fill','#333')
          .style('cursor','pointer').on('click', () => { flips[idx] = !flips[idx]; redraw(); });
        const rowSel = panG.append('g').attr('transform', `translate(0,${y0})`);
        const row = rowSel.node();
        row._baseX = baseX; rowsG.push(row);
      });

      // Precompute per-row gene layouts in pixels with a global px-per-aa so equal protein lengths have equal widths across rows
      function computeLayouts(){
        // determine global pxPerAA from the row with max total aa length
        let maxSumAA = 1;
        let maxGenes = 1;
        data.loci.forEach(loc => {
          const aaSum = (loc.genes||[]).reduce((s,g)=> s + Math.max(1, +g.aa || 1), 0);
          if (aaSum > maxSumAA) maxSumAA = aaSum;
          const n = (loc.genes||[]).length;
          if (n > maxGenes) maxGenes = n;
        });
        pxPerAA = Math.max(0.05, (baseInnerW - gapPx * Math.max(0, maxGenes-1)) / maxSumAA);
        // assign layout per row
        data.loci.forEach((loc, idx) => {
          const row = rowsG[idx];
          let x = 0;
          const lay = [];
          (loc.genes||[]).forEach(g => {
            const w = Math.max(minGenePx, (Math.max(1, +g.aa || 1)) * pxPerAA);
            lay.push({start:x, w:w});
            x += w + gapPx;
          });
          row._layout = lay;
          row._rowWidth = x > 0 ? (x - gapPx) : 0;
        });
      }
      computeLayouts();

      function rowScale(idx){ const base = rowsG[idx]._baseX; const dom=base.domain().slice(); return d3.scaleLinear().domain(flips[idx]?dom.reverse():dom).range([0, baseInnerW]); }

      const lociIndexMaps = data.loci.map(loc => { const m=new Map(); loc.genes.forEach((g,i)=>m.set(i,g)); return {map:m}; });
      const edgesG = panG.append('g').attr('class','edges');
      // Render edges beneath gene glyphs
      try { edgesG.lower(); } catch (e) {}

      // Background overlay for pan/zoom capture
      const overlay = root.append('rect')
        .attr('x',0).attr('y',0).attr('width', baseInnerW).attr('height', contentH)
        .style('fill','transparent')
        .style('pointer-events','none')
        .style('cursor','grab');

      const zoomBehavior = d3.zoom().scaleExtent([minZoom, maxZoom]).on('zoom', (event) => {
        view = event.transform;
        panG.attr('transform', view.toString());
        zoomLbl.text(` (${(view.k).toFixed(2)}x)`);
      });
      svg.call(zoomBehavior).on('dblclick.zoom', null);
      svg.call(zoomBehavior.transform, view);

      function geneCenterX(rowIdx, geneIdx){
        const row = rowsG[rowIdx];
        const lay = (row._layout||[])[geneIdx] || {start:0,w:0};
        const rowW = row._rowWidth || 0;
        const local = flips[rowIdx] ? (rowW - (lay.start + lay.w/2)) : (lay.start + lay.w/2);
        return local + offsets[rowIdx];
      }

      function drawEdges(){
        const allEdges = data.edges || [];
        const E = allEdges.filter(ed => {
          const hasCos = (ed.cos != null && isFinite(ed.cos));
          if (hasCos) return (+ed.cos) >= tau;
          // keep PFAM-only edges regardless of tau if they have overlap
          return ((+ed.pf || 0) > 0);
        });
        const esel = edgesG.selectAll('line.edge').data(E, d=> `${d.a}-${d.b}-${d.ai}-${d.bi}`);
        esel.exit().remove();
        const enter = esel.enter().append('line').attr('class','edge')
          .attr('stroke', d => { const hasCos=(d.cos!=null&&isFinite(d.cos)); const hasPf=(+d.pf||0)>0; if (hasCos && hasPf) return '#7b3294'; if (hasCos) return '#2c7bb6'; if (hasPf) return '#fdae61'; return '#999'; })
          .attr('stroke-width', 1.0)
          .on('mouseover', function(event, ed){ d3.select(this).attr('stroke-width',2.0); const cos=(ed.cos!=null&&isFinite(ed.cos))?(+ed.cos).toFixed(3):'n/a'; const pf=(ed.pf!=null)?ed.pf:'n/a'; tooltip.style('display','block').html('Match: '+(ed.type||'pfam')+'<br>Cosine: '+cos+'<br>PFAM overlap: '+pf); })
          .on('mousemove', function(event){ const cw=container.clientWidth||0,ch=container.clientHeight||0; const tw=tooltip.node().offsetWidth||0, th=tooltip.node().offsetHeight||0; const pt=d3.pointer(event,container); let left=pt[0]+12, top=pt[1]+12; if (left+tw>cw-4) left=Math.max(4, pt[0]-12-tw); if (top+th>ch-4) top=Math.max(4, pt[1]-12-th); tooltip.style('left', left+'px').style('top', top+'px'); })
          .on('mouseout', function(){ d3.select(this).attr('stroke-width',1.0); tooltip.style('display','none'); });
        enter.merge(esel)
          .attr('stroke', d => { const hasCos=(d.cos!=null&&isFinite(d.cos)); const hasPf=(+d.pf||0)>0; if (hasCos && hasPf) return '#7b3294'; if (hasCos) return '#2c7bb6'; if (hasPf) return '#fdae61'; return '#999'; })
          .attr('x1', d => geneCenterX(d.a, d.ai))
          .attr('y1', d => d.a*(rowH+rowGap) + rowH/2)
          .attr('x2', d => geneCenterX(d.b, d.bi))
          .attr('y2', d => d.b*(rowH+rowGap) + rowH/2);
      }

      function redraw(){
        // Keep fixed SVG size; pan/zoom handled by zoomBehavior on panG
        zoomLbl.text(` (${(view.k).toFixed(2)}x)`);
        data.loci.forEach((loc, idx) => {
          const rowNode = rowsG[idx]; const row = d3.select(rowNode); const yMid=rowH/2;
          row.attr('transform', `translate(${offsets[idx]},${idx*(rowH+rowGap)})`);
          const genesSel = row.selectAll('path.gene').data(loc.genes, d=>d.id);
          genesSel.enter().append('path').attr('class','gene'); genesSel.exit().remove();
          row.selectAll('path.gene')
            .attr('d', (d,i) => { const lay=(rowNode._layout||[])[i]||{start:0,w:0}; const rowW=rowNode._rowWidth||0; const w=lay.w; const x0 = flips[idx] ? (rowW - (lay.start + lay.w)) : lay.start; const dir=d.strand>=0?1:-1; const head=6; const x1=x0, x2=x0+w; return dir>=0? `M ${x1},${yMid-8} L ${x2-head},${yMid-8} L ${x2},${yMid} L ${x2-head},${yMid+8} L ${x1},${yMid+8} Z` : `M ${x2},${yMid-8} L ${x1+head},${yMid-8} L ${x1},${yMid} L ${x1+head},${yMid+8} L ${x2},${yMid+8} Z`; })
            .attr('fill', d => colorPF((d.pfam||'').split(';')[0]||''))
            .attr('stroke','rgba(0,0,0,0.2)')
            .attr('stroke-width',0.8)
            .style('cursor','pointer')
            .on('click', (event,d)=> centerOn(idx,d))
            .on('dblclick', ()=> { flips[idx] = !flips[idx]; redraw(); })
            .on('mouseover', function(event, d){ const strand=(d.strand>=0? '+':'-'); const len=Math.max(0, Math.round(Math.abs(d.end-d.start))); tooltip.style('display','block').html('<b>'+d.id+'</b><br>Length: '+len.toLocaleString()+' aa<br>Strand: '+strand+'<br>PFAM: '+(d.pfam||'None')); })
            .on('mousemove', function(event){ const cw=container.clientWidth||0,ch=container.clientHeight||0; const tw=tooltip.node().offsetWidth||0, th=tooltip.node().offsetHeight||0; const pt=d3.pointer(event,container); let left=pt[0]+12, top=pt[1]+12; if (left+tw>cw-4) left=Math.max(4, pt[0]-12-tw); if (top+th>ch-4) top=Math.max(4, pt[1]-12-th); tooltip.style('left', left+'px').style('top', top+'px'); })
            .on('mouseout', ()=> tooltip.style('display','none'));
          let base=row.selectAll('line.base').data([0]); base=base.enter().append('line').attr('class','base').merge(base); const rowW=rowNode._rowWidth||baseInnerW; base.attr('x1',0).attr('x2',Math.max(baseInnerW,rowW)).attr('y1',yMid).attr('y2',yMid).attr('stroke','#ccc').attr('stroke-width',0.5);
        });
        drawEdges();
      }

      function cosine(a,b){ if (!a||!b||!a.emb||!b.emb) return null; let s=0,na=0,nb=0; for (let i=0;i<a.emb.length && i<b.emb.length;i++){ s+=a.emb[i]*b.emb[i]; na+=a.emb[i]*a.emb[i]; nb+=b.emb[i]*b.emb[i]; } if (na<=0||nb<=0) return null; return s/Math.sqrt(na*nb); }
      function pfOverlap(a,b){ const sA=new Set((a.pfam||'').split(';').map(x=>x.trim()).filter(Boolean)); const sB=new Set((b.pfam||'').split(';').map(x=>x.trim()).filter(Boolean)); let c=0; sA.forEach(x=>{ if(sB.has(x)) c++; }); return c; }
      function centerOn(rowIndex, gene){
        const targetX = baseInnerW/2;
        // find index of clicked gene in its row
        const seedIdx = (data.loci[rowIndex].genes||[]).findIndex(g => g.id === gene.id);
        if (seedIdx < 0) { return; }
        const matches = {}; // row -> gene index
        matches[rowIndex] = seedIdx;
        // helper to select best edge match between adjacent rows, preferring highest cosine
        function bestEdge(fromRow, fromIdx, toRow){
          let bestCos = null, bestCosScore = -Infinity;
          let bestPf = null, bestPfScore = -Infinity;
          (data.edges||[]).forEach(ed => {
            if ((ed.a === fromRow && ed.b === toRow && ed.ai === fromIdx) ||
                (ed.b === fromRow && ed.a === toRow && ed.bi === fromIdx)){
              const toIdx = (ed.a === fromRow) ? ed.bi : ed.ai;
              const cos = (ed.cos != null && isFinite(ed.cos)) ? +ed.cos : -Infinity;
              const pf  = (+ed.pf || 0);
              if (isFinite(cos) && cos > bestCosScore){ bestCosScore = cos; bestCos = {row: toRow, idx: toIdx, score: cos}; }
              if (!isFinite(cos) && pf > bestPfScore){ bestPfScore = pf; bestPf = {row: toRow, idx: toIdx, score: pf}; }
            }
          });
          if (bestCos && bestCos.score >= tau) return bestCos; // prefer cosine meeting tau
          if (bestPf && bestPf.score > 0) return bestPf; // fallback on PFAM when cosine missing or below tau
          return null;
        }
        // propagate downwards (towards row 0)
        for (let r = rowIndex; r > 0; r--){
          const prev = bestEdge(r, matches[r], r-1);
          if (!prev) break;
          matches[r-1] = prev.idx;
        }
        // propagate upwards (towards last row)
        for (let r = rowIndex; r < data.loci.length-1; r++){
          const nxt = bestEdge(r, matches[r], r+1);
          if (!nxt) break;
          matches[r+1] = nxt.idx;
        }
        // apply absolute offsets for matched rows to center the chosen genes
        const halfW = baseInnerW/2;
        function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
        Object.keys(matches).forEach(k => {
          const ri = +k; const gi = matches[ri];
          const row = rowsG[ri]; const lay = (row._layout||[])[gi] || {start:0,w:0}; const rowW=row._rowWidth||0;
          const centerLocal = flips[ri] ? (rowW - (lay.start + lay.w/2)) : (lay.start + lay.w/2);
          // Solve: (centerLocal + offset)*view.k + view.x = targetX
          const desired = (targetX - view.x)/view.k - centerLocal;
          offsets[ri] = clamp(desired, -halfW*2, halfW*2);
        });
        redraw();
      }

      // Enable per-row drag to manually align rows
      function attachRowDrag(row, idx){
        row.style('cursor','grab')
          .call(d3.drag()
            .on('start', function(){ d3.select(this).style('cursor','grabbing'); })
            .on('drag', function(event){ offsets[idx] += event.dx; redraw(); })
            .on('end', function(){ d3.select(this).style('cursor','grab'); })
          );
      }

      redraw();

      // Attach drag after first draw
      rowsG.forEach((row, idx) => attachRowDrag(d3.select(row), idx));
    </script>
    """
    html = html.replace('__DATA__', _json.dumps(data)).replace('__WIDTH__', str(width if width else 0)).replace('__HEIGHT__', str(height))
    components.html(html, height=height)


if __name__ == "__main__":
    main()
