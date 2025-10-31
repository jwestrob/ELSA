import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def integrate_micro_into_db(db_path: Path, sidecar_dir: Path) -> Dict[str, int]:
    """Integrate pair-first micro results into the browser DB.

    - Clears existing micro projections from clusters/syntenic_blocks/gene_block_mappings
    - Loads micro tables from DB if present; otherwise from sidecars in sidecar_dir
    - Upserts micro clusters into clusters with an offset after current MAX(cluster_id)
    - Projects micro pairs into syntenic_blocks (block_type='micro')
    - Projects micro_gene_pair_mappings into gene_block_mappings with relative positions

    Returns counts of inserted clusters and blocks.
    """
    db_path = Path(db_path)
    sidecar_dir = Path(sidecar_dir)
    inserted_clusters = 0
    inserted_blocks = 0

    if not db_path.exists():
        raise RuntimeError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Collect existing micro block_ids to clean mappings
        try:
            mids = [int(r[0]) for r in cur.execute("SELECT block_id FROM syntenic_blocks WHERE block_type='micro'").fetchall()]
        except Exception:
            mids = []
        if mids:
            cur.executemany("DELETE FROM gene_block_mappings WHERE block_id = ?", [(m,) for m in mids])
        # Clear prior micro projections
        cur.execute("DELETE FROM syntenic_blocks WHERE block_type='micro'")
        cur.execute("DELETE FROM clusters WHERE cluster_type='micro'")
        conn.commit()

        # Load micro pairs/mappings/clusters (prefer DB tables, else sidecars)
        def _load_df_db_or_csv(q: str, csv_path: Path, cols: List[str]) -> pd.DataFrame:
            try:
                return pd.read_sql_query(q, conn)
            except Exception:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    # ensure required columns
                    miss = [c for c in cols if c not in df.columns]
                    if miss:
                        raise RuntimeError(f"Missing columns in {csv_path}: {miss}")
                    return df
                return pd.DataFrame(columns=cols)

        mbp = _load_df_db_or_csv(
            "SELECT * FROM micro_block_pairs",
            sidecar_dir / 'micro_block_pairs.csv',
            ['block_id','cluster_id','query_genome_id','query_contig_id','query_start_bp','query_end_bp','target_genome_id','target_contig_id','target_start_bp','target_end_bp','q_core_start_bp','q_core_end_bp','t_core_start_bp','t_core_end_bp','orientation','identity','score']
        )
        mgpm = _load_df_db_or_csv(
            "SELECT * FROM micro_gene_pair_mappings",
            sidecar_dir / 'micro_gene_pair_mappings.csv',
            ['block_id','block_role','gene_id','start_pos','end_pos','strand']
        )
        mclus = _load_df_db_or_csv(
            "SELECT * FROM micro_gene_clusters",
            sidecar_dir / 'micro_gene_clusters.csv',
            ['cluster_id','size','genomes']
        )

        if mbp.empty or mclus.empty:
            return {"inserted_clusters": 0, "inserted_blocks": 0}

        # Filter micro pairs that are fully contained within macro spans on both sides
        try:
            filt_ids = _filter_micro_pairs_vs_macro(conn, mbp)
            if filt_ids is not None:
                before = len(mbp)
                mbp = mbp[mbp['block_id'].isin(filt_ids)].copy()
                after = len(mbp)
                # Optionally drop clusters that end up empty later (handled implicitly by projection)
        except Exception:
            # Non-fatal: proceed without additional filtering
            pass

        # Compute display offset and upsert clusters
        row = cur.execute("SELECT COALESCE(MAX(cluster_id),0) FROM clusters").fetchone()
        start_cid = int(row[0] or 0)
        rows = []
        for r in mclus.itertuples(index=False):
            raw = int(r.cluster_id)
            if raw <= 0:
                continue
            rows.append((start_cid + raw, int(r.size), None, None, None, None, None, 'micro'))
        if rows:
            cur.executemany(
                """
                INSERT OR REPLACE INTO clusters
                    (cluster_id, size, consensus_length, consensus_score, diversity, representative_query, representative_target, cluster_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted_clusters = len(rows)

        # Build role counts for length estimate and mappings
        counts = mgpm.groupby(['block_id','block_role']).size().unstack(fill_value=0) if not mgpm.empty else None
        def role_count(bid: int, role: str) -> int:
            try:
                return int(counts.get(role, pd.Series()).get(int(bid), 0)) if counts is not None else 0
            except Exception:
                return 0

        # Project pairs into syntenic_blocks
        sb_rows = []
        for r in mbp.itertuples(index=False):
            q_locus = f"{r.query_genome_id}:{r.query_contig_id}:{int(r.query_start_bp)}-{int(r.query_end_bp)}"
            t_locus = f"{r.target_genome_id}:{r.target_contig_id}:{int(r.target_start_bp)}-{int(r.target_end_bp)}"
            q_len = int(r.query_end_bp) - int(r.query_start_bp)
            t_len = int(r.target_end_bp) - int(r.target_start_bp)
            length = q_len if q_len >= t_len else t_len
            sb_rows.append((
                int(r.block_id),
                start_cid + int(r.cluster_id) if int(r.cluster_id) > 0 else 0,
                q_locus,
                t_locus,
                str(r.query_genome_id), str(r.target_genome_id),
                str(r.query_contig_id), str(r.target_contig_id),
                int(length), float(getattr(r, 'identity', 0.0) or 0.0), float(getattr(r, 'score', 0.0) or 0.0),
                role_count(int(r.block_id), 'query'), role_count(int(r.block_id), 'target'),
                None, None, None, None, None, None,
                'micro'
            ))
        if sb_rows:
            cur.executemany(
                """
                INSERT OR REPLACE INTO syntenic_blocks (
                    block_id, cluster_id, query_locus, target_locus,
                    query_genome_id, target_genome_id,
                    query_contig_id, target_contig_id,
                    length, identity, score,
                    n_query_windows, n_target_windows,
                    query_window_start, query_window_end,
                    target_window_start, target_window_end,
                    query_windows_json, target_windows_json,
                    block_type
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                sb_rows,
            )
            inserted_blocks = len(sb_rows)

        # Project mappings into gene_block_mappings with relative positions per block/role
        if not mgpm.empty:
            rows = []
            for (blk, role), g in mgpm.groupby(['block_id','block_role']):
                g2 = g.sort_values(['start_pos','end_pos'])
                n = len(g2)
                L = max(1, n-1)
                for idx, rr in enumerate(g2.itertuples(index=False)):
                    rel = (idx / L) if L > 0 else 0.0
                    rows.append((str(rr.gene_id), int(blk), str(role), float(rel)))
            if rows:
                for i in range(0, len(rows), 1000):
                    cur.executemany(
                        "INSERT OR REPLACE INTO gene_block_mappings (gene_id, block_id, block_role, relative_position) VALUES (?, ?, ?, ?)",
                        rows[i:i+1000]
                    )
        conn.commit()
        return {"inserted_clusters": inserted_clusters, "inserted_blocks": inserted_blocks}
    finally:
        conn.close()


def _filter_micro_pairs_vs_macro(conn, mbp: pd.DataFrame) -> List[int] | None:
    """Return list of micro pair block_ids to KEEP after removing both-side contained pairs.

    Uses DB tables for macro spans (via gene_block_mappings→genes) and micro spans
    from mbp (preferring core spans). Performs joins in SQLite for efficiency.
    """
    if mbp is None or mbp.empty:
        return None
    # Load mbp into TEMP table
    mbp_cols = ['block_id','cluster_id','query_genome_id','query_contig_id','query_start_bp','query_end_bp','target_genome_id','target_contig_id','target_start_bp','target_end_bp','q_core_start_bp','q_core_end_bp','t_core_start_bp','t_core_end_bp']
    tmp = mbp[mbp_cols].copy()
    tmp.to_sql('_tmp_mbp', conn, if_exists='replace', index=False)
    cur = conn.cursor()
    # Macro spans per side from gene_block_mappings→genes
    cur.execute("DROP TABLE IF EXISTS _macro_spans")
    cur.execute(
        """
        CREATE TEMP TABLE _macro_spans AS
        SELECT 
            gb.block_role AS side,
            g.genome_id,
            g.contig_id,
            MIN(g.start_pos) AS span_start,
            MAX(g.end_pos) AS span_end
        FROM gene_block_mappings gb
        JOIN genes g ON g.gene_id = gb.gene_id
        JOIN syntenic_blocks sb ON sb.block_id = gb.block_id
        WHERE COALESCE(sb.block_type,'macro') != 'micro'
        GROUP BY gb.block_role, g.genome_id, g.contig_id
        """
    )
    # Micro spans per side using core if available
    cur.execute("DROP TABLE IF EXISTS _micro_spans")
    cur.execute(
        """
        CREATE TEMP TABLE _micro_spans AS
        SELECT 'query' AS side, CAST(query_genome_id AS TEXT) AS genome_id, CAST(query_contig_id AS TEXT) AS contig_id,
               COALESCE(q_core_start_bp, query_start_bp) AS start_bp,
               COALESCE(q_core_end_bp, query_end_bp) AS end_bp,
               block_id
        FROM _tmp_mbp
        UNION ALL
        SELECT 'target' AS side, CAST(target_genome_id AS TEXT) AS genome_id, CAST(target_contig_id AS TEXT) AS contig_id,
               COALESCE(t_core_start_bp, target_start_bp) AS start_bp,
               COALESCE(t_core_end_bp, target_end_bp) AS end_bp,
               block_id
        FROM _tmp_mbp
        """
    )
    # Containment per side
    cur.execute("DROP TABLE IF EXISTS _m_contained_q")
    cur.execute(
        """
        CREATE TEMP TABLE _m_contained_q AS
        SELECT DISTINCT m.block_id
        FROM _micro_spans m
        JOIN _macro_spans s
          ON m.side = 'query' AND s.side = 'query'
         AND m.genome_id = s.genome_id AND m.contig_id = s.contig_id
         AND m.start_bp >= s.span_start AND m.end_bp <= s.span_end
        """
    )
    cur.execute("DROP TABLE IF EXISTS _m_contained_t")
    cur.execute(
        """
        CREATE TEMP TABLE _m_contained_t AS
        SELECT DISTINCT m.block_id
        FROM _micro_spans m
        JOIN _macro_spans s
          ON m.side = 'target' AND s.side = 'target'
         AND m.genome_id = s.genome_id AND m.contig_id = s.contig_id
         AND m.start_bp >= s.span_start AND m.end_bp <= s.span_end
        """
    )
    # Keep those not contained on both sides
    rows = cur.execute(
        """
        SELECT block_id FROM _tmp_mbp
        EXCEPT
        SELECT q.block_id FROM _m_contained_q q
        INTERSECT
        SELECT t.block_id FROM _m_contained_t t
        """
    ).fetchall()
    keep = [int(r[0]) for r in rows]
    return keep
