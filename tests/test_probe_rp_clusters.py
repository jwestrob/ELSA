import sqlite3
from pathlib import Path


DB_PATH = Path("genome_browser/genome_browser.db")


def _fetch_rp_hits(conn, markers):
    terms = [f"%{m.lower()}%" for m in markers]
    like_clause = " OR ".join(["LOWER(g.pfam_domains) LIKE ?" for _ in markers])
    sql = f"""
    SELECT gbm.block_id, sb.cluster_id, g.gene_id, g.pfam_domains
    FROM gene_block_mappings gbm
    JOIN genes g ON gbm.gene_id = g.gene_id
    JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
    WHERE {like_clause}
    """
    cur = conn.execute(sql, terms)
    rows = cur.fetchall()
    return rows


def test_probe_rp_fragmentation_report():
    assert DB_PATH.exists(), "Expected genome_browser/genome_browser.db to exist; run 'elsa analyze' first."
    conn = sqlite3.connect(str(DB_PATH))
    try:
        markers = ["Ribosomal_L22", "Ribosomal_L3", "Ribosomal_L4", "Ribosomal_S19"]
        rows = _fetch_rp_hits(conn, markers)
        # Aggregate per block marker hits
        block_hits = {}
        for block_id, cluster_id, gene_id, pfams in rows:
            key = int(block_id)
            if key not in block_hits:
                block_hits[key] = {"cluster": int(cluster_id), "markers": set(), "genes": []}
            for m in markers:
                if pfams and (m.lower() in str(pfams).lower()):
                    block_hits[key]["markers"].add(m)
            block_hits[key]["genes"].append(gene_id)

        # Summarize blocks with >= 3 marker hits (canonical-ish RP locus sample)
        canonical_blocks = {bid: info for bid, info in block_hits.items() if len(info["markers"]) >= 3}
        clusters = {}
        for bid, info in canonical_blocks.items():
            clusters.setdefault(info["cluster"], []).append(bid)

        print(f"RP probe markers: {markers}")
        print(f"Total blocks with any marker: {len(block_hits)}")
        print(f"Blocks with >=3 markers (canonical-ish): {len(canonical_blocks)}")
        print(f"Distinct clusters containing canonical-ish blocks: {len(clusters)}")
        # Show top 10 clusters by count of canonical-like blocks
        top = sorted(((cid, len(bids)) for cid, bids in clusters.items()), key=lambda x: -x[1])[:10]
        for cid, cnt in top:
            sample = canonical_blocks[clusters[cid][0]]["markers"]
            print(f"  Cluster {cid}: {cnt} canonical-like blocks; sample markers={sorted(sample)}")

        # We expect at least one canonical-like RP cluster; fragmentation indicated by >1 clusters
        assert len(canonical_blocks) >= 1
        # Print example scattered clusters for visibility
    finally:
        conn.close()
