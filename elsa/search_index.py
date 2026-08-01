"""Search over a materialized run: by domain, by genome, by contig.

Finding a cluster by what it contains is the primary way anyone navigates a run
— far more useful than scrolling a ranked list — and it was the largest gap when
the result surface moved off the browser's own database.

Domain search needs the companion Sharur database, because ELSA's result
contract holds synteny and no annotation layer. Genome and contig search need
only the sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb


def _connect(database: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(database), read_only=True)


def search_by_domain(
    database: str | Path,
    annotations_db: str | Path,
    terms: Sequence[str],
    *,
    limit: int = 50,
    source: str = "pfam",
    match_all: bool = False,
) -> List[Dict[str, Any]]:
    """Clusters containing genes annotated with the given domain names.

    Matching is case-insensitive substring, so ``ribosomal`` finds every
    ``Ribosomal_*`` family. ``match_all`` requires a cluster to carry every term
    rather than any of them, which is how you ask for a co-occurrence rather
    than a family.

    Results are ranked by how many distinct query terms a cluster carries, then
    by breadth, so a cluster holding the whole set outranks one holding a single
    member.
    """
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        return []

    connection = _connect(database)
    try:
        connection.execute(f"ATTACH '{str(annotations_db)}' AS _ann (READ_ONLY)")
        try:
            conditions = " OR ".join(["lower(a.name) LIKE ?"] * len(terms))
            # The WHERE inside the CTE already guarantees at least one hit, so a
            # default HAVING would be redundant — and referencing the annotation
            # alias out here puts it out of scope. match_all is the only real
            # constraint: one matched term is not a co-occurrence, so require a
            # distinct hit per requested term.
            having = ("HAVING COUNT(DISTINCT hits.matched_term) >= ?"
                      if match_all else "")

            term_case = " ".join(
                f"WHEN lower(a.name) LIKE ? THEN ?" for _ in terms
            )
            case_params: List[Any] = []
            for term in terms:
                case_params.extend([f"%{term.lower()}%", term.lower()])

            rows = connection.execute(
                f"""
                WITH hits AS (
                    SELECT m.cluster_key,
                           a.name AS domain_name,
                           CASE {term_case} ELSE NULL END AS matched_term
                    FROM current_elsa_cluster_members AS m
                    JOIN _ann.annotations AS a ON a.protein_id = m.protein_id
                    WHERE a.source = ? AND ({conditions})
                )
                SELECT hits.cluster_key,
                       COUNT(DISTINCT hits.domain_name) AS n_domains,
                       COUNT(DISTINCT hits.matched_term) AS n_terms,
                       COUNT(*) AS n_genes,
                       string_agg(DISTINCT hits.domain_name, ';') AS domains,
                       c.genome_support, c.size, c.locus_size_median,
                       c.mean_anchor_count
                FROM hits
                JOIN current_elsa_clusters AS c USING (cluster_key)
                GROUP BY hits.cluster_key, c.genome_support, c.size,
                         c.locus_size_median, c.mean_anchor_count
                {having}
                ORDER BY n_terms DESC, c.genome_support DESC, n_genes DESC
                LIMIT ?
                """,
                [*case_params, source, *[f"%{t.lower()}%" for t in terms],
                 *([len(terms)] if match_all else []), int(limit)],
            ).fetchall()
        finally:
            connection.execute("DETACH _ann")
    finally:
        connection.close()

    return [
        {
            "cluster_key": key,
            "n_matching_domains": int(n_domains),
            "n_matched_terms": int(n_terms),
            "n_matching_genes": int(n_genes),
            "domains": sorted((domains or "").split(";")),
            "genome_support": int(support),
            "n_blocks": int(size),
            "locus_size_median": float(median) if median is not None else None,
            "mean_anchor_count": float(anchors) if anchors is not None else None,
        }
        for (key, n_domains, n_terms, n_genes, domains, support, size,
             median, anchors) in rows
    ]


def search_by_locus(
    database: str | Path,
    *,
    genome: Optional[str] = None,
    contig: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Clusters with a locus on a given genome, or a given contig.

    Both terms are case-insensitive substrings so a partial accession works.
    """
    if not genome and not contig:
        return []
    where: List[str] = []
    params: List[Any] = []
    if genome:
        where.append("lower(l.genome_id) LIKE ?")
        params.append(f"%{genome.lower()}%")
    if contig:
        where.append("lower(l.contig_id) LIKE ?")
        params.append(f"%{contig.lower()}%")

    connection = _connect(database)
    try:
        rows = connection.execute(
            f"""
            SELECT l.cluster_key,
                   COUNT(*) AS n_loci,
                   COUNT(DISTINCT l.genome_id) AS n_genomes,
                   string_agg(DISTINCT l.contig_id, ';') AS contigs,
                   c.genome_support, c.size, c.locus_size_median,
                   c.mean_anchor_count
            FROM current_elsa_cluster_loci AS l
            JOIN current_elsa_clusters AS c USING (cluster_key)
            WHERE {" AND ".join(where)}
            GROUP BY l.cluster_key, c.genome_support, c.size,
                     c.locus_size_median, c.mean_anchor_count
            ORDER BY c.genome_support DESC, n_loci DESC
            LIMIT ?
            """,
            [*params, int(limit)],
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "cluster_key": key,
            "n_matching_loci": int(n_loci),
            "n_genomes_matched": int(n_genomes),
            "contigs": sorted((contigs or "").split(";"))[:10],
            "genome_support": int(support),
            "n_blocks": int(size),
            "locus_size_median": float(median) if median is not None else None,
            "mean_anchor_count": float(anchors) if anchors is not None else None,
        }
        for (key, n_loci, n_genomes, contigs, support, size, median, anchors) in rows
    ]


def run_overview(database: str | Path) -> Dict[str, Any]:
    """Dashboard figures: totals plus the distributions worth seeing up front."""
    connection = _connect(database)
    try:
        totals = connection.execute(
            """
            SELECT COUNT(*) FILTER (WHERE cluster_kind = 'cluster'),
                   COUNT(*) FILTER (WHERE cluster_kind = 'singleton'),
                   MAX(genome_support),
                   quantile_cont(genome_support, 0.5)
                     FILTER (WHERE cluster_kind = 'cluster')
            FROM current_elsa_clusters
            """
        ).fetchone()
        genomes = connection.execute(
            "SELECT COUNT(DISTINCT genome_id) FROM current_elsa_cluster_loci"
        ).fetchone()[0]
        tiers = connection.execute(
            """
            SELECT CASE
                     WHEN mean_anchor_count < 3 THEN 'minimal'
                     WHEN mean_anchor_count < 6 THEN 'moderate'
                     WHEN mean_anchor_count < 20 THEN 'strong'
                     ELSE 'very_strong' END AS tier,
                   COUNT(*)
            FROM current_elsa_clusters
            WHERE cluster_kind = 'cluster'
            GROUP BY 1
            """
        ).fetchall()
        sizes = connection.execute(
            """
            SELECT n_genes, COUNT(*)
            FROM current_elsa_cluster_loci
            WHERE n_genes <= 60
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        truncation = connection.execute(
            """
            SELECT SUM(truncated_locus_count), SUM(locus_count)
            FROM current_elsa_clusters
            WHERE cluster_kind = 'cluster'
            """
        ).fetchone()
    finally:
        connection.close()

    truncated, total_loci = truncation if truncation else (None, None)
    return {
        "n_clusters": int(totals[0] or 0),
        "n_singletons": int(totals[1] or 0),
        "n_genomes": int(genomes or 0),
        "max_genome_support": int(totals[2] or 0),
        "median_genome_support": float(totals[3]) if totals[3] is not None else None,
        "evidence_tiers": {tier: int(count) for tier, count in tiers},
        "locus_size_histogram": [
            {"n_genes": int(size), "count": int(count)} for size, count in sizes
        ],
        "truncated_locus_fraction": (
            float(truncated) / float(total_loci)
            if truncated is not None and total_loci else None
        ),
    }


__all__ = ["run_overview", "search_by_domain", "search_by_locus"]


def locate_in_cluster(
    database: str | Path,
    annotations_db: str | Path,
    cluster_key: str,
    terms: Sequence[str],
    *,
    source: str = "pfam",
    max_loci: int = 40,
) -> Dict[str, Any]:
    """Where inside each locus do the matching genes sit?

    Finding the cluster is half an answer. A cluster whose loci run 90 genes
    holds the three a caller asked about somewhere inside, and pointing at the
    cluster leaves them to scan it.

    The benchmark measures this as boundary precision — what fraction of the
    block is the thing you were looking for — against curated operons. The same
    quantity is computable against the caller's own query with no ground truth
    at all, which is what ``match_fraction`` reports here, so it works on any
    dataset rather than only on the annotated benchmark cohorts.

    Offsets are ``locus_order``: position within the locus, so 0 is its first
    gene regardless of where the locus sits on the contig.
    """
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        return {"cluster_key": cluster_key, "terms": [], "loci": [], "summary": {}}

    connection = _connect(database)
    try:
        connection.execute(f"ATTACH '{str(annotations_db)}' AS _ann (READ_ONLY)")
        try:
            conditions = " OR ".join(["lower(a.name) LIKE ?"] * len(terms))
            rows = connection.execute(
                f"""
                SELECT m.locus_key, m.genome_id, m.contig_id, m.protein_id,
                       m.locus_order, string_agg(DISTINCT a.name, ';') AS domains
                FROM current_elsa_cluster_members AS m
                JOIN _ann.annotations AS a ON a.protein_id = m.protein_id
                WHERE m.cluster_key = ? AND a.source = ? AND ({conditions})
                GROUP BY m.locus_key, m.genome_id, m.contig_id, m.protein_id,
                         m.locus_order
                ORDER BY m.locus_key, m.locus_order
                """,
                [cluster_key, source, *[f"%{t.lower()}%" for t in terms]],
            ).fetchall()
            lengths = dict(connection.execute(
                """
                SELECT locus_key, COUNT(*) FROM current_elsa_cluster_members
                WHERE cluster_key = ? GROUP BY locus_key
                """,
                [cluster_key],
            ).fetchall())
        finally:
            connection.execute("DETACH _ann")
    finally:
        connection.close()

    grouped: Dict[str, Dict[str, Any]] = {}
    for locus_key, genome, contig, protein, order, domains in rows:
        entry = grouped.setdefault(locus_key, {
            "locus_key": locus_key, "genome_id": genome, "contig_id": contig,
            "locus_length": int(lengths.get(locus_key, 0)), "matches": [],
        })
        entry["matches"].append({
            "protein_id": protein,
            "locus_order": int(order),
            "domains": sorted((domains or "").split(";")),
        })

    loci: List[Dict[str, Any]] = []
    for entry in grouped.values():
        orders = [m["locus_order"] for m in entry["matches"]]
        first, last = min(orders), max(orders)
        span = last - first + 1
        length = entry["locus_length"] or span
        entry["match_first"] = first
        entry["match_last"] = last
        entry["match_span"] = span
        # Fraction of the locus occupied by the matched stretch. Low values are
        # the honest signal that the locus is much larger than the query.
        entry["match_fraction"] = round(span / length, 4) if length else None
        entry["relative_position"] = round(((first + last) / 2) / max(length - 1, 1), 4)
        loci.append(entry)

    loci.sort(key=lambda e: (-len(e["matches"]), e["locus_key"]))
    fractions = [e["match_fraction"] for e in loci if e["match_fraction"] is not None]
    spans = [e["match_span"] for e in loci]
    lengths_seen = [e["locus_length"] for e in loci if e["locus_length"]]

    import statistics
    return {
        "cluster_key": cluster_key,
        "terms": list(terms),
        "n_loci_with_matches": len(loci),
        "loci": loci[:max_loci],
        "truncated": len(loci) > max_loci,
        "summary": {
            "median_match_span": statistics.median(spans) if spans else None,
            "median_locus_length": statistics.median(lengths_seen) if lengths_seen else None,
            "median_match_fraction": round(statistics.median(fractions), 4) if fractions else None,
        },
    }
