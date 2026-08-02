"""Cluster records and diagram payloads read from the normalized result sidecar.

One producer serves both audiences. A human sees a card and a gene-neighbourhood
diagram; an agent sees the same fields as JSON. Because both come from this
module, the browser and the agent API cannot drift apart.

The records carry their own limits. Several quantities in an ELSA run mean less
than their names suggest — ``genome_support`` is bounded by anchor retrieval
rather than by biology, a locus touching a contig edge may be truncated, and a
cluster resting on two anchors is not the same kind of object as one resting on
two hundred. Those qualifications travel with the data as ``caveats`` so a
consumer has to actively discard them to overclaim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import pandas as pd


# Evidence tiers follow the observed distribution of anchors per block: on a
# 3,564-genome DPANN run, 62% of clusters sit at 2-3 anchors and only 2.6%
# reach 20 or more. A single "cluster" type hides that spread entirely.
EVIDENCE_TIERS: Sequence[tuple[str, float, float]] = (
    ("minimal", 0.0, 3.0),
    ("moderate", 3.0, 6.0),
    ("strong", 6.0, 20.0),
    ("very_strong", 20.0, float("inf")),
)

SUPPORT_CAVEAT = (
    "genome_support is a lower bound: anchor retrieval is capped by hnsw_k, so a "
    "gene family present in thousands of genomes may be anchored into far fewer"
)
MINIMAL_EVIDENCE_CAVEAT = (
    "evidence tier is minimal: blocks rest on 2-3 anchors, the weakest collinearity "
    "signal the pipeline emits"
)
TRUNCATION_UNKNOWN_CAVEAT = (
    "contig-edge truncation was not recorded for this run; re-materialize to "
    "distinguish complete loci from assembly-truncated ones"
)


def _evidence_tier(mean_anchor_count: Optional[float]) -> str:
    if mean_anchor_count is None:
        return "unknown"
    for name, low, high in EVIDENCE_TIERS:
        if low <= mean_anchor_count < high:
            return name
    return "very_strong"


@dataclass(frozen=True)
class ClusterRecord:
    """One cluster, with the qualifications needed to read its numbers."""

    run_id: str
    cluster_key: str
    kind: str
    n_blocks: int
    n_loci: int
    genome_support: int
    breadth_class: str
    mean_anchor_count: Optional[float]
    evidence_tier: str
    locus_size_median: Optional[float] = None
    locus_size_iqr: Optional[float] = None
    truncated_locus_count: Optional[int] = None
    truncated_fraction: Optional[float] = None
    mean_chain_score: Optional[float] = None
    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def headline(self) -> str:
        """One line a human can read without consulting the schema."""
        extent = (
            f"{self.locus_size_median:.0f}-gene"
            if self.locus_size_median is not None
            else "variable-length"
        )
        return (
            f"{self.cluster_key}: {extent} locus in >={self.genome_support} genomes "
            f"({self.n_blocks} blocks, {self.evidence_tier} evidence)"
        )


def _columns(connection: duckdb.DuckDBPyConnection, relation: str) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    }


def _connect(database: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(database), read_only=True)


def _breadth_thresholds(connection: duckdb.DuckDBPyConnection) -> tuple[float, float]:
    """Breadth cutoffs from this run's own support distribution.

    Absolute thresholds would be meaningless because support is censored by
    retrieval, so the classes are explicitly relative to the run.
    """
    row = connection.execute(
        """
        SELECT quantile_cont(genome_support, 0.5),
               quantile_cont(genome_support, 0.9)
        FROM current_elsa_clusters
        WHERE cluster_kind = 'cluster'
        """
    ).fetchone()
    if not row or row[0] is None:
        return (0.0, 0.0)
    return (float(row[0]), float(row[1]))


def _breadth_class(support: int, median: float, upper: float) -> str:
    if support >= upper:
        return "widespread"
    if support >= median:
        return "intermediate"
    return "narrow"


def cluster_records(
    database: str | Path,
    *,
    limit: int = 50,
    min_genome_support: Optional[int] = None,
    min_evidence: Optional[str] = None,
    include_singletons: bool = False,
    cluster_keys: Optional[Sequence[str]] = None,
) -> List[ClusterRecord]:
    """Return a bounded, ranked slice of clusters.

    The default limit is deliberate: a DPANN run holds 302,581 cluster keys, and
    handing that back whole serves neither a human nor an agent. Ranking is by
    breadth then evidence, so the first page is the part of the run that can
    carry an interpretation.

    ``cluster_keys`` looks up specific clusters instead, which is how anyone
    arriving with a key in hand — from a note, a paper, or another tool — reaches
    it. Ranking a run of 300,000 clusters cannot surface one on request, and
    without this the key has to be typed into a URL.
    """
    connection = _connect(database)
    try:
        available = _columns(connection, "current_elsa_clusters")
        loci_columns = _columns(connection, "current_elsa_cluster_loci")
        truncation_recorded = "truncated_start" in loci_columns

        optional = {
            name: (name if name in available else "NULL")
            for name in ("locus_size_median", "locus_size_iqr", "truncated_locus_count")
        }
        where = [] if include_singletons else ["cluster_kind = 'cluster'"]
        params: List[Any] = []
        if cluster_keys:
            keys = [str(k).strip() for k in cluster_keys if str(k).strip()]
            if not keys:
                return []
            # An explicit lookup overrides the singleton filter: a key the caller
            # already holds should resolve whatever kind it turns out to be.
            where = [f"cluster_key IN ({','.join('?' * len(keys))})"]
            params = list(keys)
        if min_genome_support is not None:
            where.append("genome_support >= ?")
            params.append(int(min_genome_support))
        # The evidence floor has to be applied in SQL. Filtering after a limited
        # fetch ordered by support silently returns nothing whenever the widest
        # clusters happen to be thinly anchored, which is the common case.
        if min_evidence is not None:
            floors = {name: low for name, low, _ in EVIDENCE_TIERS}
            if min_evidence not in floors:
                raise ValueError(
                    f"min_evidence must be one of {sorted(floors)}; got {min_evidence!r}"
                )
            where.append("mean_anchor_count >= ?")
            params.append(float(floors[min_evidence]))
        clause = ("WHERE " + " AND ".join(where)) if where else ""

        rows = connection.execute(
            f"""
            SELECT run_id, cluster_key, cluster_kind, size, locus_count,
                   genome_support, mean_anchor_count, mean_chain_score,
                   {optional['locus_size_median']}, {optional['locus_size_iqr']},
                   {optional['truncated_locus_count']}
            FROM current_elsa_clusters
            {clause}
            ORDER BY genome_support DESC, mean_anchor_count DESC, size DESC
            LIMIT ?
            """,
            [*params, int(limit)],
        ).fetchall()

        median, upper = _breadth_thresholds(connection)
    finally:
        connection.close()

    records: List[ClusterRecord] = []
    for row in rows:
        (run_id, key, kind, size, n_loci, support, mean_anchor,
         chain_score, size_median, size_iqr, truncated) = row
        tier = _evidence_tier(mean_anchor)
        fraction = (
            float(truncated) / n_loci
            if truncated is not None and n_loci else None
        )
        caveats = [SUPPORT_CAVEAT]
        if tier == "minimal":
            caveats.append(MINIMAL_EVIDENCE_CAVEAT)
        if not truncation_recorded:
            caveats.append(TRUNCATION_UNKNOWN_CAVEAT)
        elif fraction:
            caveats.append(
                f"{truncated} of {n_loci} loci touch a contig edge and may be "
                "truncated by the assembly rather than bounded by biology"
            )

        records.append(ClusterRecord(
            run_id=run_id, cluster_key=key, kind=kind,
            n_blocks=int(size), n_loci=int(n_loci or 0),
            genome_support=int(support),
            breadth_class=_breadth_class(int(support), median, upper),
            mean_anchor_count=float(mean_anchor) if mean_anchor is not None else None,
            evidence_tier=tier,
            locus_size_median=float(size_median) if size_median is not None else None,
            locus_size_iqr=float(size_iqr) if size_iqr is not None else None,
            truncated_locus_count=int(truncated) if truncated is not None else None,
            truncated_fraction=fraction,
            mean_chain_score=float(chain_score) if chain_score is not None else None,
            caveats=caveats,
        ))
    return records


def _annotation_map(
    connection: duckdb.DuckDBPyConnection,
    annotations_db: str | Path,
    proteins: Sequence[str],
    source: str = "pfam",
) -> Dict[str, str]:
    """Domain names per protein, pulled from a Sharur database.

    ELSA's result contract deliberately holds no annotation layer, so domain
    labels come from the companion Sharur database keyed on exact protein IDs.
    Failure here is silent by nature — a mismatched identifier space yields an
    empty result that looks identical to "this protein has no domains" — so the
    caller is given the join rate rather than just the mapping.
    """
    if not proteins:
        return {}
    connection.execute(
        f"ATTACH '{str(annotations_db)}' AS _ann (READ_ONLY)"
    )
    try:
        placeholders = ", ".join("?" for _ in proteins)
        rows = connection.execute(
            f"""
            SELECT protein_id, string_agg(DISTINCT name, ';')
            FROM _ann.annotations
            WHERE source = ? AND protein_id IN ({placeholders})
            GROUP BY protein_id
            """,
            [source, *proteins],
        ).fetchall()
    finally:
        connection.execute("DETACH _ann")
    return {row[0]: row[1] or "" for row in rows}


@lru_cache(maxsize=4)
def _corpus_mean(store_dir: str, sample: int = 200_000) -> Optional[Any]:
    """Mean embedding of the store, cached per store directory.

    Needed to centre similarities. Sampled rather than exhaustive because the
    mean of 200k vectors is indistinguishable from the mean of three million for
    this purpose, and the store can be several gigabytes.
    """
    import numpy as np

    emb_path = Path(store_dir) / "embeddings.npy"
    if not emb_path.is_file():
        return None
    embeddings = np.load(emb_path, mmap_mode="r")
    n = embeddings.shape[0]
    if n <= sample:
        rows = slice(None)
    else:
        rng = np.random.default_rng(17)          # fixed: the centre must not drift
        rows = np.sort(rng.choice(n, sample, replace=False))
    return np.asarray(embeddings[rows], dtype=np.float64).mean(axis=0)


def _anchor_cosines(
    store_dir: str | Path,
    pairs: Sequence[tuple[str, str]],
) -> Dict[tuple[str, str], float]:
    """Centred cosine for each anchor pair, recomputed from the store.

    Chaining keeps only an aggregate ``chain_score``, so per-anchor similarity is
    absent from a materialized run. Both proteins still have a vector, so one dot
    product restores it.

    The vectors are centred on the corpus mean first. Mean-pooled protein
    language model embeddings are strongly anisotropic: measured on this corpus,
    two *unrelated* proteins sit at cosine 0.94 and anchored pairs at 0.996, so
    the raw scale compresses every relationship into its last few percent and a
    threshold over it cannot separate anything. Centring moves unrelated pairs to
    about 0.2 while anchors stay near 0.95, which is what makes the diagram's
    similarity control able to actually remove connections.
    """
    if not pairs:
        return {}
    store = Path(store_dir)
    meta_path = store / "metadata.parquet"
    emb_path = store / "embeddings.npy"
    if not (meta_path.is_file() and emb_path.is_file()):
        return {}

    import numpy as np

    wanted = {protein for pair in pairs for protein in pair}
    meta = pd.read_parquet(meta_path, columns=["gene_id"])
    row_of = {
        gene: index
        for index, gene in enumerate(meta.gene_id.values)
        if gene in wanted
    }
    if not row_of:
        return {}

    embeddings = np.load(emb_path, mmap_mode="r")
    rows = sorted(set(row_of.values()))
    block = np.asarray(embeddings[rows], dtype=np.float64)
    centre = _corpus_mean(str(store))
    if centre is not None and centre.shape[0] == block.shape[1]:
        block = block - centre
    block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
    local = {row: position for position, row in enumerate(rows)}

    result: Dict[tuple[str, str], float] = {}
    for query, target in pairs:
        qi, ti = row_of.get(query), row_of.get(target)
        if qi is None or ti is None:
            continue
        result[(query, target)] = float(block[local[qi]] @ block[local[ti]])
    return result


def cluster_payload(
    database: str | Path,
    cluster_key: str,
    *,
    max_loci: int = 6,
    annotations_db: Optional[str | Path] = None,
    store_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build the locus/edge payload the clinker renderers consume.

    The same structure is what an agent should receive when it asks what a
    cluster contains, so the diagram and the machine-readable answer are one
    object rendered two ways.

    Edges use track and gene indices rather than identifiers because that is the
    renderer's contract: ``a``/``b`` are locus positions in ``loci`` and
    ``ai``/``bi`` are gene positions within those loci.

    ``cos`` carries the *block-level* mean similarity, and ``similarity_source``
    says so. Per-anchor cosine is not recoverable from a materialized run:
    ``ChainedBlock`` keeps only an aggregate ``chain_score``, so the value behind
    each individual anchor is discarded during chaining rather than at
    materialization. Reporting the block mean is honest; inventing a per-anchor
    number would not be.
    """
    connection = _connect(database)
    try:
        loci = connection.execute(
            """
            SELECT locus_key, genome_id, contig_id, start_bp, end_bp,
                   n_genes, block_support
            FROM current_elsa_cluster_loci
            WHERE cluster_key = ?
            ORDER BY block_support DESC, n_genes DESC
            LIMIT ?
            """,
            [cluster_key, int(max_loci)],
        ).fetchall()
        if not loci:
            return {}

        keys = [row[0] for row in loci]
        placeholders = ", ".join("?" for _ in keys)
        members = connection.execute(
            f"""
            SELECT locus_key, protein_id, start_bp, end_bp, strand,
                   member_role, locus_order
            FROM current_elsa_cluster_members
            WHERE locus_key IN ({placeholders})
            ORDER BY locus_key, locus_order
            """,
            keys,
        ).fetchall()

        pairs = connection.execute(
            """
            SELECT pairs.query_protein_id, pairs.target_protein_id,
                   blocks.chain_score / NULLIF(blocks.n_anchors, 0)
            FROM elsa_anchor_pairs AS pairs
            JOIN elsa_blocks AS blocks
              ON blocks.run_id = pairs.run_id AND blocks.block_id = pairs.block_id
            WHERE pairs.cluster_key = ?
              AND pairs.run_id = (SELECT run_id FROM elsa_runs WHERE is_active)
            """,
            [cluster_key],
        ).fetchall()
    finally:
        connection.close()

    genes_by_locus: Dict[str, List[Dict[str, Any]]] = {key: [] for key in keys}
    for locus_key, protein, start_bp, end_bp, strand, role, _order in members:
        genes_by_locus[locus_key].append({
            "id": protein,
            "start": int(start_bp) if start_bp is not None else 0,
            "end": int(end_bp) if end_bp is not None else 0,
            "strand": 1 if (strand is None or int(strand) >= 0) else -1,
            "pfam": "",
            "role": role,
        })

    tracks: List[Dict[str, Any]] = []
    where: Dict[str, tuple[int, int]] = {}
    for track, (key, genome, contig, start_bp, end_bp, n_genes, support) in enumerate(loci):
        genes = genes_by_locus[key]
        for index, gene in enumerate(genes):
            where[gene["id"]] = (track, index)
        span = ""
        if start_bp is not None and end_bp is not None:
            span = f":{int(start_bp):,}-{int(end_bp):,}"
        tracks.append({
            "name": f"{genome}  |  {contig}{span} ({len(genes)} genes)",
            "locus_key": key,
            "genome_id": genome,
            "contig_id": contig,
            "start": int(start_bp) if start_bp is not None else 0,
            "end": int(end_bp) if end_bp is not None else 0,
            "block_support": int(support),
            "genes": genes,
        })

    exact = _anchor_cosines(store_dir, [(q, t) for q, t, _ in pairs]) if store_dir else {}

    edges: List[Dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for query, target, mean_score in pairs:
        left, right = where.get(query), where.get(target)
        if left is None or right is None or left[0] == right[0]:
            continue
        if left[0] > right[0]:
            left, right = right, left
        signature = (left[0], left[1], right[0], right[1])
        if signature in seen:
            continue
        seen.add(signature)
        cosine = exact.get((query, target))
        if cosine is None:
            cosine = float(mean_score) if mean_score is not None else None
        edges.append({
            "a": left[0], "ai": left[1],
            "b": right[0], "bi": right[1],
            "type": "anchor",
            "cos": cosine,
        })

    payload: Dict[str, Any] = {
        "cluster_key": cluster_key,
        "similarity_source": ("anchor_cosine_centered" if exact
                              else "block_mean_chain_score"),
        "loci": tracks,
        "edges": edges,
        "annotation_source": None,
        "annotated_fraction": None,
    }
    if annotations_db is not None:
        proteins = [gene["id"] for track in tracks for gene in track["genes"]]
        connection = _connect(database)
        try:
            mapping = _annotation_map(connection, annotations_db, proteins)
        finally:
            connection.close()
        for track in tracks:
            for gene in track["genes"]:
                gene["pfam"] = mapping.get(gene["id"], "")
        payload["annotation_source"] = str(annotations_db)
        payload["annotated_fraction"] = (
            len(mapping) / len(proteins) if proteins else 0.0
        )
    return payload


__all__ = [
    "ClusterRecord",
    "EVIDENCE_TIERS",
    "cluster_payload",
    "cluster_records",
]
