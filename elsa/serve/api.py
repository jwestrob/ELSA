"""HTTP interface to a materialized ELSA run.

One set of routes serves both audiences. An agent calls them and receives JSON
with the schema published at ``/docs``; the browser page calls the same routes
and draws the result. Neither can drift from the other because there is one
producer underneath: :mod:`elsa.report` over the result sidecar.

Run it with::

    elsa serve --result-db data/DATASET/synteny.duckdb \\
               --annotations data/DATASET/sharur.duckdb

Configuration also reads ``ELSA_RESULT_DB`` and ``ELSA_ANNOTATION_DB``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ..report import EVIDENCE_TIERS, cluster_payload, cluster_records
from ..search_index import run_overview, search_by_domain, search_by_locus
from ..viz.clinker import (build_consensus_html, build_multiloc_html,
                           consensus_from_data)

TIER_NAMES = [name for name, _, _ in EVIDENCE_TIERS]


@dataclass
class Settings:
    """Where the run lives. Resolved once at startup so requests stay cheap."""

    result_db: Path
    annotations_db: Optional[Path] = None
    store_dir: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "Settings":
        result = os.environ.get("ELSA_RESULT_DB")
        if not result:
            raise RuntimeError(
                "No result sidecar configured. Pass --result-db or set ELSA_RESULT_DB."
            )
        path = Path(result)
        if not path.is_file():
            raise RuntimeError(f"Result sidecar does not exist: {path}")
        annotations = os.environ.get("ELSA_ANNOTATION_DB")
        annotations_path = Path(annotations) if annotations else None
        if annotations_path is not None and not annotations_path.is_file():
            annotations_path = None
        store = os.environ.get("ELSA_STORE_DIR")
        store_path = Path(store) if store else None
        if store_path is not None and not store_path.is_dir():
            store_path = None
        return cls(result_db=path, annotations_db=annotations_path,
                   store_dir=store_path)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="ELSA results",
        version="1",
        description=(
            "Syntenic clusters from a materialized ELSA run. Every cluster carries "
            "the qualifications needed to read its numbers: genome_support is a "
            "lower bound bounded by anchor retrieval, evidence_tier reports how "
            "many anchors the blocks rest on, and truncated_fraction says how much "
            "of the cluster may be cut short by contig edges rather than biology."
        ),
    )
    app.state.settings = settings

    @app.get("/api/run", summary="Active run identity and size")
    def run_info() -> Dict[str, Any]:
        connection = duckdb.connect(str(settings.result_db), read_only=True)
        try:
            row = connection.execute(
                """
                SELECT run_id, run_label, status, gene_count, block_count,
                       cluster_count, singleton_count, anchor_pair_count
                FROM current_elsa_run
                """
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise HTTPException(status_code=404, detail="sidecar has no active run")
        keys = ("run_id", "run_label", "status", "gene_count", "block_count",
                "cluster_count", "singleton_count", "anchor_pair_count")
        payload = dict(zip(keys, row))
        payload["annotations_attached"] = settings.annotations_db is not None
        return payload

    @app.get("/api/clusters", summary="Ranked clusters, bounded")
    def clusters(
        limit: int = Query(50, ge=1, le=500),
        min_genome_support: Optional[int] = Query(None, ge=0),
        tier: Optional[str] = Query(
            None, description=f"Minimum evidence tier: {', '.join(TIER_NAMES)}"
        ),
        include_singletons: bool = False,
    ) -> List[Dict[str, Any]]:
        """Entry point into a run.

        Bounded on purpose: a large run holds hundreds of thousands of cluster
        keys, and returning them whole serves neither a reader nor an agent.
        """
        if tier is not None and tier not in TIER_NAMES:
            raise HTTPException(
                status_code=422,
                detail=f"tier must be one of {TIER_NAMES}",
            )
        records = cluster_records(
            settings.result_db,
            limit=limit,
            min_genome_support=min_genome_support,
            min_evidence=tier,
            include_singletons=include_singletons,
        )
        return [record.as_dict() for record in records]

    @app.get("/api/overview", summary="Run-level totals and distributions")
    def overview() -> Dict[str, Any]:
        """Dashboard figures: cluster and singleton counts, genome count,
        evidence-tier breakdown, locus-size histogram, truncation rate."""
        return run_overview(settings.result_db)

    @app.get("/api/search/domain", summary="Clusters containing a protein domain")
    def domain_search(
        q: str = Query(..., description="Comma-separated domain terms, e.g. 'ribosomal' or 'ATPase,ABC_tran'"),
        limit: int = Query(50, ge=1, le=500),
        match_all: bool = Query(False, description="Require every term, not just one"),
        source: str = Query("pfam"),
    ) -> List[Dict[str, Any]]:
        """Find clusters by what they contain rather than by rank.

        Requires an annotation database; ELSA's result contract holds synteny
        only, so without one there is nothing to match against.
        """
        if settings.annotations_db is None:
            raise HTTPException(
                status_code=503,
                detail="No annotation database attached; start with --annotations to search domains",
            )
        terms = [term for term in q.split(",") if term.strip()]
        if not terms:
            raise HTTPException(status_code=422, detail="q must contain at least one term")
        return search_by_domain(
            settings.result_db, settings.annotations_db, terms,
            limit=limit, source=source, match_all=match_all,
        )

    @app.get("/api/search/locus", summary="Clusters present on a genome or contig")
    def locus_search(
        genome: Optional[str] = Query(None),
        contig: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
    ) -> List[Dict[str, Any]]:
        if not genome and not contig:
            raise HTTPException(status_code=422, detail="supply genome or contig")
        return search_by_locus(
            settings.result_db, genome=genome, contig=contig, limit=limit,
        )

    @app.get("/api/cluster/{cluster_key}", summary="One cluster with its loci")
    def cluster_detail(
        cluster_key: str,
        max_loci: int = Query(6, ge=1, le=40),
    ) -> Dict[str, Any]:
        payload = cluster_payload(
            settings.result_db, cluster_key, max_loci=max_loci,
            annotations_db=settings.annotations_db, store_dir=settings.store_dir,
        )
        if not payload:
            raise HTTPException(status_code=404, detail=f"no such cluster: {cluster_key}")
        return payload

    @app.get(
        "/api/cluster/{cluster_key}/diagram",
        summary="Gene-neighbourhood diagram as standalone HTML",
        response_class=HTMLResponse,
    )
    def cluster_diagram(
        cluster_key: str,
        max_loci: int = Query(6, ge=1, le=40),
        height: int = Query(520, ge=200, le=4000),
        tau: float = Query(0.85, ge=0.0, le=1.0,
                           description="Initial cosine threshold for drawn connections"),
    ) -> HTMLResponse:
        payload = cluster_payload(
            settings.result_db, cluster_key, max_loci=max_loci,
            annotations_db=settings.annotations_db, store_dir=settings.store_dir,
        )
        if not payload:
            raise HTTPException(status_code=404, detail=f"no such cluster: {cluster_key}")
        return HTMLResponse(build_multiloc_html(payload, height=height,
                                                initial_tau=tau))

    @app.get("/api/cluster/{cluster_key}/consensus",
             summary="Consensus gene order across the cluster's loci")
    def cluster_consensus(
        cluster_key: str,
        max_loci: int = Query(12, ge=2, le=60),
        min_coverage: float = Query(0.6, ge=0.0, le=1.0,
                                    description="Fraction of loci a slot must occupy"),
    ) -> Dict[str, Any]:
        """The cluster's shared architecture, labelled by domain.

        Genes are grouped across loci through the anchor graph, then each group
        is labelled by its most common domain and reported with the fraction of
        loci that carry it. Requires annotations for the labels to be meaningful;
        without them the tokens fall back to gene identifiers.
        """
        payload = cluster_payload(
            settings.result_db, cluster_key, max_loci=max_loci,
            annotations_db=settings.annotations_db, store_dir=settings.store_dir,
        )
        if not payload:
            raise HTTPException(status_code=404, detail=f"no such cluster: {cluster_key}")
        tokens, pairs = consensus_from_data(payload, min_coverage)
        return {
            "cluster_key": cluster_key,
            "n_loci_used": len(payload["loci"]),
            "min_coverage": min_coverage,
            "annotated_fraction": payload.get("annotated_fraction"),
            "tokens": tokens,
            "adjacencies": pairs,
        }

    @app.get("/api/cluster/{cluster_key}/consensus/diagram",
             summary="Consensus architecture as gene arrows",
             response_class=HTMLResponse)
    def consensus_diagram(
        cluster_key: str,
        max_loci: int = Query(12, ge=2, le=60),
        min_coverage: float = Query(0.5, ge=0.0, le=1.0),
        height: int = Query(190, ge=120, le=1200),
    ) -> HTMLResponse:
        payload = cluster_payload(
            settings.result_db, cluster_key, max_loci=max_loci,
            annotations_db=settings.annotations_db, store_dir=settings.store_dir,
        )
        if not payload:
            raise HTTPException(status_code=404, detail=f"no such cluster: {cluster_key}")
        tokens, pairs = consensus_from_data(payload, min_coverage)
        html = build_consensus_html(tokens, pairs, height=height)
        if not html:
            return HTMLResponse(
                '<div style="background:#1a1a2e;color:#9aa0a8;padding:18px;'
                'font-family:ui-sans-serif,sans-serif;font-size:13px">'
                f'No slot reaches {min_coverage:.0%} coverage across '
                f'{len(payload["loci"])} loci.</div>'
            )
        # build_consensus_html already emits its own zoom chrome and page style.
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "result_db": str(settings.result_db)})

    return app
