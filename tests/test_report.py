"""Cluster records and diagram payloads built from the result sidecar."""

from __future__ import annotations

import duckdb
import pytest

from elsa.report import (
    EVIDENCE_TIERS,
    ClusterRecord,
    cluster_payload,
    cluster_records,
)
from elsa.results import materialize_result_store

from test_results import _blocks, _genes, _manifest


@pytest.fixture()
def sidecar(tmp_path):
    path = tmp_path / "synteny.duckdb"
    materialize_result_store(path, blocks=_blocks(), genes=_genes(),
                             manifest=_manifest())
    return path


def test_evidence_tiers_partition_the_anchor_range():
    """Every anchor count lands in exactly one tier, with no gap at a boundary."""
    from elsa.report import _evidence_tier
    assert _evidence_tier(2.0) == "minimal"
    assert _evidence_tier(2.99) == "minimal"
    assert _evidence_tier(3.0) == "moderate"
    assert _evidence_tier(6.0) == "strong"
    assert _evidence_tier(20.0) == "very_strong"
    assert _evidence_tier(1e9) == "very_strong"
    assert _evidence_tier(None) == "unknown"


def test_records_carry_the_support_caveat_unconditionally(sidecar):
    """genome_support is always a lower bound, so the qualification is never optional."""
    records = cluster_records(sidecar, limit=10)
    assert records
    for record in records:
        assert any("lower bound" in c for c in record.caveats)


def test_minimal_evidence_is_flagged_on_the_record(sidecar):
    records = cluster_records(sidecar, limit=10)
    minimal = [r for r in records if r.evidence_tier == "minimal"]
    assert minimal, "fixture blocks carry 2 anchors and should read as minimal"
    for record in minimal:
        assert any("minimal" in c for c in record.caveats)


def test_results_are_bounded_by_limit(sidecar):
    assert len(cluster_records(sidecar, limit=1)) <= 1


def test_min_evidence_filters_in_sql_not_after_the_limit(sidecar):
    """Regression: filtering post-fetch returned nothing whenever the widest
    clusters were thinly anchored, which is the ordinary case."""
    strong = cluster_records(sidecar, limit=5, min_evidence="very_strong")
    assert strong == []          # fixture has no 20+ anchor clusters
    everything = cluster_records(sidecar, limit=5)
    assert everything            # ...but unfiltered still returns rows


def test_min_evidence_rejects_an_unknown_tier(sidecar):
    with pytest.raises(ValueError, match="min_evidence"):
        cluster_records(sidecar, limit=5, min_evidence="excellent")


def test_singletons_are_excluded_by_default(sidecar):
    default = {r.cluster_key for r in cluster_records(sidecar, limit=50)}
    widened = {r.cluster_key for r in cluster_records(sidecar, limit=50,
                                                      include_singletons=True)}
    assert any(k.startswith("singleton:") for k in widened)
    assert not any(k.startswith("singleton:") for k in default)


def test_older_sidecar_without_shape_columns_degrades_gracefully(tmp_path):
    """A sidecar written before the shape columns existed must still yield records,
    reporting the extent as unknown rather than inventing one.

    Built by hand rather than by stripping a current sidecar: this is exactly the
    shape ELSA emitted before the locus-shape and truncation columns were added,
    and live runs in that state are still in use.
    """
    path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("""
        CREATE TABLE elsa_runs (run_id VARCHAR, is_active BOOLEAN);
        CREATE TABLE elsa_dataset_state (active_run_id VARCHAR);
        CREATE TABLE elsa_clusters (
            run_id VARCHAR, cluster_key VARCHAR, cluster_kind VARCHAR,
            size BIGINT, locus_count BIGINT, genome_support BIGINT,
            mean_anchor_count DOUBLE, mean_chain_score DOUBLE
        );
        CREATE TABLE elsa_cluster_loci (
            run_id VARCHAR, cluster_key VARCHAR, locus_key VARCHAR,
            genome_id VARCHAR, contig_id VARCHAR, n_genes BIGINT,
            block_support BIGINT
        );
        INSERT INTO elsa_runs VALUES ('legacy-run', TRUE);
        INSERT INTO elsa_dataset_state VALUES ('legacy-run');
        INSERT INTO elsa_clusters VALUES
            ('legacy-run', 'cluster:1', 'cluster', 12, 4, 9, 7.5, 6.1),
            ('legacy-run', 'cluster:2', 'cluster',  5, 3, 4, 2.5, 2.2);
        CREATE VIEW current_elsa_clusters AS
            SELECT c.* FROM elsa_clusters c
            JOIN elsa_dataset_state s ON s.active_run_id = c.run_id;
        CREATE VIEW current_elsa_cluster_loci AS
            SELECT l.* FROM elsa_cluster_loci l
            JOIN elsa_dataset_state s ON s.active_run_id = l.run_id;
    """)
    connection.close()

    records = cluster_records(path, limit=5)
    assert len(records) == 2
    assert all(r.locus_size_median is None for r in records)
    assert all(r.truncated_locus_count is None for r in records)
    assert all("variable-length" in r.headline() for r in records)
    # The absence must be stated, not silently read as "nothing is truncated".
    assert all(any("truncation was not recorded" in c for c in r.caveats)
               for r in records)


def test_headline_reads_without_consulting_the_schema(sidecar):
    record = cluster_records(sidecar, limit=1)[0]
    text = record.headline()
    assert record.cluster_key in text
    assert ">=" in text, "support must be presented as a bound, not an exact count"


def test_payload_shape_matches_the_renderer_contract(sidecar):
    """Field names are the renderer's, not this module's: the diagram consumes
    this dict directly, so a rename here silently breaks the browser."""
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    payload = cluster_payload(sidecar, key, max_loci=4)

    assert payload["cluster_key"] == key
    assert payload["loci"], "a real cluster must expose loci"
    for locus in payload["loci"]:
        assert {"name", "genome_id", "contig_id", "start", "end", "genes"} <= set(locus)
        for gene in locus["genes"]:
            assert {"id", "start", "end", "strand", "pfam"} <= set(gene)
    for edge in payload["edges"]:
        assert {"a", "ai", "b", "bi", "cos"} <= set(edge)


def test_payload_labels_similarity_as_block_level(sidecar):
    """Per-anchor cosine is discarded during chaining, so the payload reports the
    block mean and names it, rather than passing it off as per-anchor."""
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    payload = cluster_payload(sidecar, key)
    assert payload["similarity_source"] == "block_mean_chain_score"


def test_payload_edge_indices_address_real_genes(sidecar):
    """Edges are track/gene indices; truncating to max_loci must not leave one
    pointing off the end of a track."""
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    payload = cluster_payload(sidecar, key, max_loci=1)
    loci = payload["loci"]
    for edge in payload["edges"]:
        assert 0 <= edge["a"] < len(loci) and 0 <= edge["b"] < len(loci)
        assert 0 <= edge["ai"] < len(loci[edge["a"]]["genes"])
        assert 0 <= edge["bi"] < len(loci[edge["b"]]["genes"])
        assert edge["a"] != edge["b"], "an edge must connect two different tracks"


def test_unknown_cluster_returns_empty_payload(sidecar):
    assert cluster_payload(sidecar, "cluster:does-not-exist") == {}


def test_record_is_json_serializable(sidecar):
    import json
    record = cluster_records(sidecar, limit=1)[0]
    restored = json.loads(json.dumps(record.as_dict()))
    assert restored["cluster_key"] == record.cluster_key
    assert isinstance(restored["caveats"], list)


def _sharur_like(path, rows):
    """Minimal stand-in for a Sharur database's annotation table."""
    connection = duckdb.connect(str(path))
    connection.execute("""
        CREATE TABLE annotations (
            protein_id VARCHAR, source VARCHAR, name VARCHAR
        )
    """)
    connection.executemany("INSERT INTO annotations VALUES (?, ?, ?)", rows)
    connection.close()
    return path


def test_annotations_are_absent_unless_a_database_is_supplied(sidecar):
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    payload = cluster_payload(sidecar, key)
    assert payload["annotation_source"] is None
    assert payload["annotated_fraction"] is None
    assert all(g["pfam"] == "" for L in payload["loci"] for g in L["genes"])


def test_annotations_join_on_exact_protein_ids(tmp_path, sidecar):
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    bare = cluster_payload(sidecar, key)
    proteins = [g["id"] for L in bare["loci"] for g in L["genes"]]
    assert proteins

    ann = _sharur_like(tmp_path / "sharur.duckdb", [
        (proteins[0], "pfam", "Ribosomal_L2"),
        (proteins[0], "pfam", "Ribosomal_L2_C"),
        (proteins[0], "kofam", "K02886"),      # other sources are ignored
    ])
    payload = cluster_payload(sidecar, key, annotations_db=ann)

    labels = {g["id"]: g["pfam"] for L in payload["loci"] for g in L["genes"]}
    assert set(labels[proteins[0]].split(";")) == {"Ribosomal_L2", "Ribosomal_L2_C"}
    assert "K02886" not in labels[proteins[0]]
    assert payload["annotation_source"] == str(ann)
    assert payload["annotated_fraction"] == pytest.approx(1 / len(proteins))


def test_identifier_mismatch_is_reported_not_silently_empty(tmp_path, sidecar):
    """The interoperability failure that matters: a Sharur database keyed on a
    different identifier space yields empty labels, which reads exactly like
    "these proteins have no domains" unless the join rate is surfaced."""
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    ann = _sharur_like(tmp_path / "wrong.duckdb", [
        ("some_other_namespace_1", "pfam", "Ribosomal_L2"),
    ])
    payload = cluster_payload(sidecar, key, annotations_db=ann)

    assert payload["annotated_fraction"] == 0.0
    assert payload["annotation_source"] == str(ann)
    assert all(g["pfam"] == "" for L in payload["loci"] for g in L["genes"])


def test_annotation_attach_is_released(tmp_path, sidecar):
    """Two payload calls in a row must not fail on a lingering ATTACH."""
    key = cluster_records(sidecar, limit=1)[0].cluster_key
    ann = _sharur_like(tmp_path / "sharur.duckdb", [("x", "pfam", "Y")])
    first = cluster_payload(sidecar, key, annotations_db=ann)
    second = cluster_payload(sidecar, key, annotations_db=ann)
    assert first["loci"] and second["loci"]
