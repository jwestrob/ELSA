"""HTTP routes over a materialized run.

The same endpoints back the page and the agent API, so these tests assert the
contract both depend on: bounded responses, caveats that survive serialization,
and errors that are distinguishable from empty results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from elsa.results import materialize_result_store  # noqa: E402
from elsa.serve.api import Settings, create_app  # noqa: E402

from test_results import _blocks, _genes, _manifest  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    database = tmp_path / "synteny.duckdb"
    materialize_result_store(database, blocks=_blocks(), genes=_genes(),
                             manifest=_manifest())
    return TestClient(create_app(Settings(result_db=database)))


def test_run_endpoint_reports_identity_and_annotation_state(client):
    body = client.get("/api/run").json()
    assert body["status"] == "ready"
    assert body["run_id"].startswith("elsa-")
    assert body["annotations_attached"] is False


def test_clusters_are_bounded_and_validated(client):
    assert len(client.get("/api/clusters?limit=1").json()) <= 1
    assert client.get("/api/clusters?limit=9999").status_code == 422
    assert client.get("/api/clusters?limit=0").status_code == 422


def test_unknown_tier_is_rejected_rather_than_ignored(client):
    """Silently ignoring a bad filter would return unfiltered results that look
    filtered, which is worse than an error."""
    response = client.get("/api/clusters?tier=excellent")
    assert response.status_code == 422
    assert "tier must be one of" in response.text


def test_caveats_survive_the_json_boundary(client):
    """The qualifications are the point; an agent must receive them, not just
    a human reading a card."""
    rows = client.get("/api/clusters?limit=5").json()
    assert rows
    for row in rows:
        assert row["caveats"], "every record must carry its limits"
        assert any("lower bound" in c for c in row["caveats"])


def test_cluster_detail_returns_the_renderer_contract(client):
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    body = client.get(f"/api/cluster/{key}").json()
    assert body["cluster_key"] == key
    assert body["loci"]
    for locus in body["loci"]:
        assert {"name", "genome_id", "contig_id", "genes"} <= set(locus)


def test_diagram_route_is_reachable_and_returns_html(client):
    """Regression: a ``:path`` converter on cluster_key swallowed the /diagram
    suffix, so this route 404'd while looking correctly registered."""
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    response = client.get(f"/api/cluster/{key}/diagram?height=400")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "cluster-multiloc" in response.text
    assert "__DATA__" not in response.text


def test_missing_cluster_is_404_on_both_representations(client):
    """404 rather than an empty diagram: absent must not render as blank."""
    assert client.get("/api/cluster/cluster:absent").status_code == 404
    assert client.get("/api/cluster/cluster:absent/diagram").status_code == 404


def test_cluster_keys_containing_a_colon_route_correctly(client):
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    assert ":" in key, "fixture should exercise the colon in cluster keys"
    assert client.get(f"/api/cluster/{key}").status_code == 200


def test_openapi_schema_is_published_for_agents(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/clusters" in paths
    assert "/api/cluster/{cluster_key}" in paths
    assert "/api/cluster/{cluster_key}/diagram" in paths


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ELSA results" in response.text


def test_settings_from_env_requires_a_sidecar(monkeypatch):
    monkeypatch.delenv("ELSA_RESULT_DB", raising=False)
    with pytest.raises(RuntimeError, match="No result sidecar"):
        Settings.from_env()


def test_settings_from_env_rejects_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ELSA_RESULT_DB", str(tmp_path / "absent.duckdb"))
    with pytest.raises(RuntimeError, match="does not exist"):
        Settings.from_env()


def test_missing_annotation_db_is_ignored_not_fatal(monkeypatch, tmp_path):
    """A stale annotation path should degrade to unlabelled diagrams rather than
    refusing to serve the run at all."""
    database = tmp_path / "synteny.duckdb"
    materialize_result_store(database, blocks=_blocks(), genes=_genes(),
                             manifest=_manifest())
    monkeypatch.setenv("ELSA_RESULT_DB", str(database))
    monkeypatch.setenv("ELSA_ANNOTATION_DB", str(tmp_path / "gone.duckdb"))
    settings = Settings.from_env()
    assert settings.annotations_db is None


@pytest.fixture()
def annotated_client(tmp_path):
    """Server with an annotation database attached, for domain search."""
    import duckdb as _duck
    from elsa.report import cluster_payload

    database = tmp_path / "synteny.duckdb"
    materialize_result_store(database, blocks=_blocks(), genes=_genes(),
                             manifest=_manifest())
    payload = cluster_payload(database, "cluster:4")
    proteins = [g["id"] for L in payload["loci"] for g in L["genes"]]

    ann = tmp_path / "sharur.duckdb"
    connection = _duck.connect(str(ann))
    connection.execute(
        "CREATE TABLE annotations (protein_id VARCHAR, source VARCHAR, name VARCHAR)"
    )
    connection.executemany("INSERT INTO annotations VALUES (?, ?, ?)", [
        (proteins[0], "pfam", "Ribosomal_L2"),
        (proteins[1], "pfam", "ABC_tran"),
        (proteins[1], "pfam", "ATPase_AAA"),
    ])
    connection.close()
    return TestClient(create_app(Settings(result_db=database, annotations_db=ann)))


def test_overview_reports_tiers_and_truncation(client):
    body = client.get("/api/overview").json()
    assert body["n_clusters"] >= 1
    assert set(body["evidence_tiers"]) <= {"minimal", "moderate", "strong", "very_strong"}
    assert "locus_size_histogram" in body


def test_domain_search_finds_clusters_by_content(annotated_client):
    rows = annotated_client.get("/api/search/domain?q=ribosomal").json()
    assert rows
    assert any("Ribosomal_L2" in r["domains"] for r in rows)


def test_domain_search_is_case_insensitive_substring(annotated_client):
    lower = annotated_client.get("/api/search/domain?q=ribosomal").json()
    upper = annotated_client.get("/api/search/domain?q=RIBOSOMAL_L2").json()
    assert lower and upper


def test_match_all_requires_every_term(annotated_client):
    """Co-occurrence, not 'any of these'. One protein carries both terms here."""
    either = annotated_client.get(
        "/api/search/domain?q=Ribosomal_L2,ABC_tran").json()
    both = annotated_client.get(
        "/api/search/domain?q=Ribosomal_L2,ABC_tran&match_all=true").json()
    assert either
    for row in both:
        assert row["n_matched_terms"] >= 2


def test_domain_search_without_annotations_is_503_not_empty(client):
    """An empty result would read as 'no cluster contains this domain', which is
    a different and wrong claim from 'no annotations are attached'."""
    response = client.get("/api/search/domain?q=ribosomal")
    assert response.status_code == 503
    assert "annotation database" in response.text


def test_locus_search_matches_genome_substring(client):
    rows = client.get("/api/search/locus?genome=genome_a").json()
    assert rows
    assert all("cluster_key" in r for r in rows)


def test_locus_search_requires_a_term(client):
    assert client.get("/api/search/locus").status_code == 422


def test_blank_domain_query_is_rejected(annotated_client):
    assert annotated_client.get("/api/search/domain?q=%20").status_code == 422


def test_consensus_diagram_returns_gene_arrows(annotated_client):
    key = "cluster:4"
    response = annotated_client.get(
        f"/api/cluster/{key}/consensus/diagram?min_coverage=0.1")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<polygon" in response.text or "No slot reaches" in response.text


def test_consensus_diagram_says_so_when_nothing_meets_coverage(annotated_client):
    """An empty SVG would read as 'this cluster has no architecture'."""
    response = annotated_client.get(
        "/api/cluster/cluster:4/consensus/diagram?min_coverage=1.0")
    assert response.status_code == 200
    assert "<polygon" in response.text or "No slot reaches" in response.text


def test_diagram_accepts_a_cosine_threshold(client):
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    response = client.get(f"/api/cluster/{key}/diagram?tau=0.995")
    assert response.status_code == 200
    assert "INITIAL_TAU = 0.995" in response.text


def test_diagram_rejects_an_out_of_range_tau(client):
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    assert client.get(f"/api/cluster/{key}/diagram?tau=1.5").status_code == 422


# ---------------------------------------------------------------------------
# Locating a query inside a locus
# ---------------------------------------------------------------------------

def test_locate_reports_offsets_inside_the_locus(annotated_client):
    """Domain search names the cluster; this says where inside it to look."""
    body = annotated_client.get(
        "/api/cluster/cluster:4/locate?q=ribosomal").json()

    assert body["n_loci_with_matches"] >= 1
    entry = body["loci"][0]
    assert entry["match_first"] <= entry["match_last"]
    assert entry["match_span"] == entry["match_last"] - entry["match_first"] + 1
    assert entry["locus_length"] >= entry["match_span"]
    assert all("Ribosomal_L2" in m["domains"] for m in entry["matches"])


def test_match_fraction_is_boundary_precision_without_ground_truth(annotated_client):
    """The benchmark measures this against curated operons. Here the caller's own
    query supplies the reference, so it works on any dataset."""
    entry = annotated_client.get(
        "/api/cluster/cluster:4/locate?q=ribosomal").json()["loci"][0]

    expected = entry["match_span"] / entry["locus_length"]
    assert entry["match_fraction"] == pytest.approx(expected, abs=1e-4)
    assert 0.0 < entry["match_fraction"] <= 1.0


def test_locate_reports_absence_as_404(annotated_client):
    """'Nothing matched' must stay distinguishable from 'nothing attached'."""
    assert annotated_client.get(
        "/api/cluster/cluster:4/locate?q=nosuchdomain").status_code == 404


def test_locate_reports_missing_annotations_as_503(client):
    assert client.get("/api/cluster/cluster:4/locate?q=ribosomal").status_code == 503


def test_locate_validates_its_query(annotated_client):
    assert annotated_client.get("/api/cluster/cluster:4/locate?q=,,").status_code == 422
    assert annotated_client.get(
        "/api/cluster/cluster:4/locate?q=ribosomal&max_loci=99999").status_code == 422


# ---------------------------------------------------------------------------
# Reaching a cluster by key
# ---------------------------------------------------------------------------

def test_cluster_lookup_returns_the_full_record(client):
    """Ranking cannot surface one cluster out of hundreds of thousands on
    request, so a key in hand needs its own route."""
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    body = client.get(f"/api/search/cluster?q={key}").json()

    assert len(body) == 1
    assert body[0]["cluster_key"] == key
    # the full record, so the card can show its limits without a second fetch
    assert body[0]["caveats"]
    assert "evidence_tier" in body[0]


def test_cluster_lookup_accepts_a_bare_number(client):
    """Keys get copied as bare numbers; prefixing is a formatting detail."""
    key = client.get("/api/clusters?limit=1").json()[0]["cluster_key"]
    bare = key.split(":", 1)[1]
    assert client.get(f"/api/search/cluster?q={bare}").json()[0]["cluster_key"] == key


def test_cluster_lookup_takes_several_keys(client):
    keys = [r["cluster_key"] for r in client.get("/api/clusters?limit=2").json()]
    if len(keys) < 2:
        pytest.skip("fixture has one cluster")
    body = client.get(f"/api/search/cluster?q={','.join(keys)}").json()
    assert {r["cluster_key"] for r in body} == set(keys)


def test_cluster_lookup_distinguishes_missing_from_malformed(client):
    assert client.get("/api/search/cluster?q=cluster:99999999").status_code == 404
    assert client.get("/api/search/cluster?q=,,").status_code == 422
