"""Regression tests for the normalized, run-scoped ELSA result contract."""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from elsa.results import (
    ELSAResultContractError,
    build_run_manifest,
    materialize_result_store,
)


def _genes() -> pd.DataFrame:
    rows = []
    for genome, contig, prefix, strand in (
        ("genome_a", "contig_a", "a", 1),
        ("genome_b", "contig_b", "b", -1),
        ("genome_c", "contig_c", "c", 1),
    ):
        for index in range(5):
            rows.append(
                {
                    "sample_id": genome,
                    "contig_id": contig,
                    "gene_id": f"{prefix}{index}",
                    "start": index * 100,
                    "end": index * 100 + 89,
                    "strand": strand,
                }
            )
    return pd.DataFrame(rows)


def _blocks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_id": 10,
                "cluster_id": 4,
                "query_genome": "genome_a",
                "target_genome": "genome_b",
                "query_contig": "contig_a",
                "target_contig": "contig_b",
                "query_start": 1,
                "query_end": 3,
                "target_start": 1,
                "target_end": 3,
                "n_anchors": 2,
                "chain_score": 1.9,
                "orientation": -1,
                "query_anchor_genes": '["a1", "a3"]',
                "target_anchor_genes": '["b1", "b3"]',
            },
            {
                "block_id": 11,
                "cluster_id": 4,
                "query_genome": "genome_a",
                "target_genome": "genome_c",
                "query_contig": "contig_a",
                "target_contig": "contig_c",
                "query_start": 1,
                "query_end": 3,
                "target_start": 1,
                "target_end": 3,
                "n_anchors": 2,
                "chain_score": 1.8,
                "orientation": 1,
                "query_anchor_genes": '["a1", "a3"]',
                "target_anchor_genes": '["c1", "c3"]',
            },
            {
                "block_id": 12,
                "cluster_id": 0,
                "query_genome": "genome_b",
                "target_genome": "genome_c",
                "query_contig": "contig_b",
                "target_contig": "contig_c",
                "query_start": 0,
                "query_end": 1,
                "target_start": 0,
                "target_end": 1,
                "n_anchors": 2,
                "chain_score": 1.7,
                "orientation": 1,
                "query_anchor_genes": '["b0", "b1"]',
                "target_anchor_genes": '["c0", "c1"]',
            },
        ]
    )


def _manifest(label: str = "synthetic") -> dict:
    return build_run_manifest(
        run_label=label,
        parameters={"similarity_threshold": 0.85},
        input_artifacts={},
        dataset_id="dataset-test",
        hash_inputs=False,
    )


def test_materialization_preserves_exact_pairs_roles_and_singletons(tmp_path):
    path = tmp_path / "synteny.duckdb"
    summary = materialize_result_store(
        path,
        blocks=_blocks(),
        genes=_genes(),
        manifest=_manifest(),
    )

    assert summary.block_count == 3
    assert summary.cluster_count == 2
    assert summary.singleton_count == 1
    assert summary.anchor_pair_count == 6

    connection = duckdb.connect(str(path), read_only=True)
    inverted = connection.execute(
        """
        SELECT pair_order, query_protein_id, target_protein_id
        FROM elsa_anchor_pairs
        WHERE block_id = 10
        ORDER BY pair_order
        """
    ).fetchall()
    assert inverted == [(0, "a1", "b3"), (1, "a3", "b1")]

    singleton = connection.execute(
        """
        SELECT cluster_key, source_cluster_id, cluster_kind, size, genome_support
        FROM elsa_clusters
        WHERE cluster_kind = 'singleton'
        """
    ).fetchone()
    assert singleton == ("singleton:12", None, "singleton", 1, 2)

    roles = dict(
        connection.execute(
            """
            SELECT protein_id, member_role
            FROM elsa_cluster_members
            WHERE cluster_key = 'cluster:4'
              AND genome_id = 'genome_a'
            ORDER BY protein_id
            """
        ).fetchall()
    )
    assert roles == {"a1": "anchor", "a2": "context", "a3": "anchor"}

    orphan_blocks = connection.execute(
        """
        SELECT COUNT(*)
        FROM elsa_blocks AS blocks
        LEFT JOIN elsa_clusters AS clusters
          ON clusters.run_id = blocks.run_id
         AND clusters.cluster_key = blocks.cluster_key
        WHERE clusters.cluster_key IS NULL
        """
    ).fetchone()[0]
    assert orphan_blocks == 0


def test_run_namespace_keeps_colliding_source_cluster_ids_distinct(tmp_path):
    path = tmp_path / "synteny.duckdb"
    first = _manifest("first")
    second = _manifest("second")
    second["parameters"] = {"similarity_threshold": 0.9}
    second["identity_sha256"] = "second-" + second["identity_sha256"]
    second["run_id"] = "elsa-second"

    materialize_result_store(
        path,
        blocks=_blocks(),
        genes=_genes(),
        manifest=first,
    )
    materialize_result_store(
        path,
        blocks=_blocks(),
        genes=_genes(),
        manifest=second,
    )

    connection = duckdb.connect(str(path), read_only=True)
    rows = connection.execute(
        """
        SELECT run_id, cluster_key
        FROM elsa_clusters
        WHERE source_cluster_id = 4
        ORDER BY run_id
        """
    ).fetchall()
    assert rows == [(first["run_id"], "cluster:4"), ("elsa-second", "cluster:4")]
    active = connection.execute("SELECT run_id FROM current_elsa_run").fetchone()[0]
    assert active == "elsa-second"


def test_source_validation_rejects_malformed_anchor_arrays(tmp_path):
    blocks = _blocks()
    blocks.loc[0, "target_anchor_genes"] = '["b1"]'

    with pytest.raises(ELSAResultContractError, match="unequal_arrays"):
        materialize_result_store(
            tmp_path / "invalid.duckdb",
            blocks=blocks,
            genes=_genes(),
            manifest=_manifest(),
        )


def test_source_validation_rejects_unsorted_gene_coordinates(tmp_path):
    genes = _genes()
    genes.iloc[[0, 1]] = genes.iloc[[1, 0]].to_numpy()

    with pytest.raises(ELSAResultContractError, match="coordinate_order_violations"):
        materialize_result_store(
            tmp_path / "invalid-order.duckdb",
            blocks=_blocks(),
            genes=genes,
            manifest=_manifest(),
        )


def test_source_validation_rejects_anchor_on_wrong_block_side(tmp_path):
    blocks = _blocks()
    blocks.loc[0, "query_anchor_genes"] = '["c1", "a3"]'

    with pytest.raises(ELSAResultContractError, match="block-side coordinates"):
        materialize_result_store(
            tmp_path / "invalid-anchor-side.duckdb",
            blocks=blocks,
            genes=_genes(),
            manifest=_manifest(),
        )


def test_manifest_identity_is_deterministic_for_identical_inputs():
    first = _manifest()
    second = _manifest()

    assert first["run_id"] == second["run_id"]
    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["inputs"]["elsa_source_tree"]["hash_status"] == "complete"
    assert len(first["inputs"]["elsa_source_tree"]["sha256"]) == 64
    assert json.loads(json.dumps(first["parameters"])) == {
        "similarity_threshold": 0.85
    }


@pytest.mark.parametrize("source_kind", ["dataframe", "parquet"])
def test_gene_map_preserves_exact_store_row_order_for_coordinate_ties(
    tmp_path,
    source_kind,
):
    genes = pd.DataFrame(
        [
            {
                "sample_id": genome,
                "contig_id": contig,
                "gene_id": gene_id,
                "start": 0,
                "end": 90,
                "strand": 1,
            }
            for genome, contig, gene_ids in (
                ("genome_a", "contig_a", ("a_z", "a_a", "a_m")),
                ("genome_b", "contig_b", ("b_z", "b_a", "b_m")),
            )
            for gene_id in gene_ids
        ]
    )
    gene_path = tmp_path / "metadata.parquet"
    genes.to_parquet(gene_path, index=False)
    gene_source = genes if source_kind == "dataframe" else gene_path
    blocks = pd.DataFrame(
        [
            {
                "block_id": 20,
                "cluster_id": 0,
                "query_genome": "genome_a",
                "target_genome": "genome_b",
                "query_contig": "contig_a",
                "target_contig": "contig_b",
                "query_start": 0,
                "query_end": 2,
                "target_start": 0,
                "target_end": 2,
                "n_anchors": 2,
                "chain_score": 1.8,
                "orientation": 1,
                "query_anchor_genes": '["a_z", "a_m"]',
                "target_anchor_genes": '["b_z", "b_m"]',
            }
        ]
    )

    result_path = tmp_path / "synteny.duckdb"
    materialize_result_store(
        result_path,
        blocks=blocks,
        genes=gene_source,
        manifest=_manifest("tied-coordinates"),
    )

    connection = duckdb.connect(str(result_path), read_only=True)
    positions = connection.execute(
        """
        SELECT protein_id, position_index
        FROM elsa_genes
        WHERE genome_id = 'genome_a'
        ORDER BY position_index
        """
    ).fetchall()
    assert positions == [("a_z", 0), ("a_a", 1), ("a_m", 2)]
