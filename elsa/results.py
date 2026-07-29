"""Run-scoped, relational ELSA result storage.

The legacy CSV output remains useful for interchange and the genome browser,
but it is a poor query contract: cluster context is serialized into JSON,
cluster identifiers are local to an undocumented run, and exact reverse
membership requires scanning strings.  This module materializes the same
result into a schema-versioned DuckDB sidecar with exact protein identifiers,
explicit anchor pairs, cluster loci, and anchor/context membership roles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import duckdb
import pandas as pd

from . import __version__


RESULT_SCHEMA_VERSION = 1
MAPPING_VERSION = "coordinate-sorted-position-index-v1"


class ELSAResultContractError(RuntimeError):
    """Raised when an input or materialized result violates the result contract."""


@dataclass(frozen=True)
class ResultStoreSummary:
    """Counts and identity for one materialized ELSA run."""

    path: Path
    run_id: str
    block_count: int
    cluster_count: int
    singleton_count: int
    anchor_pair_count: int
    locus_count: int
    member_count: int
    gene_count: int


def _result_log(message: str) -> None:
    print(f"[ELSA results] {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the complete SHA-256 of a local file using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path, *, hash_content: bool = True) -> dict[str, Any]:
    """Return a provenance record for one file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved) if hash_content else None,
        "hash_status": "complete" if hash_content else "unverified",
    }


def fingerprint_source_tree() -> dict[str, Any]:
    """Fingerprint the executable ELSA Python source in the working tree."""

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "elsa").rglob("*.py"))
    project_file = root / "pyproject.toml"
    if project_file.is_file():
        paths.append(project_file)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "path": str(root),
        "file_count": len(paths),
        "sha256": digest.hexdigest(),
        "hash_status": "complete",
    }


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def read_dataset_identity(path: str | Path | None) -> tuple[str | None, dict[str, Any] | None]:
    """Read a Sharur dataset seal and return its dataset ID plus provenance."""

    if path is None:
        return None, None
    seal_path = Path(path).expanduser().resolve()
    payload = json.loads(seal_path.read_text())
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ELSAResultContractError(
            f"Dataset seal has no non-empty dataset_id: {seal_path}"
        )
    return dataset_id, fingerprint_file(seal_path)


def build_run_manifest(
    *,
    run_label: str,
    parameters: Mapping[str, Any] | Any,
    input_artifacts: Mapping[str, str | Path | Mapping[str, Any]],
    dataset_id: str | None = None,
    dataset_seal: Mapping[str, Any] | None = None,
    elsa_commit: str | None = None,
    mapping_version: str = MAPPING_VERSION,
    hash_inputs: bool = True,
) -> dict[str, Any]:
    """Build a provenance-complete manifest and deterministic run ID."""

    artifacts: dict[str, Any] = {}
    for name, artifact in sorted(input_artifacts.items()):
        if isinstance(artifact, Mapping):
            artifacts[name] = _jsonable(artifact)
        else:
            artifacts[name] = fingerprint_file(artifact, hash_content=hash_inputs)
    artifacts["elsa_source_tree"] = fingerprint_source_tree()

    identity = {
        "dataset_id": dataset_id,
        "elsa_version": __version__,
        "elsa_commit": elsa_commit or _git_commit(),
        "mapping_version": mapping_version,
        "parameters": _jsonable(parameters),
        "inputs": artifacts,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    return {
        "run_id": f"elsa-{digest[:20]}",
        "run_label": run_label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
        "dataset_seal": _jsonable(dataset_seal),
        "identity_sha256": digest,
    }


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _source_sql(source: str | Path, *, kind: str) -> str:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return (
            f"read_parquet({_sql_literal(path)}, "
            "file_row_number=true)"
        )
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\\t" if path.suffix.lower() == ".tsv" else ","
        return (
            f"read_csv_auto({_sql_literal(path)}, header=true, "
            f"delim={_sql_literal(delimiter)}, sample_size=-1, "
            "max_line_size=16777216)"
        )
    raise ELSAResultContractError(
        f"Unsupported {kind} source format {path.suffix!r}: {path}"
    )


def _columns(connection: duckdb.DuckDBPyConnection, relation: str) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    }


def _initialize_schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS elsa_schema_version (
            version INTEGER PRIMARY KEY,
            installed_at TIMESTAMPTZ NOT NULL
        );

        CREATE TABLE IF NOT EXISTS elsa_runs (
            run_id VARCHAR PRIMARY KEY,
            run_label VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            status VARCHAR NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT FALSE,
            dataset_id VARCHAR,
            dataset_seal_json VARCHAR,
            elsa_version VARCHAR NOT NULL,
            elsa_commit VARCHAR,
            mapping_version VARCHAR NOT NULL,
            parameters_json VARCHAR NOT NULL,
            inputs_json VARCHAR NOT NULL,
            identity_sha256 VARCHAR NOT NULL,
            validation_json VARCHAR,
            gene_count BIGINT,
            block_count BIGINT,
            cluster_count BIGINT,
            singleton_count BIGINT,
            anchor_pair_count BIGINT,
            locus_count BIGINT,
            member_count BIGINT
        );

        CREATE TABLE IF NOT EXISTS elsa_dataset_state (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE,
            active_run_id VARCHAR
        );
        INSERT OR IGNORE INTO elsa_dataset_state(singleton, active_run_id)
        VALUES (TRUE, NULL);

        CREATE TABLE IF NOT EXISTS elsa_genes (
            run_id VARCHAR NOT NULL,
            protein_id VARCHAR NOT NULL,
            genome_id VARCHAR NOT NULL,
            contig_id VARCHAR NOT NULL,
            position_index BIGINT NOT NULL,
            start_bp BIGINT,
            end_bp BIGINT,
            strand BIGINT
        );

        CREATE TABLE IF NOT EXISTS elsa_blocks (
            run_id VARCHAR NOT NULL,
            block_id BIGINT NOT NULL,
            cluster_key VARCHAR NOT NULL,
            source_cluster_id BIGINT,
            query_genome VARCHAR NOT NULL,
            target_genome VARCHAR NOT NULL,
            query_contig VARCHAR NOT NULL,
            target_contig VARCHAR NOT NULL,
            query_start BIGINT NOT NULL,
            query_end BIGINT NOT NULL,
            target_start BIGINT NOT NULL,
            target_end BIGINT NOT NULL,
            query_start_bp BIGINT,
            query_end_bp BIGINT,
            target_start_bp BIGINT,
            target_end_bp BIGINT,
            n_anchors BIGINT NOT NULL,
            chain_score DOUBLE,
            orientation SMALLINT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS elsa_anchor_pairs (
            run_id VARCHAR NOT NULL,
            block_id BIGINT NOT NULL,
            cluster_key VARCHAR NOT NULL,
            pair_order BIGINT NOT NULL,
            query_protein_id VARCHAR NOT NULL,
            target_protein_id VARCHAR NOT NULL,
            query_position_index BIGINT NOT NULL,
            target_position_index BIGINT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS elsa_clusters (
            run_id VARCHAR NOT NULL,
            cluster_key VARCHAR NOT NULL,
            source_cluster_id BIGINT,
            cluster_kind VARCHAR NOT NULL,
            size BIGINT NOT NULL,
            genome_support BIGINT NOT NULL,
            mean_anchor_count DOUBLE NOT NULL,
            mean_chain_score DOUBLE,
            locus_count BIGINT NOT NULL DEFAULT 0,
            member_count BIGINT NOT NULL DEFAULT 0,
            anchor_member_count BIGINT NOT NULL DEFAULT 0,
            source_metadata_present BOOLEAN NOT NULL DEFAULT FALSE,
            reported_size BIGINT,
            reported_genome_support BIGINT,
            reported_mean_chain_length DOUBLE,
            truncated_locus_count BIGINT,
            locus_size_median DOUBLE,
            locus_size_iqr DOUBLE
        );

        CREATE TABLE IF NOT EXISTS elsa_cluster_loci (
            run_id VARCHAR NOT NULL,
            cluster_key VARCHAR NOT NULL,
            locus_key VARCHAR NOT NULL,
            locus_index BIGINT NOT NULL,
            genome_id VARCHAR NOT NULL,
            contig_id VARCHAR NOT NULL,
            start_position_index BIGINT NOT NULL,
            end_position_index BIGINT NOT NULL,
            start_bp BIGINT,
            end_bp BIGINT,
            start_protein_id VARCHAR,
            end_protein_id VARCHAR,
            n_genes BIGINT NOT NULL,
            block_support BIGINT NOT NULL,
            truncated_start BOOLEAN,
            truncated_end BOOLEAN
        );

        CREATE TABLE IF NOT EXISTS elsa_cluster_members (
            run_id VARCHAR NOT NULL,
            cluster_key VARCHAR NOT NULL,
            locus_key VARCHAR NOT NULL,
            protein_id VARCHAR NOT NULL,
            genome_id VARCHAR NOT NULL,
            contig_id VARCHAR NOT NULL,
            position_index BIGINT NOT NULL,
            locus_order BIGINT NOT NULL,
            member_role VARCHAR NOT NULL,
            start_bp BIGINT,
            end_bp BIGINT,
            strand BIGINT
        );
        """
    )
    # Additive columns for sidecars created before they existed. These carry
    # defaults, so a sidecar written by an older ELSA stays readable and simply
    # reports zero truncation until its run is re-materialized.
    for table, column, spec in (
        ("elsa_cluster_loci", "truncated_start", "BOOLEAN"),
        ("elsa_cluster_loci", "truncated_end", "BOOLEAN"),
        ("elsa_clusters", "truncated_locus_count", "BIGINT"),
        ("elsa_clusters", "locus_size_median", "DOUBLE"),
        ("elsa_clusters", "locus_size_iqr", "DOUBLE"),
    ):
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {spec}"
        )

    connection.execute(
        """
        INSERT OR IGNORE INTO elsa_schema_version
        VALUES (?, current_timestamp)
        """,
        [RESULT_SCHEMA_VERSION],
    )


_RUN_TABLES = (
    "elsa_cluster_members",
    "elsa_cluster_loci",
    "elsa_clusters",
    "elsa_anchor_pairs",
    "elsa_blocks",
    "elsa_genes",
)


def _delete_run(connection: duckdb.DuckDBPyConnection, run_id: str) -> None:
    for table in _RUN_TABLES:
        connection.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
    connection.execute("DELETE FROM elsa_runs WHERE run_id = ?", [run_id])


def _register_source(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    source: str | Path | pd.DataFrame,
    *,
    kind: str,
) -> None:
    if isinstance(source, pd.DataFrame):
        frame = source.reset_index(drop=True)
        if name == "source_genes_raw" and "file_row_number" not in frame:
            frame = frame.copy()
            frame["file_row_number"] = pd.RangeIndex(len(frame))
        registration = f"_{name}_dataframe"
        connection.register(registration, frame)
        connection.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {registration}")
        return
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {name} AS "
        f"SELECT * FROM {_source_sql(source, kind=kind)}"
    )


def _prepare_sources(
    connection: duckdb.DuckDBPyConnection,
    *,
    blocks: str | Path | pd.DataFrame,
    genes: str | Path | pd.DataFrame,
    clusters: str | Path | pd.DataFrame | None,
) -> tuple[str, str]:
    _register_source(connection, "source_blocks_raw", blocks, kind="block")
    _register_source(connection, "source_genes_raw", genes, kind="gene")
    if clusters is not None:
        _register_source(connection, "source_clusters_raw", clusters, kind="cluster")
    else:
        connection.execute(
            """
            CREATE OR REPLACE TEMP VIEW source_clusters_raw AS
            SELECT
                NULL::BIGINT AS cluster_id,
                NULL::BIGINT AS size,
                NULL::BIGINT AS genome_support,
                NULL::DOUBLE AS mean_chain_length
            WHERE FALSE
            """
        )

    block_columns = _columns(connection, "source_blocks_raw")
    gene_columns = _columns(connection, "source_genes_raw")
    required_blocks = {
        "block_id",
        "cluster_id",
        "query_genome",
        "target_genome",
        "query_contig",
        "target_contig",
        "query_start",
        "query_end",
        "target_start",
        "target_end",
        "n_anchors",
        "chain_score",
        "orientation",
    }
    required_genes = {"sample_id", "contig_id", "gene_id", "start", "end"}
    if missing := sorted(required_blocks - block_columns):
        raise ELSAResultContractError(f"Block source is missing columns: {missing}")
    if missing := sorted(required_genes - gene_columns):
        raise ELSAResultContractError(f"Gene source is missing columns: {missing}")

    if {"query_anchor_genes", "target_anchor_genes"} <= block_columns:
        query_anchor = "query_anchor_genes"
        target_anchor = "target_anchor_genes"
    elif {"anchor_query_gene_ids", "anchor_target_gene_ids"} <= block_columns:
        query_anchor = "anchor_query_gene_ids"
        target_anchor = "anchor_target_gene_ids"
    else:
        raise ELSAResultContractError(
            "Block source needs query/target anchor protein-ID arrays"
        )

    optional = {
        name: name if name in block_columns else f"NULL::{sql_type} AS {name}"
        for name, sql_type in {
            "query_start_bp": "BIGINT",
            "query_end_bp": "BIGINT",
            "target_start_bp": "BIGINT",
            "target_end_bp": "BIGINT",
        }.items()
    }
    orientation = """
        CASE
            WHEN lower(trim(CAST(orientation AS VARCHAR)))
                 IN ('1', '+1', 'same', 'forward', 'fwd') THEN 1
            WHEN lower(trim(CAST(orientation AS VARCHAR)))
                 IN ('-1', 'inverted', 'inverse', 'reverse', 'rev') THEN -1
            ELSE NULL
        END
    """
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source_blocks AS
        SELECT
            CAST(block_id AS BIGINT) AS block_id,
            CASE
                WHEN TRY_CAST(cluster_id AS BIGINT) > 0
                    THEN 'cluster:' || CAST(TRY_CAST(cluster_id AS BIGINT) AS VARCHAR)
                ELSE 'singleton:' || CAST(block_id AS VARCHAR)
            END AS cluster_key,
            CASE
                WHEN TRY_CAST(cluster_id AS BIGINT) > 0
                    THEN TRY_CAST(cluster_id AS BIGINT)
                ELSE NULL
            END AS source_cluster_id,
            CAST(query_genome AS VARCHAR) AS query_genome,
            CAST(target_genome AS VARCHAR) AS target_genome,
            CAST(query_contig AS VARCHAR) AS query_contig,
            CAST(target_contig AS VARCHAR) AS target_contig,
            CAST(query_start AS BIGINT) AS query_start,
            CAST(query_end AS BIGINT) AS query_end,
            CAST(target_start AS BIGINT) AS target_start,
            CAST(target_end AS BIGINT) AS target_end,
            CAST(n_anchors AS BIGINT) AS n_anchors,
            CAST(chain_score AS DOUBLE) AS chain_score,
            CAST(({orientation}) AS SMALLINT) AS orientation,
            CAST({query_anchor} AS VARCHAR) AS query_anchor_genes,
            CAST({target_anchor} AS VARCHAR) AS target_anchor_genes,
            {optional['query_start_bp']},
            {optional['query_end_bp']},
            {optional['target_start_bp']},
            {optional['target_end_bp']}
        FROM source_blocks_raw
        """
    )
    strand_expression = (
        "TRY_CAST(strand AS BIGINT)" if "strand" in gene_columns else "NULL::BIGINT"
    )
    source_order_expression = (
        "file_row_number"
        if "file_row_number" in gene_columns
        else "row_number() OVER () - 1"
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE source_gene_positions AS
        WITH ordered AS (
            SELECT *, {source_order_expression} AS source_order
            FROM source_genes_raw
        )
        SELECT
            CAST(gene_id AS VARCHAR) AS protein_id,
            CAST(sample_id AS VARCHAR) AS genome_id,
            CAST(contig_id AS VARCHAR) AS contig_id,
            row_number() OVER (
                PARTITION BY sample_id, contig_id
                ORDER BY source_order
            ) - 1 AS position_index,
            TRY_CAST(start AS BIGINT) AS start_bp,
            TRY_CAST("end" AS BIGINT) AS end_bp,
            {strand_expression} AS strand
        FROM ordered
        """
    )
    return query_anchor, target_anchor


def _validate_sources(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts = connection.execute(
        """
        SELECT
            COUNT(*) AS blocks,
            COUNT(DISTINCT block_id) AS distinct_blocks,
            COUNT(*) FILTER (WHERE orientation IS NULL) AS invalid_orientations,
            COUNT(*) FILTER (
                WHERE json_array_length(query_anchor_genes)
                   != json_array_length(target_anchor_genes)
            ) AS unequal_arrays,
            COUNT(*) FILTER (
                WHERE n_anchors != json_array_length(query_anchor_genes)
            ) AS anchor_count_mismatches,
            COUNT(*) FILTER (
                WHERE query_start < 0 OR target_start < 0
                   OR query_end < query_start
                   OR target_end < target_start
            ) AS invalid_intervals,
            COUNT(*) FILTER (WHERE n_anchors <= 0) AS nonpositive_anchor_counts
        FROM source_blocks
        """
    ).fetchone()
    result = {
        "blocks": int(counts[0]),
        "distinct_blocks": int(counts[1]),
        "invalid_orientations": int(counts[2]),
        "unequal_arrays": int(counts[3]),
        "anchor_count_mismatches": int(counts[4]),
        "invalid_intervals": int(counts[5]),
        "nonpositive_anchor_counts": int(counts[6]),
    }
    result["invalid_cluster_ids"] = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM source_blocks_raw
            WHERE TRY_CAST(cluster_id AS BIGINT) IS NULL
            """
        ).fetchone()[0]
    )
    gene_counts = connection.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT protein_id)
        FROM source_gene_positions
        """
    ).fetchone()
    result["genes"] = int(gene_counts[0])
    result["distinct_genes"] = int(gene_counts[1])
    gene_contract = connection.execute(
        """
        WITH ordered AS (
            SELECT
                *,
                lag(start_bp) OVER (
                    PARTITION BY genome_id, contig_id
                    ORDER BY position_index
                ) AS previous_start_bp,
                lag(end_bp) OVER (
                    PARTITION BY genome_id, contig_id
                    ORDER BY position_index
                ) AS previous_end_bp
            FROM source_gene_positions
        )
        SELECT
            COUNT(*) FILTER (
                WHERE start_bp IS NULL OR end_bp IS NULL OR end_bp < start_bp
            ) AS invalid_gene_coordinates,
            COUNT(*) FILTER (
                WHERE previous_start_bp IS NOT NULL
                  AND (
                      start_bp < previous_start_bp
                      OR (
                          start_bp = previous_start_bp
                          AND end_bp < previous_end_bp
                      )
                  )
            ) AS coordinate_order_violations
        FROM ordered
        """
    ).fetchone()
    result["invalid_gene_coordinates"] = int(gene_contract[0])
    result["coordinate_order_violations"] = int(gene_contract[1])
    failures = {
        key: value
        for key, value in result.items()
        if key in {
            "invalid_orientations",
            "unequal_arrays",
            "anchor_count_mismatches",
            "invalid_intervals",
            "nonpositive_anchor_counts",
            "invalid_cluster_ids",
            "invalid_gene_coordinates",
            "coordinate_order_violations",
        }
        and value
    }
    if result["blocks"] != result["distinct_blocks"]:
        failures["duplicate_block_ids"] = result["blocks"] - result["distinct_blocks"]
    if result["genes"] != result["distinct_genes"]:
        failures["duplicate_protein_ids"] = result["genes"] - result["distinct_genes"]
    if failures:
        raise ELSAResultContractError(
            f"ELSA result source validation failed: {failures}"
        )
    return result


def _materialize_run(
    connection: duckdb.DuckDBPyConnection,
    *,
    manifest: Mapping[str, Any],
    activate: bool,
) -> ResultStoreSummary:
    run_id = str(manifest["run_id"])
    connection.execute(
        """
        INSERT INTO elsa_runs(
            run_id, run_label, created_at, status, is_active, dataset_id,
            dataset_seal_json, elsa_version, elsa_commit, mapping_version,
            parameters_json, inputs_json, identity_sha256
        ) VALUES (?, ?, ?, 'building', FALSE, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            manifest["run_label"],
            manifest["created_at"],
            manifest.get("dataset_id"),
            _canonical_json(manifest.get("dataset_seal")),
            manifest.get("elsa_version", __version__),
            manifest.get("elsa_commit"),
            manifest.get("mapping_version", MAPPING_VERSION),
            _canonical_json(manifest.get("parameters", {})),
            _canonical_json(manifest.get("inputs", {})),
            manifest["identity_sha256"],
        ],
    )
    _result_log("materializing exact gene map")
    connection.execute(
        """
        INSERT INTO elsa_genes
        SELECT ?, protein_id, genome_id, contig_id, position_index,
               start_bp, end_bp, strand
        FROM source_gene_positions
        """,
        [run_id],
    )
    _result_log("materializing blocks")
    connection.execute(
        """
        INSERT INTO elsa_blocks
        SELECT
            ?, block_id, cluster_key, source_cluster_id,
            query_genome, target_genome, query_contig, target_contig,
            query_start, query_end, target_start, target_end,
            query_start_bp, query_end_bp, target_start_bp, target_end_bp,
            n_anchors, chain_score, orientation
        FROM source_blocks
        """,
        [run_id],
    )
    _result_log("resolving orientation-aware anchor pairs")
    connection.execute(
        """
        INSERT INTO elsa_anchor_pairs
        WITH parsed AS (
            SELECT
                block_id,
                cluster_key,
                orientation,
                from_json(query_anchor_genes, '["VARCHAR"]') AS query_ids,
                from_json(target_anchor_genes, '["VARCHAR"]') AS target_ids
            FROM source_blocks
        ),
        expanded AS (
            SELECT
                block_id,
                cluster_key,
                i - 1 AS pair_order,
                query_ids[i] AS query_protein_id,
                target_ids[
                    CASE
                        WHEN orientation = 1 THEN i
                        ELSE len(target_ids) - i + 1
                    END
                ] AS target_protein_id
            FROM parsed, generate_series(1, len(query_ids)) AS series(i)
        )
        SELECT
            ?,
            expanded.block_id,
            expanded.cluster_key,
            expanded.pair_order,
            expanded.query_protein_id,
            expanded.target_protein_id,
            query_gene.position_index,
            target_gene.position_index
        FROM expanded
        JOIN source_gene_positions AS query_gene
          ON query_gene.protein_id = expanded.query_protein_id
        JOIN source_gene_positions AS target_gene
          ON target_gene.protein_id = expanded.target_protein_id
        """,
        [run_id],
    )
    expected_pairs = int(
        connection.execute("SELECT SUM(n_anchors) FROM source_blocks").fetchone()[0] or 0
    )
    observed_pairs = int(
        connection.execute(
            "SELECT COUNT(*) FROM elsa_anchor_pairs WHERE run_id = ?",
            [run_id],
        ).fetchone()[0]
    )
    if expected_pairs != observed_pairs:
        raise ELSAResultContractError(
            "Anchor proteins failed exact gene-map resolution: "
            f"expected {expected_pairs}, materialized {observed_pairs}"
        )
    misplaced_anchor_pairs = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM elsa_anchor_pairs AS pairs
            JOIN source_blocks AS blocks
              ON blocks.block_id = pairs.block_id
             AND blocks.cluster_key = pairs.cluster_key
            JOIN source_gene_positions AS query_gene
              ON query_gene.protein_id = pairs.query_protein_id
            JOIN source_gene_positions AS target_gene
              ON target_gene.protein_id = pairs.target_protein_id
            WHERE pairs.run_id = ?
              AND (
                  query_gene.genome_id != blocks.query_genome
                  OR query_gene.contig_id != blocks.query_contig
                  OR query_gene.position_index NOT BETWEEN
                      blocks.query_start AND blocks.query_end
                  OR target_gene.genome_id != blocks.target_genome
                  OR target_gene.contig_id != blocks.target_contig
                  OR target_gene.position_index NOT BETWEEN
                      blocks.target_start AND blocks.target_end
              )
            """,
            [run_id],
        ).fetchone()[0]
    )
    if misplaced_anchor_pairs:
        raise ELSAResultContractError(
            "Anchor proteins violate block-side coordinates: "
            f"{misplaced_anchor_pairs} pair(s)"
        )

    _result_log("building complete cluster summaries")
    connection.execute(
        """
        INSERT INTO elsa_clusters
        WITH genomes AS (
            SELECT cluster_key, query_genome AS genome_id FROM source_blocks
            UNION
            SELECT cluster_key, target_genome AS genome_id FROM source_blocks
        ),
        support AS (
            SELECT cluster_key, COUNT(*) AS genome_support
            FROM genomes
            GROUP BY cluster_key
        ),
        reported AS (
            SELECT
                TRY_CAST(cluster_id AS BIGINT) AS source_cluster_id,
                MAX(TRY_CAST(size AS BIGINT)) AS reported_size,
                MAX(TRY_CAST(genome_support AS BIGINT)) AS reported_genome_support,
                MAX(TRY_CAST(mean_chain_length AS DOUBLE)) AS reported_mean_chain_length
            FROM source_clusters_raw
            GROUP BY TRY_CAST(cluster_id AS BIGINT)
        )
        SELECT
            ?,
            blocks.cluster_key,
            MAX(blocks.source_cluster_id),
            CASE WHEN COUNT(*) = 1 THEN 'singleton' ELSE 'cluster' END,
            COUNT(*),
            MAX(support.genome_support),
            AVG(blocks.n_anchors),
            AVG(blocks.chain_score),
            0,
            0,
            0,
            reported.source_cluster_id IS NOT NULL,
            reported.reported_size,
            reported.reported_genome_support,
            reported.reported_mean_chain_length,
            NULL,
            NULL,
            NULL
        FROM source_blocks AS blocks
        JOIN support USING (cluster_key)
        LEFT JOIN reported
          ON blocks.source_cluster_id = reported.source_cluster_id
        GROUP BY
            blocks.cluster_key,
            reported.source_cluster_id,
            reported.reported_size,
            reported.reported_genome_support,
            reported.reported_mean_chain_length
        """,
        [run_id],
    )
    _result_log("merging cluster intervals into loci")
    connection.execute(
        """
        INSERT INTO elsa_cluster_loci
        WITH contig_extent AS (
            SELECT genome_id, contig_id, MAX(position_index) AS last_position_index
            FROM source_gene_positions
            GROUP BY genome_id, contig_id
        ),
        half_loci AS (
            SELECT
                cluster_key, block_id, query_genome AS genome_id,
                query_contig AS contig_id, query_start AS start_idx,
                query_end AS end_idx
            FROM source_blocks
            UNION ALL
            SELECT
                cluster_key, block_id, target_genome AS genome_id,
                target_contig AS contig_id, target_start AS start_idx,
                target_end AS end_idx
            FROM source_blocks
        ),
        running AS (
            SELECT
                *,
                MAX(end_idx) OVER (
                    PARTITION BY cluster_key, genome_id, contig_id
                    ORDER BY start_idx, end_idx, block_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS previous_max_end
            FROM half_loci
        ),
        marked AS (
            SELECT
                *,
                CASE
                    WHEN previous_max_end IS NULL OR start_idx > previous_max_end + 1
                    THEN 1 ELSE 0
                END AS starts_group
            FROM running
        ),
        grouped AS (
            SELECT
                *,
                SUM(starts_group) OVER (
                    PARTITION BY cluster_key, genome_id, contig_id
                    ORDER BY start_idx, end_idx, block_id
                    ROWS UNBOUNDED PRECEDING
                ) AS interval_group
            FROM marked
        ),
        merged AS (
            SELECT
                cluster_key, genome_id, contig_id, interval_group,
                MIN(start_idx) AS start_idx,
                MAX(end_idx) AS end_idx,
                COUNT(DISTINCT block_id) AS block_support
            FROM grouped
            GROUP BY cluster_key, genome_id, contig_id, interval_group
        ),
        indexed AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY cluster_key
                    ORDER BY genome_id, contig_id, start_idx, end_idx
                ) - 1 AS locus_index
            FROM merged
        )
        SELECT
            ?,
            indexed.cluster_key,
            indexed.cluster_key || ':locus:' || CAST(indexed.locus_index AS VARCHAR),
            indexed.locus_index,
            indexed.genome_id,
            indexed.contig_id,
            indexed.start_idx,
            indexed.end_idx,
            start_gene.start_bp,
            end_gene.end_bp,
            start_gene.protein_id,
            end_gene.protein_id,
            indexed.end_idx - indexed.start_idx + 1,
            indexed.block_support,
            indexed.start_idx = 0,
            indexed.end_idx = contig_extent.last_position_index
        FROM indexed
        JOIN contig_extent
          ON contig_extent.genome_id = indexed.genome_id
         AND contig_extent.contig_id = indexed.contig_id
        LEFT JOIN source_gene_positions AS start_gene
          ON start_gene.genome_id = indexed.genome_id
         AND start_gene.contig_id = indexed.contig_id
         AND start_gene.position_index = indexed.start_idx
        LEFT JOIN source_gene_positions AS end_gene
          ON end_gene.genome_id = indexed.genome_id
         AND end_gene.contig_id = indexed.contig_id
         AND end_gene.position_index = indexed.end_idx
        """,
        [run_id],
    )
    _result_log("expanding exact anchor/context membership")
    connection.execute(
        """
        INSERT INTO elsa_cluster_members
        WITH anchors AS (
            SELECT DISTINCT blocks.cluster_key, pairs.query_protein_id AS protein_id
            FROM elsa_anchor_pairs AS pairs
            JOIN elsa_blocks AS blocks
              ON blocks.run_id = pairs.run_id AND blocks.block_id = pairs.block_id
            WHERE pairs.run_id = ?
            UNION
            SELECT DISTINCT blocks.cluster_key, pairs.target_protein_id AS protein_id
            FROM elsa_anchor_pairs AS pairs
            JOIN elsa_blocks AS blocks
              ON blocks.run_id = pairs.run_id AND blocks.block_id = pairs.block_id
            WHERE pairs.run_id = ?
        )
        SELECT
            loci.run_id,
            loci.cluster_key,
            loci.locus_key,
            genes.protein_id,
            genes.genome_id,
            genes.contig_id,
            genes.position_index,
            genes.position_index - loci.start_position_index,
            CASE WHEN anchors.protein_id IS NULL THEN 'context' ELSE 'anchor' END,
            genes.start_bp,
            genes.end_bp,
            genes.strand
        FROM elsa_cluster_loci AS loci
        JOIN elsa_genes AS genes
          ON genes.run_id = loci.run_id
         AND genes.genome_id = loci.genome_id
         AND genes.contig_id = loci.contig_id
         AND genes.position_index BETWEEN
             loci.start_position_index AND loci.end_position_index
        LEFT JOIN anchors
          ON anchors.cluster_key = loci.cluster_key
         AND anchors.protein_id = genes.protein_id
        WHERE loci.run_id = ?
        """,
        [run_id, run_id, run_id],
    )
    _result_log("computing cluster locus and membership counts")
    connection.execute(
        """
        UPDATE elsa_clusters AS clusters
        SET
            locus_count = counts.locus_count,
            member_count = counts.member_count,
            anchor_member_count = counts.anchor_member_count,
            truncated_locus_count = counts.truncated_locus_count,
            locus_size_median = counts.locus_size_median,
            locus_size_iqr = counts.locus_size_iqr
        FROM (
            WITH membership AS (
                SELECT
                    loci.run_id,
                    loci.cluster_key,
                    COUNT(DISTINCT loci.locus_key) AS locus_count,
                    COUNT(DISTINCT members.protein_id) AS member_count,
                    COUNT(DISTINCT members.protein_id)
                        FILTER (WHERE members.member_role = 'anchor')
                        AS anchor_member_count
                FROM elsa_cluster_loci AS loci
                LEFT JOIN elsa_cluster_members AS members
                  ON members.run_id = loci.run_id
                 AND members.locus_key = loci.locus_key
                WHERE loci.run_id = ?
                GROUP BY loci.run_id, loci.cluster_key
            ),
            -- Locus-grain statistics must not be computed across the member
            -- join, which repeats each locus once per member protein.
            locus_shape AS (
                SELECT
                    run_id,
                    cluster_key,
                    COUNT(*) FILTER (WHERE truncated_start OR truncated_end)
                        AS truncated_locus_count,
                    MEDIAN(n_genes) AS locus_size_median,
                    quantile_cont(n_genes, 0.75)
                        - quantile_cont(n_genes, 0.25) AS locus_size_iqr
                FROM elsa_cluster_loci
                WHERE run_id = ?
                GROUP BY run_id, cluster_key
            )
            SELECT
                membership.run_id,
                membership.cluster_key,
                membership.locus_count,
                membership.member_count,
                membership.anchor_member_count,
                locus_shape.truncated_locus_count,
                locus_shape.locus_size_median,
                locus_shape.locus_size_iqr
            FROM membership
            JOIN locus_shape USING (run_id, cluster_key)
        ) AS counts
        WHERE clusters.run_id = counts.run_id
          AND clusters.cluster_key = counts.cluster_key
        """,
        [run_id, run_id],
    )

    missing_endpoints = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM elsa_cluster_loci
            WHERE run_id = ?
              AND (start_protein_id IS NULL OR end_protein_id IS NULL)
            """,
            [run_id],
        ).fetchone()[0]
    )
    orphan_blocks = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM elsa_blocks AS blocks
            LEFT JOIN elsa_clusters AS clusters
              ON clusters.run_id = blocks.run_id
             AND clusters.cluster_key = blocks.cluster_key
            WHERE blocks.run_id = ? AND clusters.cluster_key IS NULL
            """,
            [run_id],
        ).fetchone()[0]
    )
    if missing_endpoints or orphan_blocks:
        raise ELSAResultContractError(
            "Materialized result integrity failed: "
            f"missing_locus_endpoints={missing_endpoints}, orphan_blocks={orphan_blocks}"
        )

    counts = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM elsa_genes WHERE run_id = ?),
            (SELECT COUNT(*) FROM elsa_blocks WHERE run_id = ?),
            (SELECT COUNT(*) FROM elsa_clusters WHERE run_id = ?),
            (SELECT COUNT(*) FROM elsa_clusters
             WHERE run_id = ? AND cluster_kind = 'singleton'),
            (SELECT COUNT(*) FROM elsa_anchor_pairs WHERE run_id = ?),
            (SELECT COUNT(*) FROM elsa_cluster_loci WHERE run_id = ?),
            (SELECT COUNT(*) FROM elsa_cluster_members WHERE run_id = ?)
        """,
        [run_id] * 7,
    ).fetchone()
    validation = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "exact_anchor_resolution": True,
        "misplaced_anchor_pairs": misplaced_anchor_pairs,
        "missing_locus_endpoints": missing_endpoints,
        "orphan_blocks": orphan_blocks,
    }
    connection.execute(
        """
        UPDATE elsa_runs
        SET status = 'ready',
            validation_json = ?,
            gene_count = ?,
            block_count = ?,
            cluster_count = ?,
            singleton_count = ?,
            anchor_pair_count = ?,
            locus_count = ?,
            member_count = ?
        WHERE run_id = ?
        """,
        [
            _canonical_json(validation),
            *[int(value) for value in counts],
            run_id,
        ],
    )
    if activate:
        connection.execute("UPDATE elsa_runs SET is_active = FALSE")
        connection.execute(
            "UPDATE elsa_runs SET is_active = TRUE WHERE run_id = ?",
            [run_id],
        )
        connection.execute(
            "UPDATE elsa_dataset_state SET active_run_id = ? WHERE singleton = TRUE",
            [run_id],
        )

    return ResultStoreSummary(
        path=Path(),
        run_id=run_id,
        gene_count=int(counts[0]),
        block_count=int(counts[1]),
        cluster_count=int(counts[2]),
        singleton_count=int(counts[3]),
        anchor_pair_count=int(counts[4]),
        locus_count=int(counts[5]),
        member_count=int(counts[6]),
    )


def _create_indexes_and_views(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_elsa_genes_protein
            ON elsa_genes(run_id, protein_id);
        CREATE INDEX IF NOT EXISTS idx_elsa_genes_locus
            ON elsa_genes(run_id, genome_id, contig_id, position_index);
        CREATE INDEX IF NOT EXISTS idx_elsa_blocks_cluster
            ON elsa_blocks(run_id, cluster_key);
        CREATE INDEX IF NOT EXISTS idx_elsa_anchor_query
            ON elsa_anchor_pairs(run_id, query_protein_id);
        CREATE INDEX IF NOT EXISTS idx_elsa_anchor_target
            ON elsa_anchor_pairs(run_id, target_protein_id);
        CREATE INDEX IF NOT EXISTS idx_elsa_clusters_source
            ON elsa_clusters(run_id, source_cluster_id);
        CREATE INDEX IF NOT EXISTS idx_elsa_loci_cluster
            ON elsa_cluster_loci(run_id, cluster_key);
        CREATE INDEX IF NOT EXISTS idx_elsa_members_protein
            ON elsa_cluster_members(run_id, protein_id);
        CREATE INDEX IF NOT EXISTS idx_elsa_members_cluster
            ON elsa_cluster_members(run_id, cluster_key);

        CREATE OR REPLACE VIEW current_elsa_run AS
        SELECT runs.*
        FROM elsa_runs AS runs
        JOIN elsa_dataset_state AS state
          ON state.active_run_id = runs.run_id
        WHERE state.singleton = TRUE;

        CREATE OR REPLACE VIEW current_elsa_blocks AS
        SELECT blocks.*
        FROM elsa_blocks AS blocks
        JOIN elsa_dataset_state AS state
          ON state.active_run_id = blocks.run_id
        WHERE state.singleton = TRUE;

        CREATE OR REPLACE VIEW current_elsa_clusters AS
        SELECT clusters.*
        FROM elsa_clusters AS clusters
        JOIN elsa_dataset_state AS state
          ON state.active_run_id = clusters.run_id
        WHERE state.singleton = TRUE;

        CREATE OR REPLACE VIEW current_elsa_cluster_loci AS
        SELECT loci.*
        FROM elsa_cluster_loci AS loci
        JOIN elsa_dataset_state AS state
          ON state.active_run_id = loci.run_id
        WHERE state.singleton = TRUE;

        CREATE OR REPLACE VIEW current_elsa_cluster_members AS
        SELECT members.*
        FROM elsa_cluster_members AS members
        JOIN elsa_dataset_state AS state
          ON state.active_run_id = members.run_id
        WHERE state.singleton = TRUE;
        """
    )


def materialize_result_store(
    output_path: str | Path,
    *,
    blocks: str | Path | pd.DataFrame,
    genes: str | Path | pd.DataFrame,
    manifest: Mapping[str, Any],
    clusters: str | Path | pd.DataFrame | None = None,
    activate: bool = True,
    replace_run: bool = False,
    threads: int | None = None,
) -> ResultStoreSummary:
    """Materialize one run into a normalized DuckDB sidecar.

    ``genes`` must use the exact row order supplied to ELSA after canonical
    coordinate sorting.  A :class:`SyntenyStore` ``metadata.parquet`` file
    satisfies this contract.
    """

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        raise ELSAResultContractError("Run manifest requires run_id")

    connection = duckdb.connect(str(output))
    try:
        if threads is not None:
            if threads < 1:
                raise ValueError("threads must be at least 1")
            connection.execute(f"SET threads = {int(threads)}")
        connection.execute("SET preserve_insertion_order = false")
        _initialize_schema(connection)
        exists = connection.execute(
            "SELECT COUNT(*) FROM elsa_runs WHERE run_id = ?",
            [run_id],
        ).fetchone()[0]
        if exists and not replace_run:
            raise ELSAResultContractError(
                f"Run {run_id} already exists in {output}; pass replace_run=True"
            )

        _prepare_sources(
            connection,
            blocks=blocks,
            genes=genes,
            clusters=clusters,
        )
        _result_log("validating source IDs, orientations, and anchor arrays")
        source_validation = _validate_sources(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            if exists:
                _delete_run(connection, run_id)
            summary = _materialize_run(
                connection,
                manifest=manifest,
                activate=activate,
            )
            connection.execute(
                """
                UPDATE elsa_runs
                SET validation_json = ?
                WHERE run_id = ?
                """,
                [
                    _canonical_json(
                        {
                            **json.loads(
                                connection.execute(
                                    "SELECT validation_json FROM elsa_runs WHERE run_id = ?",
                                    [run_id],
                                ).fetchone()[0]
                            ),
                            "source": source_validation,
                        }
                    ),
                    run_id,
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        _result_log("creating exact query indexes and active-run views")
        _create_indexes_and_views(connection)
        _result_log("checkpointing normalized sidecar")
        connection.execute("CHECKPOINT")
        return ResultStoreSummary(
            **{**asdict(summary), "path": output}
        )
    finally:
        connection.close()


def infer_result_sources(output_dir: str | Path) -> tuple[Path, Path | None]:
    """Return canonical block and optional cluster sources from an output directory."""

    root = Path(output_dir).expanduser().resolve()
    blocks = root / "micro_chain_blocks.csv"
    if not blocks.is_file():
        raise FileNotFoundError(blocks)
    clusters = root / "micro_chain_clusters.csv"
    return blocks, clusters if clusters.is_file() else None


__all__ = [
    "ELSAResultContractError",
    "MAPPING_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ResultStoreSummary",
    "build_run_manifest",
    "fingerprint_file",
    "fingerprint_source_tree",
    "infer_result_sources",
    "materialize_result_store",
    "read_dataset_identity",
    "sha256_file",
]
