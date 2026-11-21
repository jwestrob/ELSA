# ELSA normalized result contract

ELSA’s canonical agent-facing result is a schema-versioned DuckDB sidecar.
The sidecar preserves exact protein identity, run provenance, orientation-
resolved anchors, complete clusters, loci, and many-to-many membership.

Legacy CSVs remain checkpoint and interchange artifacts.

## Materialization

New Sharur-backed runs create the sidecar automatically:

```bash
elsa synteny \
    --db data/DATASET/sharur.duckdb \
    --embeddings data/DATASET/embeddings/protein_embeddings.h5 \
    --store data/DATASET/synteny/store \
    --result-db data/DATASET/synteny.duckdb \
    --run-label DATASET-production \
    --jobs 16 \
    -o data/DATASET/synteny/results/
```

An existing completed checkpoint can be materialized directly:

```bash
elsa materialize-results data/DATASET/synteny/results/ \
    --store data/DATASET/synteny/store \
    --result-db data/DATASET/synteny.duckdb \
    --dataset-seal data/DATASET/dataset.seal.json \
    --parameters-file data/DATASET/synteny/run_parameters.json \
    --run-label DATASET-production \
    --threads 16
```

The store must be the exact store used to produce the block checkpoint.
`metadata.parquet` supplies the coordinate-sorted protein map.

## Run identity

Each run receives a deterministic identifier:

```text
elsa-<20 hexadecimal characters>
```

The identity digest covers:

- Sharur `dataset_id`, when a dataset seal is available;
- ELSA package version and Git commit;
- a complete content hash of the executable ELSA Python source tree;
- coordinate mapping version;
- complete chain parameters;
- complete SHA-256 fingerprints for input artifacts.

`elsa_runs` stores these fields, validation results, row counts, and active-run
state. Source cluster integers are scoped by `run_id`; `cluster_key` is the
canonical cluster reference inside that run.

## Schema

Schema version 1 contains:

| Relation | Grain | Purpose |
|---|---|---|
| `elsa_schema_version` | schema version | Compatibility check |
| `elsa_runs` | run | Provenance, status, counts, validation |
| `elsa_dataset_state` | sidecar | Active run pointer |
| `elsa_genes` | run × protein | Exact protein-to-ELSA-position map |
| `elsa_blocks` | run × block | Block coordinates, score, orientation, cluster |
| `elsa_anchor_pairs` | run × block × pair | Exact orientation-resolved homologous anchors |
| `elsa_clusters` | run × cluster | Complete cluster summary |
| `elsa_cluster_loci` | run × cluster × locus | Merged genomic intervals |
| `elsa_cluster_members` | run × cluster × protein | Exact `anchor`/`context` membership |

The `current_elsa_*` views expose the active run.

## Cluster semantics

Positive source cluster IDs map to:

```text
cluster:<source_cluster_id>
```

Each unclustered block receives its own key:

```text
singleton:<block_id>
```

This representation retains every block and makes singleton metadata
explicit. `cluster_kind` reflects the realized block count: `singleton` for
one block and `cluster` for larger groups.

## Coordinate semantics

`position_index` is ELSA’s zero-based position in coordinate-sorted
`metadata.parquet`, partitioned by genome and contig. It is the coordinate
contract used by block endpoints and locus intervals.

Sharur’s `proteins.gene_index` is an independent ingest field. Cross-contract
conversion proceeds through exact protein IDs.

## Anchor orientation

ELSA block CSVs serialize query and target anchor arrays independently in
ascending genomic-coordinate order.

- orientation `1`: query `i` pairs with target `i`;
- orientation `-1`: query `i` pairs with target `n - 1 - i`.

`elsa_anchor_pairs` stores the resolved pairs directly. Raw interchange audits
can use:

```python
from sharur.elsa_blocks import elsa_anchor_pairs_from_block

pairs = elsa_anchor_pairs_from_block(block_row)
```

## Exact queries

Protein membership:

```sql
SELECT members.*, clusters.size, clusters.genome_support
FROM current_elsa_cluster_members AS members
JOIN current_elsa_clusters AS clusters USING (run_id, cluster_key)
WHERE members.protein_id = ?
ORDER BY members.member_role, clusters.genome_support DESC, members.cluster_key;
```

Anchor blocks:

```sql
SELECT pairs.*, blocks.orientation, blocks.n_anchors, blocks.chain_score
FROM elsa_anchor_pairs AS pairs
JOIN elsa_blocks AS blocks USING (run_id, block_id)
WHERE pairs.run_id = ?
  AND (pairs.query_protein_id = ? OR pairs.target_protein_id = ?);
```

Run-scoped cluster:

```sql
SELECT *
FROM elsa_clusters
WHERE run_id = ? AND cluster_key = ?;
```

## Materialization invariants

A run reaches `ready` after all of these checks pass:

- unique block IDs;
- unique protein IDs in the supplied gene map;
- valid, nondecreasing gene coordinates in exact store row order;
- nonnegative, ordered block intervals and positive anchor counts;
- recognized block orientation;
- equal query and target anchor-array lengths;
- `n_anchors` agrees with the serialized arrays;
- every anchor resolves to an exact protein row;
- every anchor belongs to its declared block side, contig, and interval;
- every cluster locus resolves both endpoint proteins;
- every block resolves to exactly one explicit cluster key.

The materializer creates query indexes only after the run data and validation
records are complete.

## Architecture summaries

Schema/architecture generation accepts the real embedding matrix separately
from lightweight gene metadata. Production store runs pass
`store/embeddings.npy` as a memory-mapped array. Missing embeddings fail
closed. Architecture files are built in a staging directory and replace the
current `schema/` directory only after the complete stage succeeds, which
prevents zero-similarity or partial summaries from being published.
