# Operon Embedding Architecture

This project discovers conserved operon-scale loci by chaining a set of modular
stages. Each stage can be used independently through the command line and is
fully configurable via `config.yaml` (or a user-supplied override).

1. **Preprocess (`fit-preproc`)** – fits Ledoit–Wolf whitening followed by PCA
   dimensionality reduction and L2 normalisation. Artifacts are saved as
   `preprocessor.joblib` and reused in downstream stages.
2. **Shingling (`build-shingles`)** – converts contig-ordered gene embeddings
   into order-invariant shingle vectors (size *k*, stride configurable) and
   stores both vectors and the underlying gene spans.
3. **Retrieval index (`build-index`)** – builds an HNSW index over shingle
   vectors. `retrieve` queries the index to emit top-*k* candidate neighbour
   shingles paired with their underlying gene IDs.
4. **Set-to-set similarity (`rerank`)** – projects per-gene embeddings through
   the preprocessor and computes Sinkhorn/EMD similarities for each candidate
   pair produced by the retrieval stage.
5. **Graph construction (`graph`)** – fuses cosine similarity and Sinkhorn
   distance to build a reciprocal k-NN graph of candidate operons.
6. **Clustering (`cluster`)** – partitions the graph via Leiden (or HDBSCAN as
   an alternative), producing JSON assignments for downstream consumption.
7. **Evaluation (`eval`)** – summarises similarities and clustering outcomes
   (mean/variance, ARI/AMI when labels are available) and writes a JSON report.

Wrapper scripts in `scripts/` simply forward to the matching CLI subcommand so
automations can invoke each stage with minimal boilerplate.
