"""Command line interface for the operon embedding project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np

import json

from .config import load_config
from .io import ensure_directory
from .index_hnsw import build_hnsw_index, save_metadata
from .preprocess import (
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_preprocessor,
)
from .shingle import build_shingles
from .sinkhorn import sinkhorn_distance
from .graph import build_knn_graph
from .cluster import hdbscan_cluster, leiden_cluster

import igraph as ig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_CANDIDATES = (
    "embeddings.npy",
    "gene_embeddings.npy",
    "embeddings.npz",
    "gene_embeddings.npz",
)


def _placeholder(action: str) -> Callable[[argparse.Namespace], int]:
    def _inner(_args: argparse.Namespace) -> int:
        logger.error("Subcommand '%s' is not implemented yet", action)
        return 1

    return _inner


def _load_embedding_array(path: Path) -> np.ndarray:
    if path.is_dir():
        for candidate in DEFAULT_EMBEDDING_CANDIDATES:
            candidate_path = path / candidate
            if candidate_path.exists():
                return _load_embedding_array(candidate_path)
        raise FileNotFoundError(
            f"No embeddings found in {path}. Expected one of: {', '.join(DEFAULT_EMBEDDING_CANDIDATES)}",
        )

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as archive:
            if "embeddings" in archive:
                return archive["embeddings"]
            return archive[archive.files[0]]

    raise ValueError(f"Unsupported embedding format: {path.suffix}")


def _resolve_embeddings_path(candidate: Optional[str], config: Dict[str, Any]) -> Path:
    if candidate:
        return Path(candidate).expanduser().resolve()

    paths_cfg = config.get("paths", {}) if config else {}
    embeddings_path = paths_cfg.get("embeddings")
    if embeddings_path:
        return Path(str(embeddings_path)).expanduser().resolve()

    data_root = paths_cfg.get("data_root")
    if data_root:
        return Path(str(data_root)).expanduser().resolve()

    raise FileNotFoundError(
        "Provide --embeddings or set paths.embeddings / paths.data_root in the config",
    )


def _resolve_sinkhorn_path(candidate: Optional[str], config: Dict[str, Any]) -> Path:
    if candidate:
        return Path(candidate).expanduser().resolve()

    paths_cfg = config.get("paths", {}) if config else {}
    sinkhorn = paths_cfg.get("sinkhorn_costs")
    if sinkhorn:
        return Path(str(sinkhorn)).expanduser().resolve()

    raise FileNotFoundError(
        "Provide --sinkhorn-costs or set paths.sinkhorn_costs in the config",
    )


def _resolve_preprocessor_path(
    candidate: Optional[str], config: Dict[str, Any]
) -> Path:
    if candidate:
        return Path(candidate).expanduser().resolve()

    paths_cfg = config.get("paths", {}) if config else {}
    preproc = paths_cfg.get("preprocessor")
    if preproc:
        return Path(str(preproc)).expanduser().resolve()

    raise FileNotFoundError(
        "Provide --preprocessor or set paths.preprocessor in the config",
    )


def _load_contig_sizes(path: Optional[str], total_genes: int) -> List[int]:
    if path is None:
        return [total_genes]

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not all(isinstance(x, int) for x in data):
        raise ValueError("Contig sizes file must be a JSON list of integers")

    sizes = [int(x) for x in data if int(x) > 0]
    if sum(sizes) != total_genes:
        raise ValueError(
            "Sum of contig sizes does not match number of embeddings provided",
        )
    return sizes


def _resolve_shingles_path(candidate: Optional[str], config: Dict[str, Any]) -> Path:
    if candidate:
        return Path(candidate).expanduser().resolve()

    shingles_path = config.get("paths", {}).get("shingles") if config else None
    if shingles_path:
        return Path(str(shingles_path)).expanduser().resolve()

    raise FileNotFoundError(
        "Provide --shingles or set paths.shingles in the config",
    )


def cmd_fit_preproc(args: argparse.Namespace) -> int:
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1

    try:
        embeddings_root = _resolve_embeddings_path(args.embeddings, config)
        data = _load_embedding_array(embeddings_root)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.limit is not None:
        data = data[: args.limit]

    dims = args.dims
    eps = args.eps
    preprocess_cfg = config.get("preprocess", {}) if config else {}

    if dims is None:
        dims = int(preprocess_cfg.get("pca_dims", 96))
    if eps is None:
        eps = float(preprocess_cfg.get("eps", 1e-5))

    try:
        artifacts = fit_preprocessor(data, dims_out=dims, eps=eps)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    output_dir = ensure_directory(args.output_dir)
    artifact_path = output_dir / "preprocessor.joblib"
    joblib.dump(save_preprocessor(artifacts), artifact_path)
    logger.info("Saved preprocessor artifacts to %s", artifact_path)
    print(f"Saved preprocessor artifacts to {artifact_path}")  # noqa: T201
    return 0


def cmd_build_shingles(args: argparse.Namespace) -> int:
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1

    try:
        embeddings_root = _resolve_embeddings_path(args.embeddings, config)
        data = _load_embedding_array(embeddings_root)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.limit is not None:
        data = data[: args.limit]

    try:
        preproc_path = _resolve_preprocessor_path(args.preprocessor, config)
        artifacts = load_preprocessor(preproc_path)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    transformed = transform_preprocessor(data, artifacts)

    try:
        contig_sizes = _load_contig_sizes(
            args.contig_sizes or config.get("paths", {}).get("contig_sizes"),
            len(transformed),
        )
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    contigs: List[np.ndarray] = []
    offset = 0
    for size in contig_sizes:
        contigs.append(transformed[offset : offset + size])
        offset += size

    result = build_shingles(contigs, k=args.k, stride=args.stride)

    output_dir = ensure_directory(args.output_dir)
    output_path = output_dir / "shingles.npz"
    indices_array = np.asarray(result.gene_indices, dtype=np.int64)
    if indices_array.ndim == 1:
        indices_array = indices_array.reshape(-1, 2)
    np.savez_compressed(
        output_path,
        vectors=result.vectors,
        gene_indices=indices_array,
        k=args.k,
        stride=args.stride,
    )
    logger.info("Saved shingle vectors to %s", output_path)
    print(f"Saved shingle vectors to {output_path}")  # noqa: T201
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1

    try:
        shingles_path = _resolve_shingles_path(args.shingles, config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    resolved = Path(shingles_path).expanduser().resolve()
    if not resolved.exists():
        logger.error("Shingles file not found: %s", resolved)
        return 1

    with np.load(resolved) as archive:
        if "vectors" not in archive:
            logger.error("Shingles file missing 'vectors' array")
            return 1
        vectors = np.asarray(archive["vectors"], dtype=np.float32)

    if args.limit is not None:
        vectors = vectors[: args.limit]

    if vectors.size == 0:
        logger.error("No vectors available to build index")
        return 1

    try:
        index = build_hnsw_index(
            vectors,
            m=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            space="cosine",
        )
    except ImportError as exc:
        logger.error(str(exc))
        return 1

    output_dir = ensure_directory(args.output_dir)
    index_path = output_dir / "hnsw_index.bin"
    metadata_path = output_dir / "hnsw_index.json"

    index.save(index_path)
    save_metadata(
        {
            "dim": index.dim,
            "space": index.space,
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "num_vectors": int(vectors.shape[0]),
        },
        metadata_path,
    )
    logger.info("Saved HNSW index to %s", index_path)
    print(f"Saved HNSW index to {index_path}")  # noqa: T201
    return 0


def cmd_rerank(args: argparse.Namespace) -> int:
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1

    try:
        embeddings_root = _resolve_embeddings_path(args.embeddings, config)
        data = _load_embedding_array(embeddings_root)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.limit is not None:
        data = data[: args.limit]

    try:
        preproc_path = _resolve_preprocessor_path(args.preprocessor, config)
        artifacts = load_preprocessor(preproc_path)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    transformed = transform_preprocessor(data, artifacts)

    pairs_path = Path(args.pairs).expanduser().resolve()
    if not pairs_path.exists():
        logger.error("Pairs file not found: %s", pairs_path)
        return 1

    with pairs_path.open("r", encoding="utf-8") as handle:
        pairs_data = json.load(handle)

    results = []
    for idx, pair in enumerate(pairs_data):
        query_ids = pair.get("query", [])
        target_ids = pair.get("target", [])
        if not query_ids or not target_ids:
            continue
        try:
            query_set = transformed[np.asarray(query_ids, dtype=np.int64)]
            target_set = transformed[np.asarray(target_ids, dtype=np.int64)]
        except IndexError:
            logger.warning("Skipping pair %s due to invalid indices", idx)
            continue

        try:
            result = sinkhorn_distance(
                query_set,
                target_set,
                epsilon=args.epsilon,
                n_iter=args.iterations,
                top_k=args.topk,
            )
        except ValueError as exc:
            logger.warning("Skipping pair %s: %s", idx, exc)
            continue

        results.append(
            {
                "pair_index": idx,
                "transport_cost": result.transport_cost,
                "similarity": result.similarity,
            }
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info("Saved Sinkhorn rerank results to %s", output_path)
    print(f"Saved Sinkhorn rerank results to {output_path}")  # noqa: T201
    return 0


def _load_cost_matrix(path: Path) -> np.ndarray:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Sinkhorn cost matrix not found: {resolved}")
    if resolved.suffix == ".npy":
        return np.asarray(np.load(resolved), dtype=np.float64)
    if resolved.suffix == ".npz":
        with np.load(resolved) as archive:
            if "costs" in archive:
                return np.asarray(archive["costs"], dtype=np.float64)
            return np.asarray(archive[archive.files[0]], dtype=np.float64)
    raise ValueError(f"Unsupported sinkhorn matrix format: {resolved.suffix}")


def cmd_graph(args: argparse.Namespace) -> int:
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1

    try:
        embeddings_path = _resolve_embeddings_path(args.features, config)
        embeddings = _load_embedding_array(embeddings_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        sinkhorn_path = _resolve_sinkhorn_path(args.sinkhorn_costs, config)
        sinkhorn_costs = _load_cost_matrix(sinkhorn_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    graph_cfg = config.get("graph", {}) if config else {}
    k = int(args.k or graph_cfg.get("knn", 64))
    prune_threshold = float(args.prune or graph_cfg.get("prune_threshold", 0.5))
    lambda_cosine = float(args.lambda_cosine or graph_cfg.get("lambda_cosine", 0.7))
    tau = float(args.tau or graph_cfg.get("tau", 0.1))

    try:
        graph = build_knn_graph(
            embeddings,
            sinkhorn_costs,
            k=k,
            prune_threshold=prune_threshold,
            lambda_cosine=lambda_cosine,
            tau=tau,
        )
    except ValueError as exc:
        logger.error("Failed to construct graph: %s", exc)
        return 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edges = graph.get_edgelist()
    sources = np.array([edge[0] for edge in edges], dtype=np.int64)
    targets = np.array([edge[1] for edge in edges], dtype=np.int64)
    weights = (
        np.array(graph.es["weight"], dtype=np.float32)
        if graph.ecount()
        else np.array([], dtype=np.float32)
    )

    np.savez(
        output_path,
        sources=sources,
        targets=targets,
        weights=weights,
        n_vertices=np.array([graph.vcount()], dtype=np.int64),
    )

    logger.info(
        "Saved graph with %d vertices and %d edges to %s",
        graph.vcount(),
        graph.ecount(),
        output_path,
    )
    print(  # noqa: T201
        f"Saved graph with {graph.vcount()} vertices and {graph.ecount()} edges to {output_path}"
    )
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1

    graph_path = Path(args.graph).expanduser().resolve()
    if not graph_path.exists():
        logger.error("Graph file not found: %s", graph_path)
        return 1

    with np.load(graph_path) as archive:
        sources = archive.get("sources", np.array([], dtype=np.int64)).astype(np.int64)
        targets = archive.get("targets", np.array([], dtype=np.int64)).astype(np.int64)
        weights = archive.get("weights", np.array([], dtype=np.float32)).astype(
            np.float32
        )
        n_vertices_array = archive.get("n_vertices")
        if n_vertices_array is None:
            logger.error("Graph file missing 'n_vertices'")
            return 1
        n_vertices = int(np.asarray(n_vertices_array).flatten()[0])

    graph = ig.Graph()
    graph.add_vertices(n_vertices)
    if len(sources) != len(targets):
        logger.error("sources and targets length mismatch in graph file")
        return 1
    if sources.size:
        edges = list(zip(sources.tolist(), targets.tolist()))
        graph.add_edges(edges)
        if weights.size:
            graph.es["weight"] = weights.tolist()

    cluster_cfg = config.get("cluster", {}) if config else {}
    method = (args.method or cluster_cfg.get("method", "leiden")).lower()

    if method == "leiden":
        resolution = float(args.resolution or cluster_cfg.get("leiden_resolution", 1.0))
        assignments = leiden_cluster(graph, resolution)
    elif method == "hdbscan":
        features_path = args.features or cluster_cfg.get("features")
        if not features_path:
            logger.error("--features must be provided for HDBSCAN clustering")
            return 1
        try:
            embeddings_path = _resolve_embeddings_path(features_path, config)
            features = _load_embedding_array(embeddings_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        if features.shape[0] < n_vertices:
            logger.error(
                "Feature matrix has fewer rows (%d) than graph vertices (%d)",
                features.shape[0],
                n_vertices,
            )
            return 1
        min_cluster_size = int(
            args.min_cluster_size or cluster_cfg.get("min_cluster_size", 5)
        )
        min_samples = args.min_samples or cluster_cfg.get("min_samples")
        assignments = hdbscan_cluster(
            features[:n_vertices],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
    else:
        logger.error("Unknown clustering method: %s", method)
        return 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"assignments": assignments}
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(
        "Saved clustering assignments (%d vertices) to %s",
        len(assignments),
        output_path,
    )
    print(f"Saved clustering assignments to {output_path}")  # noqa: T201
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operon embedding pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    commands: Dict[str, Callable[[argparse.Namespace], int]] = {
        "fit-preproc": cmd_fit_preproc,
        "build-shingles": cmd_build_shingles,
        "build-index": cmd_build_index,
        "train-metric": _placeholder("train-metric"),
        "retrieve": _placeholder("retrieve"),
        "rerank": cmd_rerank,
        "graph": cmd_graph,
        "cluster": cmd_cluster,
        "eval": _placeholder("eval"),
    }

    fit_parser = subparsers.add_parser(
        "fit-preproc", help="Fit whitening + PCA preprocessor"
    )
    fit_parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to embeddings file (.npy/.npz) or directory",
    )
    fit_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the preprocessor joblib",
    )
    fit_parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help="Number of PCA dimensions (overrides config)",
    )
    fit_parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Eigenvalue stabilizer (overrides config)",
    )
    fit_parser.add_argument(
        "--limit",
        type=int,
        help="Optional sample limit for fitting",
    )
    fit_parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline configuration YAML",
    )

    shingle_parser = subparsers.add_parser(
        "build-shingles", help="Construct order-invariant shingle vectors"
    )
    shingle_parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to embeddings file (.npy/.npz) or directory",
    )
    shingle_parser.add_argument(
        "--preprocessor",
        type=str,
        help="Path to preprocessor.joblib (overrides config)",
    )
    shingle_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the shingle NPZ",
    )
    shingle_parser.add_argument(
        "--contig-sizes",
        type=str,
        help="JSON file listing contig lengths",
    )
    shingle_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Shingle size (number of genes)",
    )
    shingle_parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between shingles",
    )
    shingle_parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on number of genes processed",
    )
    shingle_parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline configuration YAML",
    )

    index_parser = subparsers.add_parser(
        "build-index", help="Build an HNSW index over shingle vectors"
    )
    index_parser.add_argument(
        "--shingles",
        type=str,
        help="Path to shingles NPZ (overrides config)",
    )
    index_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the index files will be written",
    )
    index_parser.add_argument(
        "--M",
        type=int,
        default=32,
        help="HNSW graph degree (M)",
    )
    index_parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW efConstruction parameter",
    )
    index_parser.add_argument(
        "--ef-search",
        type=int,
        default=128,
        help="HNSW efSearch parameter",
    )
    index_parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on number of vectors indexed",
    )
    index_parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline configuration YAML",
    )

    rerank_parser = subparsers.add_parser(
        "rerank", help="Compute Sinkhorn similarities for candidate pairs"
    )
    rerank_parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to embeddings file (.npy/.npz) or directory",
    )
    rerank_parser.add_argument(
        "--preprocessor",
        type=str,
        help="Path to preprocessor.joblib (overrides config)",
    )
    rerank_parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="JSON file containing candidate gene index pairs",
    )
    rerank_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for similarities",
    )
    rerank_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Sinkhorn regularization strength",
    )
    rerank_parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Number of Sinkhorn iterations",
    )
    rerank_parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Keep top-K partners per row",
    )
    rerank_parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on embeddings used",
    )
    rerank_parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline configuration YAML",
    )

    graph_parser = subparsers.add_parser(
        "graph", help="Construct a reciprocal kNN graph from embeddings"
    )
    graph_parser.add_argument(
        "--features",
        type=str,
        help="Path to embeddings array (.npy/.npz) or directory",
    )
    graph_parser.add_argument(
        "--sinkhorn-costs",
        type=str,
        required=True,
        help="Path to Sinkhorn cost matrix (.npy/.npz)",
    )
    graph_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination NPZ file for the graph",
    )
    graph_parser.add_argument(
        "--k",
        type=int,
        help="Number of neighbours to keep per vertex (overrides config)",
    )
    graph_parser.add_argument(
        "--lambda-cosine",
        type=float,
        help="Cosine similarity weight (overrides config)",
    )
    graph_parser.add_argument(
        "--tau",
        type=float,
        help="Temperature for Sinkhorn similarity (overrides config)",
    )
    graph_parser.add_argument(
        "--prune",
        type=float,
        help="Minimum combined similarity threshold (overrides config)",
    )
    graph_parser.add_argument(
        "--config",
        type=str,
        help="Pipeline configuration YAML",
    )

    cluster_parser = subparsers.add_parser(
        "cluster", help="Cluster a graph using Leiden or HDBSCAN"
    )
    cluster_parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Graph NPZ produced by the 'graph' command",
    )
    cluster_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for cluster assignments",
    )
    cluster_parser.add_argument(
        "--method",
        type=str,
        choices=["leiden", "hdbscan"],
        help="Clustering method (overrides config)",
    )
    cluster_parser.add_argument(
        "--resolution",
        type=float,
        help="Leiden resolution parameter",
    )
    cluster_parser.add_argument(
        "--features",
        type=str,
        help="Feature matrix required for HDBSCAN clustering",
    )
    cluster_parser.add_argument(
        "--min-cluster-size",
        type=int,
        help="Minimum HDBSCAN cluster size",
    )
    cluster_parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum samples for HDBSCAN core points",
    )
    cluster_parser.add_argument(
        "--config",
        type=str,
        help="Pipeline configuration YAML",
    )

    for name, handler in commands.items():
        if name in {"fit-preproc", "build-shingles"}:
            continue
        subparsers.add_parser(name)

    parser.set_defaults(_commands=commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    handler = args._commands[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
