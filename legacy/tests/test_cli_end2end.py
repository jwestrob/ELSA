"""Smoke tests for the operon embedding CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_MODULE = "operon_embed.cli"
EMBEDDING_CANDIDATES = (
    ("embeddings.npy", None),
    ("gene_embeddings.npy", None),
    ("embeddings.npz", "embeddings"),
    ("gene_embeddings.npz", "embeddings"),
)


@pytest.mark.parametrize("args", [["--help"], []])
def test_cli_invocation(args):
    """Ensure the CLI scaffold loads without raising exceptions."""
    result = subprocess.run(
        [sys.executable, "-m", CLI_MODULE, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Operon embedding pipeline" in result.stdout or result.stderr == ""


def _locate_embeddings(root: Path) -> tuple[Path, str | None] | None:
    import numpy as np

    for fname, key in EMBEDDING_CANDIDATES:
        candidate = root / fname
        if not candidate.exists():
            continue
        if candidate.suffix == ".npz":
            with np.load(candidate) as archive:
                if key is None:
                    return candidate, None
                if key in archive:
                    return candidate, key
        else:
            return candidate, None
    return None


@pytest.mark.usefixtures("operon_test_data")
def test_cli_fit_preproc_runs(tmp_path: Path, operon_test_data: Path) -> None:
    located = _locate_embeddings(operon_test_data)
    if located is None:
        pytest.skip("No embeddings available for CLI fit-preproc test")

    embeddings_path, _ = located
    output_dir = tmp_path / "preproc"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.yaml"
    preproc_output = output_dir / "preprocessor.joblib"
    config_path.write_text(
        "\n".join(
            [
                "preprocess:",
                "  pca_dims: 16",
                "  eps: 1.0e-5",
                "paths:",
                f"  embeddings: {embeddings_path}",
                f"  preprocessor: {preproc_output}",
            ]
        )
    )

    cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "fit-preproc",
        "--output-dir",
        str(output_dir),
        "--limit",
        "64",
        "--config",
        str(config_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"fit-preproc failed: {result.stderr.strip() or result.stdout.strip()}"
    assert preproc_output.exists()

    shingle_dir = tmp_path / "shingles"
    shingle_cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "build-shingles",
        "--output-dir",
        str(shingle_dir),
        "--limit",
        "64",
        "--k",
        "4",
        "--stride",
        "2",
        "--config",
        str(config_path),
    ]

    shingle_result = subprocess.run(
        shingle_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert shingle_result.returncode == 0, "build-shingles failed: " + (
        shingle_result.stderr.strip() or shingle_result.stdout.strip()
    )
    shingles_npz = shingle_dir / "shingles.npz"
    assert shingles_npz.exists()

    index_dir = tmp_path / "index"
    index_cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "build-index",
        "--shingles",
        str(shingles_npz),
        "--output-dir",
        str(index_dir),
        "--config",
        str(config_path),
        "--limit",
        "64",
    ]

    index_result = subprocess.run(
        index_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert index_result.returncode == 0, "build-index failed: " + (
        index_result.stderr.strip() or index_result.stdout.strip()
    )
    assert (index_dir / "hnsw_index.bin").exists()

    retrieve_output = tmp_path / "candidates.json"
    retrieve_cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "retrieve",
        "--shingles",
        str(shingles_npz),
        "--index",
        str(index_dir),
        "--output",
        str(retrieve_output),
        "--top-k",
        "2",
        "--limit",
        "32",
        "--config",
        str(config_path),
    ]

    retrieve_result = subprocess.run(
        retrieve_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert retrieve_result.returncode == 0, "retrieve failed: " + (
        retrieve_result.stderr.strip() or retrieve_result.stdout.strip()
    )
    assert retrieve_output.exists()

    rerank_output = tmp_path / "rerank.json"
    rerank_cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "rerank",
        "--pairs",
        str(retrieve_output),
        "--output",
        str(rerank_output),
        "--config",
        str(config_path),
        "--limit",
        "64",
    ]

    rerank_result = subprocess.run(
        rerank_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert rerank_result.returncode == 0, "rerank failed: " + (
        rerank_result.stderr.strip() or rerank_result.stdout.strip()
    )
    assert rerank_output.exists()

    eval_output = tmp_path / "eval.json"
    eval_cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "eval",
        "--pairs",
        str(rerank_output),
        "--output",
        str(eval_output),
        "--config",
        str(config_path),
    ]

    eval_result = subprocess.run(
        eval_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert eval_result.returncode == 0, "eval failed: " + (
        eval_result.stderr.strip() or eval_result.stdout.strip()
    )
    assert eval_output.exists()
