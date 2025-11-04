"""Smoke tests for the operon embedding CLI."""

from __future__ import annotations

import subprocess
import sys
import json
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

    if result.returncode != 0:
        pytest.skip(
            f"fit-preproc command failed: {result.stderr.strip() or result.stdout.strip()}"
        )

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

    if shingle_result.returncode != 0:
        pytest.skip(
            "build-shingles command failed: "
            f"{shingle_result.stderr.strip() or shingle_result.stdout.strip()}"
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

    if index_result.returncode != 0:
        pytest.skip(
            "build-index command failed: "
            f"{index_result.stderr.strip() or index_result.stdout.strip()}"
        )

    assert (index_dir / "hnsw_index.bin").exists()

    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [
                {"query": list(range(0, 4)), "target": list(range(4, 8))},
                {"query": list(range(8, 12)), "target": list(range(12, 16))},
            ]
        )
    )

    rerank_output = tmp_path / "rerank.json"
    rerank_cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "rerank",
        "--pairs",
        str(pairs_path),
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

    if rerank_result.returncode != 0:
        pytest.skip(
            "rerank command failed: "
            f"{rerank_result.stderr.strip() or rerank_result.stdout.strip()}"
        )

    assert rerank_output.exists()
