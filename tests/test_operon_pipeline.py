"""Tests for the operon micro pipeline (Milestone 5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elsa.params import OperonConfig, ELSAConfig


def _hnswlib_available() -> bool:
    """Check if hnswlib is installed."""
    try:
        import hnswlib
        return True
    except ImportError:
        return False


class TestOperonConfig:
    """Tests for OperonConfig validation."""

    def test_default_config(self):
        """Default config should validate successfully."""
        cfg = OperonConfig()
        assert cfg.pca_dims == 96
        assert cfg.hnsw_m == 32
        assert cfg.hnsw_ef_construction == 200
        assert cfg.hnsw_ef_search == 128
        assert cfg.similarity_tau == 0.55
        assert cfg.min_genome_support == 2

    def test_custom_hnsw_params(self):
        """Custom HNSW params should be accepted."""
        cfg = OperonConfig(
            hnsw_m=16,
            hnsw_ef_construction=100,
            hnsw_ef_search=64,
            hnsw_top_k=8,
        )
        assert cfg.hnsw_m == 16
        assert cfg.hnsw_ef_construction == 100
        assert cfg.hnsw_ef_search == 64
        assert cfg.hnsw_top_k == 8

    def test_invalid_pca_dims(self):
        """PCA dims out of range should raise."""
        with pytest.raises(ValueError):
            OperonConfig(pca_dims=1)
        with pytest.raises(ValueError):
            OperonConfig(pca_dims=2000)

    def test_invalid_similarity_tau(self):
        """similarity_tau out of [0,1] should raise."""
        with pytest.raises(ValueError):
            OperonConfig(similarity_tau=-0.1)
        with pytest.raises(ValueError):
            OperonConfig(similarity_tau=1.5)

    def test_invalid_hnsw_m(self):
        """hnsw_m out of range should raise."""
        with pytest.raises(ValueError):
            OperonConfig(hnsw_m=1)
        with pytest.raises(ValueError):
            OperonConfig(hnsw_m=200)

    def test_operon_config_in_elsa_config(self):
        """OperonConfig should be accessible via ELSAConfig."""
        cfg = ELSAConfig()
        assert hasattr(cfg.analyze, 'operon')
        assert isinstance(cfg.analyze.operon, OperonConfig)
        assert cfg.analyze.operon.hnsw_m == 32


class TestOperonPipeline:
    """Tests for run_operon_pipeline function."""

    @pytest.fixture
    def synthetic_genes_parquet(self, tmp_path: Path) -> Path:
        """Create a synthetic genes.parquet for testing."""
        np.random.seed(42)
        n_genes = 50
        emb_dim = 64

        # Create synthetic gene data across 2 genomes, 2 contigs each
        data = {
            "gene_id": [f"gene_{i}" for i in range(n_genes)],
            "sample_id": ["genome_A"] * 25 + ["genome_B"] * 25,
            "contig_id": (["contig_1"] * 13 + ["contig_2"] * 12) * 2,
            "start": list(range(0, 25000, 500))[:25] + list(range(0, 25000, 500))[:25],
            "end": list(range(500, 25500, 500))[:25] + list(range(500, 25500, 500))[:25],
            "strand": [1, -1] * 25,
        }

        # Add embedding columns
        embeddings = np.random.randn(n_genes, emb_dim).astype(np.float32)
        for i in range(emb_dim):
            data[f"emb_{i}"] = embeddings[:, i]

        df = pd.DataFrame(data)
        parquet_path = tmp_path / "genes.parquet"
        df.to_parquet(parquet_path, index=False)
        return parquet_path

    def test_pipeline_runs_without_hnswlib(self, synthetic_genes_parquet: Path, tmp_path: Path):
        """Pipeline should handle missing hnswlib gracefully."""
        from elsa.analyze.micro_operon import run_operon_pipeline

        output_dir = tmp_path / "operon_out"
        summary = run_operon_pipeline(
            synthetic_genes_parquet,
            output_dir,
            pca_dims=16,
            shingle_k=3,
            shingle_stride=1,
            min_genome_support=1,
        )

        # Basic checks
        assert summary.num_genes == 50
        assert summary.num_blocks > 0
        assert (output_dir / "operon_blocks.csv").exists()
        assert (output_dir / "operon_summary.json").exists()

    @pytest.mark.skipif(
        not _hnswlib_available(),
        reason="hnswlib not installed"
    )
    def test_pipeline_with_hnsw(self, synthetic_genes_parquet: Path, tmp_path: Path):
        """Pipeline should build HNSW index when hnswlib is available."""
        from elsa.analyze.micro_operon import run_operon_pipeline

        output_dir = tmp_path / "operon_out"
        summary = run_operon_pipeline(
            synthetic_genes_parquet,
            output_dir,
            pca_dims=16,
            shingle_k=3,
            shingle_stride=1,
            hnsw_m=8,
            hnsw_ef_construction=50,
            hnsw_ef_search=32,
            hnsw_top_k=5,
            min_genome_support=1,
        )

        assert summary.index_built
        assert (output_dir / "hnsw_index.bin").exists()
        assert (output_dir / "hnsw_index.json").exists()

    def test_pipeline_output_structure(self, synthetic_genes_parquet: Path, tmp_path: Path):
        """Verify expected output files and their structure."""
        from elsa.analyze.micro_operon import run_operon_pipeline

        output_dir = tmp_path / "operon_out"
        summary = run_operon_pipeline(
            synthetic_genes_parquet,
            output_dir,
            pca_dims=16,
            shingle_k=3,
            min_genome_support=1,
        )

        # Check blocks CSV structure
        blocks_df = pd.read_csv(output_dir / "operon_blocks.csv")
        required_cols = ["block_id", "sample_id", "contig_id", "start_gene", "end_gene"]
        for col in required_cols:
            assert col in blocks_df.columns, f"Missing column: {col}"

        # Check summary JSON
        import json
        with open(output_dir / "operon_summary.json") as f:
            summary_data = json.load(f)
        assert "num_genes" in summary_data
        assert "num_blocks" in summary_data
        assert "index_built" in summary_data
