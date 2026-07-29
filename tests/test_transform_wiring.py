"""The transform must reach the vectors that get indexed, not just the report.

Before this wiring, ``frozen_pca_path`` was a silent no-op at the shipped
default: ``project_to_D`` is 0, which short-circuited the whole projection stage,
so a configured transform was never loaded. These tests pin the behaviour that
fix depends on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from elsa.params import PLMConfig
from elsa.transform import fit_transform, mean_norm


def _anisotropic(n=400, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=dim); shared /= np.linalg.norm(shared)
    matrix = 3.0 * shared + rng.normal(scale=0.5, size=(n, dim))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_transform_is_configurable_and_validated():
    assert PLMConfig().embedding_transform == "none"
    assert PLMConfig(embedding_transform="soft_zca").embedding_transform == "soft_zca"
    with pytest.raises(Exception):
        PLMConfig(embedding_transform="not_a_transform")


def test_transform_stays_active_when_dimensionality_reduction_is_off():
    """The blocker this fix exists for.

    ``project_to_D=0`` is the shipped default. A transform that preserves
    dimension must still apply, or configuring one does nothing and says nothing.
    """
    from elsa.projection import ProjectionSystem
    from elsa.manifest import ELSAManifest
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        config = PLMConfig(project_to_D=0, embedding_transform="r1_center")
        system = ProjectionSystem(config, Path(tmp), ELSAManifest(Path(tmp)))
        assert system.skip_pca, "fixture should exercise the default path"
        assert system.transform_kind == "r1_center", (
            "a dimension-preserving transform must remain reachable when "
            "project_to_D is 0"
        )


def test_no_transform_configured_means_no_transform_applied():
    from elsa.projection import ProjectionSystem
    from elsa.manifest import ELSAManifest
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        config = PLMConfig(project_to_D=0)
        system = ProjectionSystem(config, Path(tmp), ELSAManifest(Path(tmp)))
        assert system.transform_kind == "none"
        assert system._load_or_fit_transform(_anisotropic()) is None


def test_frozen_transform_is_loaded_rather_than_refitted(tmp_path):
    """A frozen transform is the deployable form; refitting per dataset would
    put every run in a slightly different space."""
    from elsa.projection import ProjectionSystem
    from elsa.manifest import ELSAManifest

    matrix = _anisotropic()
    fitted = fit_transform("r1_center", matrix, sample=400)
    path = fitted.save(tmp_path / "frozen.npz")

    config = PLMConfig(project_to_D=0, embedding_transform="r1_center",
                       transform_path=str(path))
    system = ProjectionSystem(config, tmp_path, ELSAManifest(tmp_path))
    loaded = system._load_or_fit_transform(matrix)
    np.testing.assert_allclose(loaded.state["mean"], fitted.state["mean"])


def test_missing_frozen_transform_fails_loudly(tmp_path):
    """Falling back to a per-dataset fit would silently produce a different
    space than the one the run claims to use."""
    from elsa.projection import ProjectionSystem
    from elsa.manifest import ELSAManifest

    config = PLMConfig(project_to_D=0, embedding_transform="r1_center",
                       transform_path=str(tmp_path / "absent.npz"))
    system = ProjectionSystem(config, tmp_path, ELSAManifest(tmp_path))
    with pytest.raises(FileNotFoundError):
        system._load_or_fit_transform(_anisotropic())


def test_store_records_the_embedding_space(tmp_path):
    """Two stores can share a dimension and hold incomparable vectors. Querying
    one with the other's vectors returns confident nonsense, so the space has to
    be on the record."""
    from elsa.store import SyntenyStore

    matrix = _anisotropic(n=60, dim=16)
    genes = pd.DataFrame({
        "sample_id": ["g1"] * 30 + ["g2"] * 30,
        "contig_id": ["c1"] * 60,
        "gene_id": [f"p{i}" for i in range(60)],
        "start": np.arange(60) * 100,
        "end": np.arange(60) * 100 + 90,
        "strand": 1,
        **{f"emb_{j}": matrix[:, j] for j in range(16)},
    })
    store_dir = tmp_path / "store"
    SyntenyStore.create(store_dir, genes, embedding_transform="soft_zca(alpha=0.1)")

    config = json.loads((store_dir / "config.json").read_text())
    assert config["embedding_transform"] == "soft_zca(alpha=0.1)"
    assert config["schema_version"] == 2
    # the fingerprint must reflect the vectors actually stored
    assert abs(config["embedding_mean_norm"] - mean_norm(matrix)) < 0.02


def test_store_transform_defaults_to_null_for_untreated_vectors(tmp_path):
    from elsa.store import SyntenyStore

    matrix = _anisotropic(n=40, dim=8)
    genes = pd.DataFrame({
        "sample_id": ["g1"] * 20 + ["g2"] * 20,
        "contig_id": ["c1"] * 40,
        "gene_id": [f"p{i}" for i in range(40)],
        "start": np.arange(40) * 100,
        "end": np.arange(40) * 100 + 90,
        "strand": 1,
        **{f"emb_{j}": matrix[:, j] for j in range(8)},
    })
    SyntenyStore.create(tmp_path / "s", genes)
    config = json.loads((tmp_path / "s" / "config.json").read_text())
    assert config["embedding_transform"] is None
