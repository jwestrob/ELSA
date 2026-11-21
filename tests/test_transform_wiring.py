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
from elsa.transform import EmbeddingTransform, fit_transform, mean_norm


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


# ---------------------------------------------------------------------------
# Query path: a search must enter the space its index lives in
# ---------------------------------------------------------------------------

def test_fitted_transform_is_persisted_for_later_queries(tmp_path):
    """A dataset-fitted transform that is never written cannot be reproduced.

    ``elsa search`` embeds a handful of query proteins long after ``elsa embed``
    ran. Refitting on those would estimate a different mean and covariance, so
    the fit has to survive on disk or the query lands in a different space.
    """
    from elsa.projection import ProjectionSystem
    from elsa.manifest import ELSAManifest

    config = PLMConfig(project_to_D=0, embedding_transform="soft_zca",
                       transform_alpha=0.1)
    system = ProjectionSystem(config, tmp_path, ELSAManifest(tmp_path))
    fitted = system._load_or_fit_transform(_anisotropic(n=200, dim=16))

    assert system.transform_model_path.is_file()
    reloaded = EmbeddingTransform.load(system.transform_model_path)
    probe = _anisotropic(n=5, dim=16, seed=7)
    np.testing.assert_allclose(reloaded.apply(probe), fitted.apply(probe), atol=1e-10)


def test_query_embeddings_enter_the_indexed_space(tmp_path):
    """The bug this closes: the query path ignored the transform entirely.

    Cosines between two spaces stay well-formed, so the failure is silent — the
    search returns ranked, confident, wrong neighbours.
    """
    from elsa.search import _apply_indexed_transform

    # One draw split into fit and query halves. Both must share a dominant
    # direction: a whitening transform is fitted to one corpus geometry, and a
    # query drawn around a *different* dominant direction comes back more
    # anisotropic, not less, because the fit suppresses a direction the query
    # does not have.
    everything = _anisotropic(n=500, dim=16)
    corpus, query = everything[:300], everything[300:]

    transform = fit_transform("soft_zca", corpus, alpha=0.1)
    transform.save(tmp_path / "ingest" / "embedding_transform.npz")

    config = PLMConfig(project_to_D=0, embedding_transform="soft_zca")
    treated = _apply_indexed_transform(query, config, tmp_path)

    np.testing.assert_allclose(treated, transform.apply(query), atol=1e-10)
    # and it genuinely moved: untreated queries would keep raw's anisotropy.
    # This needs a sample, not a handful — the mean of four unit vectors has a
    # large norm whatever the space looks like.
    assert mean_norm(treated) < mean_norm(query) / 2


def test_query_path_is_untouched_when_no_transform_is_configured(tmp_path):
    from elsa.search import _apply_indexed_transform

    query = _anisotropic(n=4, dim=16)
    out = _apply_indexed_transform(query, PLMConfig(project_to_D=0), tmp_path)
    assert out is query


def test_missing_transform_refuses_rather_than_searching_the_wrong_space(tmp_path):
    """An index built before the setting existed holds untreated vectors.

    Silently skipping the transform would compare across two spaces; silently
    fitting one on the query would invent a third.
    """
    from elsa.search import _apply_indexed_transform

    config = PLMConfig(project_to_D=0, embedding_transform="soft_zca")
    with pytest.raises(FileNotFoundError, match="re-run 'elsa embed'"):
        _apply_indexed_transform(_anisotropic(n=4, dim=16), config, tmp_path)


def test_dimension_mismatch_is_caught_at_the_boundary(tmp_path):
    from elsa.search import _apply_indexed_transform

    fit_transform("r1_center", _anisotropic(n=50, dim=16)).save(
        tmp_path / "ingest" / "embedding_transform.npz")
    config = PLMConfig(project_to_D=0, embedding_transform="r1_center")
    with pytest.raises(ValueError, match="expects 16-d"):
        _apply_indexed_transform(_anisotropic(n=4, dim=32), config, tmp_path)


def test_whitening_fitted_on_a_different_geometry_makes_things_worse():
    """Why the frozen-external arm is not free.

    Soft-ZCA suppresses the directions its fit corpus was spread along. Applied
    to embeddings whose spread lies elsewhere, it suppresses the wrong ones and
    leaves the query *more* anisotropic than raw. A frozen transform is only
    safe on a corpus resembling the one it was fitted on, so shipping one means
    measuring the gap rather than assuming it.
    """
    corpus = _anisotropic(n=300, dim=16, seed=0)
    other_geometry = _anisotropic(n=300, dim=16, seed=3)  # different dominant axis

    transform = fit_transform("soft_zca", corpus, alpha=0.1)

    assert mean_norm(transform.apply(corpus)) < 0.2          # in-corpus: works
    assert mean_norm(transform.apply(other_geometry)) > mean_norm(other_geometry)
