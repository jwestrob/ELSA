"""The embedding-transform ladder used for the anisotropy ablation."""

from __future__ import annotations

import numpy as np
import pytest

from elsa.transform import (
    LADDER,
    EmbeddingTransform,
    fit_ladder,
    fit_transform,
    isotropy_score,
    l2_normalize,
    mean_norm,
)


@pytest.fixture()
def anisotropic():
    """A corpus with a strong shared direction, like a real PLM embedding space."""
    rng = np.random.default_rng(0)
    dim, n = 48, 3000
    shared = rng.normal(size=dim)
    shared /= np.linalg.norm(shared)
    matrix = 3.0 * shared + rng.normal(scale=0.5, size=(n, dim))
    return l2_normalize(matrix)


def test_fixture_is_actually_anisotropic(anisotropic):
    assert mean_norm(anisotropic) > 0.4


def test_every_rung_fits_and_preserves_shape(anisotropic):
    for name, transform in fit_ladder(anisotropic, sample=3000).items():
        out = transform.apply(anisotropic)
        assert out.shape[0] == anisotropic.shape[0], name
        assert np.isfinite(out).all(), name


def test_mean_removing_rungs_actually_remove_the_mean(anisotropic):
    fitted = fit_ladder(anisotropic, sample=3000)
    for name in ("r1_center", "r2_project", "abtt_k1", "abtt_k3"):
        assert mean_norm(fitted[name].apply(anisotropic)) < 0.05, name


def test_random_direction_control_leaves_anisotropy_intact(anisotropic):
    """The control must NOT behave like the mean-removal arms.

    If projecting out an arbitrary direction reduced anisotropy as much as
    projecting out the mean, an improvement in the treatment arms could not be
    attributed to anisotropy at all.
    """
    fitted = fit_ladder(anisotropic, sample=3000)
    before = mean_norm(anisotropic)
    control = mean_norm(fitted["random_direction"].apply(anisotropic))
    treated = mean_norm(fitted["r2_project"].apply(anisotropic))
    assert control > 0.8 * before, "control removed too much anisotropy to be a control"
    assert treated < 0.05
    assert control > 10 * treated


def test_whitening_is_more_isotropic_than_single_direction_removal(anisotropic):
    """Dose ordering: the ladder must actually be a ladder."""
    fitted = fit_ladder(anisotropic, sample=3000)
    single = isotropy_score(fitted["r1_center"].apply(anisotropic))
    whitened = isotropy_score(fitted["pca_whiten"].apply(anisotropic))
    assert whitened > single


def test_soft_zca_alpha_interpolates_toward_whitening(anisotropic):
    strong = fit_transform("soft_zca", anisotropic, alpha=0.01, sample=3000)
    weak = fit_transform("soft_zca", anisotropic, alpha=0.5, sample=3000)
    assert (isotropy_score(strong.apply(anisotropic))
            > isotropy_score(weak.apply(anisotropic)))


def test_output_is_unit_norm_when_renormalizing(anisotropic):
    transform = fit_transform("r1_center", anisotropic, sample=3000)
    norms = np.linalg.norm(transform.apply(anisotropic), axis=1)
    assert np.allclose(norms, 1.0)


def test_transform_round_trips_through_disk(tmp_path, anisotropic):
    """A treatment must be reproducible from a file, or a run cannot record which
    space produced it."""
    for name, transform in fit_ladder(anisotropic, sample=3000).items():
        path = transform.save(tmp_path / f"{name}.npz")
        restored = EmbeddingTransform.load(path)
        assert restored.kind == transform.kind, name
        assert restored.params == transform.params, name
        np.testing.assert_allclose(
            restored.apply(anisotropic), transform.apply(anisotropic),
            rtol=1e-10, atol=1e-10, err_msg=name,
        )


def test_fitting_corpus_and_application_corpus_can_differ(anisotropic):
    """Fitting on held-out data is the whole point of the leakage protocol."""
    fit_half, apply_half = anisotropic[:1500], anisotropic[1500:]
    transform = fit_transform("r2_project", fit_half, sample=1500)
    out = transform.apply(apply_half)
    assert out.shape == apply_half.shape
    assert mean_norm(out) < mean_norm(apply_half)


def test_raw_is_identity_up_to_normalization(anisotropic):
    transform = fit_transform("raw", anisotropic, sample=3000)
    np.testing.assert_allclose(transform.apply(anisotropic), anisotropic,
                               rtol=1e-10, atol=1e-10)


def test_memory_mapped_input_is_accepted(tmp_path, anisotropic):
    """The DPANN store is 4.3 GB on disk; fitting must not require loading it."""
    path = tmp_path / "emb.npy"
    np.save(path, anisotropic.astype(np.float32))
    memmapped = np.load(path, mmap_mode="r")
    transform = fit_transform("r1_center", memmapped, sample=500)
    assert np.isfinite(transform.apply(memmapped)).all()


def test_unknown_kind_is_rejected(anisotropic):
    with pytest.raises(ValueError, match="unknown transform kind"):
        fit_transform("bogus", anisotropic, sample=100)
    with pytest.raises(ValueError, match="unknown transform kind"):
        EmbeddingTransform("bogus", {}).apply(anisotropic)


def test_ladder_names_are_stable_identifiers():
    """Result tables key on these; renaming one silently invalidates comparisons."""
    assert set(LADDER) == {
        "raw", "r1_center", "r2_project", "random_direction",
        "abtt_k1", "abtt_k2", "abtt_k3",
        "soft_zca_a50", "soft_zca_a10", "soft_zca_a01", "pca_whiten",
    }
