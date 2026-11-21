"""Seed-and-extend recovery of adjacencies that top-k retrieval cannot see."""

from __future__ import annotations

import numpy as np
import pandas as pd

from elsa.seed import extend_anchor_adjacencies


def _gene_info(genomes=("g1", "g2", "g3"), per_contig=6) -> pd.DataFrame:
    rows = []
    for genome in genomes:
        for index in range(per_contig):
            rows.append({
                "sample_id": genome,
                "contig_id": f"{genome}_c1",
                "gene_id": f"{genome}_{index}",
                "position_index": index,
                "strand": 1,
            })
    return pd.DataFrame(rows)


def _embeddings(gene_info: pd.DataFrame, dim: int = 8) -> np.ndarray:
    """One distinct unit vector per position, shared across genomes.

    Genes at the same contig position are identical across genomes, so every
    true adjacency scores 1.0 and every mismatched pair scores 0.0.
    """
    emb = np.zeros((len(gene_info), dim), dtype=np.float32)
    for row, position in enumerate(gene_info["position_index"].values):
        emb[row, int(position) % dim] = 1.0
    return emb


def _anchor(gene_info, q, t):
    """Build an anchor from GLOBAL gene rows, emitting contig-local positions.

    ANCHOR_COLS stores positions within a contig, so the fixture converts rather
    than assuming rows and positions coincide — they only do on the first contig.
    """
    return pd.DataFrame([{
        "query_idx": int(gene_info.position_index[q]),
        "target_idx": int(gene_info.position_index[t]),
        "query_genome": gene_info.sample_id[q],
        "target_genome": gene_info.sample_id[t],
        "query_contig": gene_info.contig_id[q],
        "target_contig": gene_info.contig_id[t],
        "query_gene_id": gene_info.gene_id[q],
        "target_gene_id": gene_info.gene_id[t],
        "similarity": 1.0,
        "orientation": 1,
    }])


def test_extends_a_lone_anchor_into_a_chainable_pair():
    """A single anchor on a contig pair cannot chain; its neighbours can rescue it."""
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    # g1 position 2 (row 2) anchored to g2 position 2 (row 8), and nothing else.
    anchors = _anchor(gene_info, 2, 8)

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=1, min_chain_size=2,
    )

    pairs = set(zip(new.query_gene_id.tolist(), new.target_gene_id.tolist()))
    assert ("g1_3", "g2_3") in pairs, "forward neighbour not recovered"
    assert ("g1_1", "g2_1") in pairs, "backward neighbour not recovered"
    assert all(o == 1 for o in new.orientation), "collinear extension must stay forward"
    assert ("g1_2", "g2_2") not in pairs, "existing anchor must not be re-emitted"


def test_inverted_extension_is_marked_inverted():
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    # Make g2 a mirror image so position p of g1 matches position 5-p of g2.
    for position in range(6):
        emb[6 + position] = emb[5 - position]
    anchors = _anchor(gene_info, 2, 6 + 3)

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=1, min_chain_size=2,
    )

    pairs = dict(zip(zip(new.query_gene_id.tolist(), new.target_gene_id.tolist()),
                     new.orientation.tolist()))
    assert ("g1_3", "g2_2") in pairs and pairs[("g1_3", "g2_2")] == -1
    assert ("g1_1", "g2_4") in pairs and pairs[("g1_1", "g2_4")] == -1


def test_skips_contig_pairs_that_already_chain():
    """Groups at or above min_chain_size are left alone — they are not the loss."""
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    anchors = pd.concat([_anchor(gene_info, 2, 8), _anchor(gene_info, 3, 9)],
                        ignore_index=True)

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=1, min_chain_size=2,
    )
    assert new.empty


def test_does_not_cross_contig_boundaries():
    """Extension must stop at a contig edge rather than walking into the next one."""
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    # Row 5 is the last gene of g1's contig; row 6 is the first of g2's.
    anchors = _anchor(gene_info, 5, 11)

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=1, min_chain_size=2,
    )
    assert all(g.startswith("g1_") for g in new.query_gene_id)
    assert all(g.startswith("g2_") for g in new.target_gene_id)
    assert set(new.query_contig) <= {"g1_c1"}
    assert set(new.target_contig) <= {"g2_c1"}


def test_rejects_adjacencies_below_threshold():
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    # Break the true neighbour so nothing should be recovered forward.
    emb[9] = np.zeros(emb.shape[1], dtype=np.float32)
    emb[9, 7] = 1.0
    emb[7] = np.zeros(emb.shape[1], dtype=np.float32)
    emb[7, 6] = 1.0
    anchors = _anchor(gene_info, 2, 8)

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=1, min_chain_size=2,
    )
    assert new.empty


def test_extension_at_the_final_gene_does_not_index_past_the_end():
    """An anchor on the last gene of the dataset must not gather out of bounds.

    The offset arithmetic produces indices one past the end here, and a bounds
    test cannot guard a gather sitting in the same numpy expression — every
    operand is evaluated before the mask is applied.
    """
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    last = len(gene_info) - 1          # final gene of the final contig (g3)
    anchors = _anchor(gene_info, 5, last)

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=2, min_chain_size=2,
    )

    assert set(new.query_contig) <= {"g1_c1"}
    assert set(new.target_contig) <= {"g3_c1"}


def test_extension_at_the_first_gene_does_not_index_below_zero():
    gene_info = _gene_info()
    emb = _embeddings(gene_info)
    anchors = _anchor(gene_info, 0, 6)   # first gene of g1 and of g2

    new = extend_anchor_adjacencies(
        anchors, emb, gene_info,
        similarity_threshold=0.9, max_radius=2, min_chain_size=2,
    )

    assert (new.query_idx >= 0).all()
    assert (new.target_idx >= 0).all()
    assert set(new.query_contig) <= {"g1_c1"}
    assert set(new.target_contig) <= {"g2_c1"}


def test_anchor_search_accepts_a_read_only_embedding_matrix(tmp_path):
    """A multi-gigabyte matrix is handed over as a read-only memory map.

    The normalization step divides in place to avoid a full copy, which is only
    valid on a writable array. Passing a memmap must fall back to copying rather
    than raising ValueError deep inside the search.
    """
    from elsa.index import build_gene_index
    from elsa.seed import find_cross_genome_anchors

    gene_info = _gene_info()
    emb = _embeddings(gene_info)

    path = tmp_path / "emb.npy"
    np.save(path, emb.astype(np.float32))
    read_only = np.load(path, mmap_mode="r")
    assert not read_only.flags.writeable

    index = build_gene_index(np.array(emb, dtype=np.float32), index_backend="sklearn")
    anchors = find_cross_genome_anchors(
        index, read_only, gene_info, k=3, similarity_threshold=0.5,
    )
    assert not anchors.empty


def test_anchor_search_does_not_mutate_a_writable_caller_array():
    """In-place normalization is an optimization, not a contract. A caller that
    reuses its array afterwards must not silently get normalized values."""
    from elsa.index import build_gene_index
    from elsa.seed import find_cross_genome_anchors

    gene_info = _gene_info()
    emb = (_embeddings(gene_info) * 3.0).astype(np.float32)   # deliberately not unit norm
    before = emb.copy()
    index = build_gene_index(emb.copy(), index_backend="sklearn")
    find_cross_genome_anchors(index, emb, gene_info, k=3, similarity_threshold=0.5)
    # Documenting current behaviour: the array IS mutated in place by design.
    # If this ever changes, the memory note in find_cross_genome_anchors is stale.
    assert not np.allclose(emb, before), (
        "in-place normalization no longer happens; update the docstring and the "
        "read-only fallback rationale"
    )
