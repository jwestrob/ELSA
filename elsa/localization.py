"""How tightly does a block localize what it contains?

Recall answers "is the locus inside some block". On its own that is a weak
question, because a block spanning an entire contig answers it perfectly. The
July 2026 benchmark audit measured the gap directly: at 98.9% strict recall, the
median boundary precision is 0.0403 and the median smallest fully-containing
block overruns its locus by 174 genes. Both numbers describe the same runs. Only
the first was ever reported.

This module exists so the pair cannot be separated again. :func:`localize_pairs`
returns coverage and precision from one computation over one block selection, and
:func:`summarize` refuses to emit a recall figure without the precision that
qualifies it.

Definitions follow ``benchmarks/scripts/audit_canonical_benchmarks.py`` exactly,
so values here are directly comparable to the audit rather than merely similar:

* **coverage** - overlap / locus span. The recall side: how much of the locus the
  block covers.
* **boundary precision** - overlap / block span. How much of the block is the
  locus. A block ten times the locus scores 0.1 however perfectly it covers.
* **IoU** - overlap / union. One number that falls if either side is wrong.
* **pair IoU** - geometric mean of the two sides, so a block that localizes well
  in one genome and sprawls in the other cannot average its way to a good score.
* **containing excess** - genes by which the smallest fully-containing block
  overruns the locus, summed over both sides. Reported in genes because that is
  what a reader has to scroll past to find the thing they searched for.

Nothing here needs ELSA internals: it is interval geometry over
(start, end) pairs, so the same functions serve the operon benchmark, a cluster
self-consistency check, and any future ground truth.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "inclusive_overlap",
    "localize_pairs",
    "summarize",
    "format_summary",
    "block_localization",
]


def inclusive_overlap(start_a, end_a, start_b, end_b):
    """Overlap of two inclusive integer intervals, floored at zero.

    Positions are gene indices, so an interval [4, 4] has span 1, not 0.
    """
    return np.maximum(
        0,
        np.minimum(end_a, end_b) - np.maximum(start_a, start_b) + 1,
    )


def localize_pairs(
    loci: pd.DataFrame,
    blocks: pd.DataFrame,
    *,
    min_coverage: float = 0.5,
    locus_cols: Sequence[str] = ("genome_a", "contig_a", "start_a", "end_a",
                                 "genome_b", "contig_b", "start_b", "end_b"),
    block_cols: Sequence[str] = ("query_genome", "query_contig", "query_start",
                                 "query_end", "target_genome", "target_contig",
                                 "target_start", "target_end"),
) -> pd.DataFrame:
    """Score every locus against the block that best localizes it.

    A locus is a two-sided interval: the same region observed in two genomes.
    A block is scored only when it covers the locus on *both* sides, matching
    the "strict" convention the canonical numbers use — otherwise a block that
    sprawls across one genome and misses the other would count as a hit.

    Block selection prefers coverage first, then tightness (pair IoU), then the
    smaller block, then the lower id. Ties are broken deterministically so the
    result does not depend on row order.

    Returns one row per locus. Loci with no qualifying block are returned with
    ``hit=False`` and zeroed geometry, so recall and precision are computed over
    the same denominator and a caller cannot accidentally average precision over
    hits alone while reporting recall over everything.
    """
    ga, ca, sa, ea, gb, cb, sb, eb = locus_cols
    bga, bca, bsa, bea, bgb, bcb, bsb, beb = block_cols

    if "block_id" in blocks.columns:
        block_ids = blocks["block_id"].to_numpy()
    else:
        block_ids = np.arange(len(blocks))

    # Index blocks by the genome/contig pair they connect. Loci are matched
    # within a pair, so this replaces a scan of every block per locus.
    side_a = list(zip(blocks[bga].astype(str), blocks[bca].astype(str)))
    side_b = list(zip(blocks[bgb].astype(str), blocks[bcb].astype(str)))
    by_pair: Dict[Tuple[Tuple[str, str], Tuple[str, str]], List[int]] = {}
    for row, (key_a, key_b) in enumerate(zip(side_a, side_b)):
        by_pair.setdefault((key_a, key_b), []).append(row)
        if key_a != key_b:
            # A block is undirected: it localizes the locus whichever way round
            # the two genomes were named.
            by_pair.setdefault((key_b, key_a), []).append(row)

    b_start_a = blocks[bsa].to_numpy(); b_end_a = blocks[bea].to_numpy()
    b_start_b = blocks[bsb].to_numpy(); b_end_b = blocks[beb].to_numpy()
    b_genome_a = blocks[bga].astype(str).to_numpy()

    records: List[Dict[str, object]] = []
    for locus in loci.itertuples(index=False):
        key_a = (str(getattr(locus, ga)), str(getattr(locus, ca)))
        key_b = (str(getattr(locus, gb)), str(getattr(locus, cb)))
        locus_start_a, locus_end_a = getattr(locus, sa), getattr(locus, ea)
        locus_start_b, locus_end_b = getattr(locus, sb), getattr(locus, eb)
        span_a = locus_end_a - locus_start_a + 1
        span_b = locus_end_b - locus_start_b + 1

        rows = np.asarray(by_pair.get((key_a, key_b), ()), dtype=int)
        empty = {
            "hit": False, "coverage_a": 0.0, "coverage_b": 0.0,
            "precision_a": 0.0, "precision_b": 0.0, "pair_iou": 0.0,
            "block_span_a": 0, "block_span_b": 0,
            "containing_excess": np.nan, "block_id": -1,
            "locus_span_a": int(span_a), "locus_span_b": int(span_b),
        }
        if rows.size == 0:
            records.append(empty)
            continue

        # Orient each candidate so side A of the block matches side A of the
        # locus; a block stored the other way round is the same block.
        flip = b_genome_a[rows] != key_a[0]
        start_a = np.where(flip, b_start_b[rows], b_start_a[rows])
        end_a = np.where(flip, b_end_b[rows], b_end_a[rows])
        start_b = np.where(flip, b_start_a[rows], b_start_b[rows])
        end_b = np.where(flip, b_end_a[rows], b_end_b[rows])

        block_span_a = end_a - start_a + 1
        block_span_b = end_b - start_b + 1
        overlap_a = inclusive_overlap(start_a, end_a, locus_start_a, locus_end_a)
        overlap_b = inclusive_overlap(start_b, end_b, locus_start_b, locus_end_b)

        coverage_a = overlap_a / span_a
        coverage_b = overlap_b / span_b
        precision_a = overlap_a / block_span_a
        precision_b = overlap_b / block_span_b
        iou_a = overlap_a / (span_a + block_span_a - overlap_a)
        iou_b = overlap_b / (span_b + block_span_b - overlap_b)
        pair_iou = np.sqrt(iou_a * iou_b)
        minimum_coverage = np.minimum(coverage_a, coverage_b)

        qualifies = minimum_coverage >= min_coverage
        if not qualifies.any():
            records.append(empty)
            continue

        combined_span = block_span_a + block_span_b
        candidates = np.flatnonzero(qualifies)
        order = np.lexsort((
            block_ids[rows][candidates],
            combined_span[candidates],
            -pair_iou[candidates],
            -minimum_coverage[candidates],
        ))
        best = int(candidates[order[0]])

        containing = np.flatnonzero((coverage_a >= 1.0) & (coverage_b >= 1.0))
        if containing.size:
            local = np.lexsort((block_ids[rows][containing],
                                combined_span[containing]))
            smallest = int(containing[local[0]])
            excess = float(block_span_a[smallest] + block_span_b[smallest]
                           - span_a - span_b)
        else:
            excess = np.nan

        records.append({
            "hit": True,
            "coverage_a": float(coverage_a[best]),
            "coverage_b": float(coverage_b[best]),
            "precision_a": float(precision_a[best]),
            "precision_b": float(precision_b[best]),
            "pair_iou": float(pair_iou[best]),
            "block_span_a": int(block_span_a[best]),
            "block_span_b": int(block_span_b[best]),
            "containing_excess": excess,
            "block_id": int(block_ids[rows][best]),
            "locus_span_a": int(span_a),
            "locus_span_b": int(span_b),
        })

    frame = pd.DataFrame.from_records(records)
    for column in loci.columns:
        if column not in frame.columns:
            frame[column] = loci[column].to_numpy()
    return frame


def _bootstrap_median(values: np.ndarray, groups: Optional[np.ndarray],
                      n_boot: int, seed: int) -> Tuple[float, float]:
    """Percentile interval for a median, resampling groups rather than rows.

    Operon pair-rows are not independent — the same operon contributes many
    rows across genome pairs — so resampling rows would report an interval far
    tighter than the evidence supports.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = [np.median(rng.choice(values, values.size, replace=True))
                 for _ in range(n_boot)]
    else:
        groups = groups[: len(values)]
        unique = np.unique(groups)
        if unique.size < 2:
            # Between-group variance is not estimable from one group, and
            # resampling it returns a zero-width interval that reads as
            # certainty. Say "unknown" instead.
            return (float("nan"), float("nan"))
        index: Dict[object, np.ndarray] = {
            key: np.flatnonzero(groups == key) for key in unique
        }
        draws = []
        for _ in range(n_boot):
            picked = rng.choice(unique, unique.size, replace=True)
            rows = np.concatenate([index[key] for key in picked])
            draws.append(np.median(values[rows]))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def summarize(
    frame: pd.DataFrame,
    *,
    group_column: Optional[str] = None,
    n_boot: int = 1000,
    seed: int = 17,
) -> Dict[str, float]:
    """Recall and localization from one frame, as one indivisible result.

    ``recall`` is the fraction of loci with a qualifying block. Every precision
    statistic is computed over the hits, because localization is undefined where
    nothing was found — which is exactly why the two must be read together and
    why this returns them in a single dict rather than leaving a caller to fetch
    them separately.
    """
    total = len(frame)
    hits = frame[frame["hit"]] if total else frame
    groups = (hits[group_column].to_numpy()
              if group_column and group_column in hits.columns else None)

    precision = np.r_[hits["precision_a"].to_numpy(),
                      hits["precision_b"].to_numpy()] if len(hits) else np.array([])
    excess = hits["containing_excess"].to_numpy() if len(hits) else np.array([])
    paired_groups = np.r_[groups, groups] if groups is not None else None

    low, high = _bootstrap_median(precision, paired_groups, n_boot, seed)
    return {
        "n_loci": int(total),
        "n_hits": int(len(hits)),
        "recall": float(len(hits) / total) if total else float("nan"),
        "median_boundary_precision": float(np.median(precision)) if precision.size else float("nan"),
        "boundary_precision_ci_low": low,
        "boundary_precision_ci_high": high,
        "median_pair_iou": float(hits["pair_iou"].median()) if len(hits) else float("nan"),
        "median_containing_excess_genes": float(np.nanmedian(excess)) if excess.size else float("nan"),
        "median_block_span": float(np.median(np.r_[
            hits["block_span_a"].to_numpy(), hits["block_span_b"].to_numpy()
        ])) if len(hits) else float("nan"),
        "median_locus_span": float(np.median(np.r_[
            hits["locus_span_a"].to_numpy(), hits["locus_span_b"].to_numpy()
        ])) if len(hits) else float("nan"),
    }


def format_summary(summary: Dict[str, float], label: str = "") -> str:
    """Render recall and localization together, never one without the other."""
    head = f"{label}\n" if label else ""
    recall = summary["recall"]
    precision = summary["median_boundary_precision"]
    excess = summary["median_containing_excess_genes"]
    ratio = (summary["median_block_span"] / summary["median_locus_span"]
             if summary["median_locus_span"] else float("nan"))
    return (
        f"{head}"
        f"  recall                    {recall:.4f}  ({summary['n_hits']:,}/{summary['n_loci']:,})\n"
        f"  median boundary precision {precision:.4f}  "
        f"95% CI [{summary['boundary_precision_ci_low']:.4f}, "
        f"{summary['boundary_precision_ci_high']:.4f}]\n"
        f"  median pair IoU           {summary['median_pair_iou']:.4f}\n"
        f"  median block / locus span {summary['median_block_span']:.0f} / "
        f"{summary['median_locus_span']:.0f} genes  ({ratio:.1f}x)\n"
        f"  median containing excess  {excess:.0f} genes\n"
        f"  -> {recall:.1%} of loci sit inside some block; that block is "
        f"{precision:.1%} locus by extent."
    )


def block_localization(blocks: pd.DataFrame) -> pd.DataFrame:
    """Per-block tightness that needs no ground truth.

    Anchor density is the ground-truth-free analogue of boundary precision: a
    block whose anchors are sparse over a long span is padding out a chain, and
    that is visible without knowing what the block was supposed to find. Useful
    on any dataset, which the operon benchmark is not.
    """
    span_a = blocks["query_end"] - blocks["query_start"] + 1
    span_b = blocks["target_end"] - blocks["target_start"] + 1
    span = np.maximum(span_a, span_b)
    anchors = (blocks["n_anchors"] if "n_anchors" in blocks.columns
               else pd.Series(np.nan, index=blocks.index))
    return pd.DataFrame({
        "block_id": blocks["block_id"] if "block_id" in blocks.columns
                    else np.arange(len(blocks)),
        "span": span,
        "n_anchors": anchors,
        "anchor_density": anchors / span.replace(0, np.nan),
    })
