"""Bounded, one-hop neighborhood support for directional protein candidates.

Original directional seeds supply all witnesses. Newly proposed pairs never
generate further proposals. A query keeps at most ``k`` targets across the
entire reference, ranked by support, best two-anchor cosine sum, focal cosine,
and physical target row. Witness multiplicity contributes no additional score.

Inputs retain complete physical row identities, including unembedded proteins.
Metadata positions determine distance; missing positions are never compressed.
This selection precedes reciprocal-pair deduplication in discovery workflows.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
import os

import numpy as np
from threadpoolctl import threadpool_limits

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        def decorate(function):
            return function
        return decorate


@dataclass
class NeighborhoodSelection:
    """Numeric selected pairs; evidence arrays are optional to bound storage."""

    query_rows: np.ndarray
    target_rows: np.ndarray
    similarities: np.ndarray
    counts: dict
    is_original: np.ndarray | None = None
    witness_query_rows: np.ndarray | None = None
    witness_target_rows: np.ndarray | None = None
    context_scores: np.ndarray | None = None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _integer_rows(values, size, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f'{name} must be a one-dimensional integer array')
    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(f'{name} contains an invalid physical row')
    return values


def _metadata(metadata, count):
    """Accept array mappings and column-oriented tables in physical row order."""
    result = {}
    for key in ['sample_id', 'contig_id', 'position_index']:
        values = np.asarray(metadata[key])
        if values.ndim != 1 or len(values) != count:
            raise ValueError(f'Metadata {key} must match physical embedding rows')
        result[key] = values
    if 'row_index' in metadata and not np.array_equal(metadata['row_index'], np.arange(count)):
        raise ValueError('Metadata row_index must preserve physical embedding rows')
    positions = result['position_index']
    if not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0):
        raise ValueError('Metadata positions must be nonnegative integers')
    for key in ['sample_id', 'contig_id']:
        values = result[key]
        # String identifiers or numeric codes are accepted; missing identifiers
        # cannot safely establish scaffold boundaries or cross-sample matches.
        if any(v is None or v != v for v in values):
            raise ValueError(f'Missing metadata {key}')
    strand = np.asarray(metadata['strand']) if 'strand' in metadata else np.zeros(count, dtype=np.int8)
    if strand.ndim != 1 or len(strand) != count or not np.isin(strand, [-1, 0, 1]).all():
        raise ValueError('Strands must be -1, 0 (unknown), or 1')
    result['strand'] = strand.astype(np.int8, copy=False)
    return result


def _neighbors(metadata, sample_codes, radius):
    """Physical neighbors at offsets [-radius..-1, 1..radius]."""
    count = len(sample_codes)
    dtype = np.int32 if count < 2**31 else np.int64
    result = np.full((count, 2 * radius), -1, dtype=dtype)
    if not count:
        return result
    _, contigs = np.unique(metadata['contig_id'], return_inverse=True)
    position = metadata['position_index']
    order = np.lexsort((position, contigs, sample_codes))
    sample, contig, pos = sample_codes[order], contigs[order], position[order]
    same = (sample[1:] == sample[:-1]) & (contig[1:] == contig[:-1])
    if np.any(same & (pos[1:] == pos[:-1])):
        raise ValueError('Duplicate position within a sample/contig')
    # A true coordinate neighbor within radius has at most radius intervening
    # stored rows, even when a supplied locus has gaps in physical positions.
    for distance in range(1, min(radius + 1, count)):
        delta = pos[distance:] - pos[:-distance]
        valid = ((sample[distance:] == sample[:-distance]) &
                 (contig[distance:] == contig[:-distance]) & (delta > 0) & (delta <= radius))
        left, right = order[:-distance][valid], order[distance:][valid]
        step = delta[valid].astype(np.intp)
        result[left, radius + step - 1] = right
        result[right, radius - step] = left
    return result


def _eligible_rows(vectors, normalize=False, chunk_size=4096):
    result = np.empty(len(vectors), dtype=bool)
    for start in range(0, len(vectors), chunk_size):
        values = np.asarray(vectors[start:start + chunk_size])
        good = np.isfinite(values).all(axis=1) & np.any(values != 0, axis=1)
        if normalize:
            values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
            with np.errstate(over='ignore', invalid='ignore'):
                norm = np.linalg.norm(values, axis=1)
            good &= np.isfinite(norm) & (norm > 0)
        result[start:start + len(values)] = good
    return result


def _seed_csr(query, target, score, nq, nr, qsample, rsample,
              qeligible, religible, threshold, k):
    query = _integer_rows(query, nq, 'seed_query_rows')
    target = _integer_rows(target, nr, 'seed_target_rows')
    score = np.asarray(score)
    if score.ndim != 1 or len(query) != len(target) or len(query) != len(score):
        raise ValueError('Directional seed arrays must align')
    if not np.issubdtype(score.dtype, np.floating) or not np.isfinite(score).all():
        raise ValueError('Original seed similarities must be finite floating values')
    score = score.astype(np.result_type(score.dtype, np.float32), copy=False)
    if np.any(score < threshold):
        raise ValueError('Original seeds must satisfy the configured cosine cutoff')
    order = np.lexsort((target, query))
    query, target, score = query[order], target[order], score[order]
    repeated = (query[1:] == query[:-1]) & (target[1:] == target[:-1])
    if np.any(repeated & (score[1:] != score[:-1])):
        raise ValueError('Repeated original pair has conflicting similarities')
    unique = np.r_[True, ~repeated] if len(query) else np.empty(0, dtype=bool)
    unique_count = int(unique.sum())
    keep = unique & qeligible[query] & religible[target] & (qsample[query] != rsample[target])
    query, target, score = query[keep], target[keep], score[keep]
    counts = np.bincount(query.astype(np.int64), minlength=nq)
    if np.any(counts > k):
        raise ValueError('Original directional seeds exceed the per-query budget k')
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    return offsets, target, score, unique_count


@njit(cache=True, nogil=True)
def _proposals(query, offsets, targets, scores, qneighbors, rneighbors,
               qstrand, rstrand, qsample, rsample, radius, k):
    capacity = 2 * radius * radius * k if qstrand[query] != 0 else 0
    proposed = np.empty(capacity, dtype=np.int64)
    witness_q = np.empty(capacity, dtype=np.int64)
    witness_t = np.empty(capacity, dtype=np.int64)
    witness_s = np.empty(capacity, dtype=np.float64)
    count, occurrences = 0, 0
    if qstrand[query] == 0:
        return proposed[:0], witness_q[:0], witness_t[:0], witness_s[:0], occurrences
    for column in range(2 * radius):
        donor = qneighbors[query, column]
        if donor < 0 or qstrand[donor] == 0:
            continue
        query_step = column - radius if column < radius else column - radius + 1
        for seed in range(offsets[donor], offsets[donor + 1]):
            source = targets[seed]
            if rstrand[source] == 0:
                continue
            orientation = qstrand[donor] * rstrand[source]
            forward = (query_step < 0 and orientation > 0) or (query_step > 0 and orientation < 0)
            first = radius if forward else 0
            for target_column in range(first, first + radius):
                target = rneighbors[source, target_column]
                if target < 0:
                    continue
                occurrences += 1
                if (rstrand[target] == 0 or qstrand[query] * rstrand[target] != orientation
                        or rsample[target] == qsample[query]):
                    continue
                proposed[count] = target
                witness_q[count], witness_t[count] = donor, source
                witness_s[count] = scores[seed]
                count += 1
    return proposed[:count], witness_q[:count], witness_t[:count], witness_s[:count], occurrences


def select_neighborhood_candidates(
    query_vectors, reference_vectors, seed_query_rows, seed_target_rows, seed_scores, *,
    query_metadata, reference_metadata, k=60, threshold=.85, max_gap=2,
    query_rows=None, batch_size=256, n_jobs=None, normalize_reference=False,
    include_evidence=False,
):
    """Select bounded directional candidates using original-neighbor evidence.

    ``query_vectors`` is an L2-normalized floating matrix. Reference vectors
    share its representation and normalization; ``normalize_reference=True``
    instead normalizes just the gathered rows used for new comparisons, using
    ``norm + 1e-9``. Original seed scores remain unchanged. Nonfinite/zero
    vectors are ineligible. Strand0 retains original matches and supplies no
    neighborhood proposals.

    Metadata is a mapping/table of ``sample_id``, ``contig_id``,
    ``position_index`` and optional ``strand`` in physical vector-row order.
    Query and reference sample identifiers share the same namespace. Optional
    ``query_rows`` selects focal output rows; witnesses use any original seeds
    supplied for neighboring query rows. Only supplied query context is used.

    For radius r=max_gap+1, at most 2*r*r*k proposals and (1+2*r*r)*k unique
    pool candidates occur per query. Each batch handles at most batch_size
    queries, limiting new dots to batch_size*2*r*r*k. Concurrent vector scratch
    is O(min(n_jobs,batch_size)*(1+2*r*r)*k*dimension), metadata O((Nq+Nr)*r),
    and output O(number_of_focal_queries*k). Original seed CSR occupies O(S).
    Changing batch size or worker count preserves selection order and scores.

    ``include_evidence`` returns original flags and strongest-witness rows/
    scores. Missing witnesses use row -1 and context score NaN. Witness ties
    use physical query row, then target row. Candidates rank by supported
    first, best witness+focal cosine, focal cosine, then target row. Support
    after independent per-query budgeting is a separate property: a selected
    candidate's original witness can itself lose its neighbor's final budget.
    """
    k, batch_size = _positive_integer(k, 'k'), _positive_integer(batch_size, 'batch_size')
    workers = _positive_integer(n_jobs if n_jobs is not None else (os.cpu_count() or 1), 'n_jobs')
    if isinstance(max_gap, bool) or not isinstance(max_gap, Integral) or max_gap < 0:
        raise ValueError('Bounded neighborhood extension requires a nonnegative integer max_gap')
    radius = int(max_gap) + 1
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('threshold must be a finite cosine cutoff in [-1, 1]')
    query_vectors, reference_vectors = np.asarray(query_vectors), np.asarray(reference_vectors)
    if (query_vectors.ndim != 2 or reference_vectors.ndim != 2 or
            query_vectors.shape[1] != reference_vectors.shape[1] or query_vectors.shape[1] == 0 or
            not np.issubdtype(query_vectors.dtype, np.floating) or
            not np.issubdtype(reference_vectors.dtype, np.floating)):
        raise ValueError('Query/reference floating matrices must have equal nonzero dimension')
    nq, nr = len(query_vectors), len(reference_vectors)
    dtype = np.result_type(query_vectors.dtype, reference_vectors.dtype, np.asarray(seed_scores).dtype, np.float32)
    qmeta = _metadata(query_metadata, nq)
    shared_metadata = query_metadata is reference_metadata and nq == nr
    rmeta = qmeta if shared_metadata else _metadata(reference_metadata, nr)
    if shared_metadata:
        _, qsample = np.unique(qmeta['sample_id'], return_inverse=True)
        rsample = qsample
    else:
        _, codes = np.unique(np.concatenate([qmeta['sample_id'], rmeta['sample_id']]), return_inverse=True)
        qsample, rsample = codes[:nq], codes[nq:]
    qneighbors = _neighbors(qmeta, qsample, radius)
    rneighbors = qneighbors if shared_metadata else _neighbors(rmeta, rsample, radius)
    qeligible = _eligible_rows(query_vectors)
    same_vectors = (query_vectors.shape == reference_vectors.shape and
                    query_vectors.strides == reference_vectors.strides and
                    query_vectors.dtype == reference_vectors.dtype and
                    query_vectors.ctypes.data == reference_vectors.ctypes.data)
    religible = qeligible if same_vectors and not normalize_reference else _eligible_rows(reference_vectors, normalize_reference)
    offsets, targets, scores, unique_seeds = _seed_csr(seed_query_rows, seed_target_rows, seed_scores,
        nq, nr, qsample, rsample, qeligible, religible, threshold, k)
    queries = np.arange(nq, dtype=np.int64) if query_rows is None else _integer_rows(query_rows, nq, 'query_rows')
    if query_rows is not None and len(np.unique(queries)) != len(queries):
        raise ValueError('query_rows must contain distinct physical rows')
    proposal_args = (offsets, targets, scores, qneighbors, rneighbors,
                     qmeta['strand'], rmeta['strand'], qsample, rsample, radius, k)
    count_names = ['proposed_occurrences', 'unique_proposed', 'dot_products', 'pool_size',
                   'selected', 'selected_supported', 'selected_original']
    totals = np.zeros(len(count_names), dtype=np.int64)

    def one(query):
        counts = np.zeros(len(count_names), dtype=np.int64)
        if not qeligible[query]:
            empty = np.empty(0, dtype=np.int64)
            return (empty, empty, np.empty(0, dtype=dtype)), None, counts
        pt, wq, wt, ws, occurrences = _proposals(int(query), *proposal_args)
        counts[0] = occurrences
        if len(pt):
            order = np.lexsort((wt, wq, -ws, pt))
            pt, wq, wt, ws = (a[order] for a in (pt, wq, wt, ws))
            first = np.r_[True, pt[1:] != pt[:-1]]
            pt, wq, wt, ws = (a[first] for a in (pt, wq, wt, ws))
        counts[1] = len(pt)
        start, end = offsets[query], offsets[query + 1]
        original, original_scores = targets[start:end], scores[start:end]
        union = np.union1d(original, pt)
        union = union[religible[union]]
        is_original = np.isin(union, original)
        similarity = np.empty(len(union), dtype=dtype)
        similarity[is_original] = original_scores[np.searchsorted(original, union[is_original])]
        new = ~is_original
        counts[2] = int(new.sum())
        if new.any():
            # A bounded GEMV with the same sorted target rows and original
            # floating arithmetic; source matrices are read-only to this API.
            values = reference_vectors[union[new]].astype(np.result_type(reference_vectors.dtype, np.float32), copy=False)
            if normalize_reference:
                values = values / (np.linalg.norm(values, axis=1, keepdims=True) + values.dtype.type(1e-9))
            qvector = query_vectors[query].astype(np.result_type(query_vectors.dtype, np.float32), copy=False)
            similarity[new] = values @ qvector
        keep = is_original | (np.isfinite(similarity) & (similarity >= threshold))
        union, similarity, is_original = union[keep], similarity[keep], is_original[keep]
        supported = np.isin(union, pt)
        context = np.full(len(union), np.nan, dtype=np.float64)
        where = np.searchsorted(pt, union[supported])
        context[supported] = similarity[supported].astype(np.float64) + ws[where]
        chosen = np.lexsort((union, -similarity, -np.where(supported, context, 0.),
                             -supported.astype(np.int8)))[:k]
        counts[3:] = len(union), len(chosen), int(supported[chosen].sum()), int(is_original[chosen].sum())
        numeric = (np.full(len(chosen), query, dtype=np.int64), union[chosen], similarity[chosen])
        evidence = None
        if include_evidence:
            chosen_q = np.full(len(union), -1, dtype=np.int64)
            chosen_t = np.full(len(union), -1, dtype=np.int64)
            chosen_q[supported], chosen_t[supported] = wq[where], wt[where]
            evidence = (is_original[chosen], chosen_q[chosen], chosen_t[chosen], context[chosen])
        return numeric, evidence, counts

    parts = [[], [], []]
    evidence_parts = [[], [], [], []]
    # Limit BLAS only. OpenMP runtime settings used by an upstream FAISS index
    # remain untouched; proposals release the GIL when Numba is installed.
    with threadpool_limits(limits=1, user_api='blas'):
        with ThreadPoolExecutor(max_workers=min(workers, batch_size)) as executor:
            for start in range(0, len(queries), batch_size):
                batch = queries[start:start + batch_size]
                collected = [[], [], []]
                evidence_batch = [[], [], [], []]
                for numeric, evidence, counts in executor.map(one, batch):
                    totals += counts
                    for destination, values in zip(collected, numeric):
                        destination.append(values)
                    if evidence is not None:
                        for destination, values in zip(evidence_batch, evidence):
                            destination.append(values)
                for destination, values in zip(parts, collected):
                    destination.append(np.concatenate(values))
                for destination, values in zip(evidence_parts, evidence_batch):
                    if values:
                        destination.append(np.concatenate(values))
    arrays = [np.concatenate(p) if p else np.empty(0, dtype=d)
              for p, d in zip(parts, [np.int64, np.int64, dtype])]
    evidence = [np.concatenate(p) if p else np.empty(0, dtype=d)
                for p, d in zip(evidence_parts, [bool, np.int64, np.int64, np.float64])]
    counts = dict(zip(count_names, map(int, totals)))
    counts.update(query_rows=len(queries), eligible_query_rows=int(qeligible[queries].sum()),
        original_seed_rows=len(seed_scores), unique_original_pairs=unique_seeds,
        eligible_original_pairs=len(targets), per_query_budget=k,
        per_query_proposal_bound=2 * radius * radius * k,
        per_query_pool_bound=(1 + 2 * radius * radius) * k,
        per_batch_dot_product_bound=batch_size * 2 * radius * radius * k)
    optional = (dict(zip(['is_original', 'witness_query_rows', 'witness_target_rows',
                          'context_scores'], evidence)) if include_evidence else {})
    return NeighborhoodSelection(*arrays, counts=counts, **optional)
