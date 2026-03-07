"""
Collinear anchor chaining via LIS-based dynamic programming.

Pure algorithm module — no I/O, no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

from .seed import GeneAnchor


@dataclass
class ChainedBlock:
    """A chained syntenic block from collinear gene anchors."""
    block_id: int
    query_genome: str
    target_genome: str
    query_contig: str
    target_contig: str
    query_start: int         # Gene index (0-based)
    query_end: int           # Gene index (inclusive)
    target_start: int
    target_end: int
    anchors: List[GeneAnchor] = field(default_factory=list)
    chain_score: float = 0.0
    orientation: int = 1     # +1 forward, -1 inverted

    @property
    def n_anchors(self) -> int:
        return len(self.anchors)

    @property
    def query_span(self) -> int:
        return self.query_end - self.query_start + 1

    @property
    def target_span(self) -> int:
        return self.target_end - self.target_start + 1

    def query_gene_ids(self) -> List[str]:
        """Get ordered query gene IDs from anchors."""
        return [a.query_gene_id for a in sorted(self.anchors, key=lambda a: a.query_idx)]

    def target_gene_ids(self) -> List[str]:
        """Get ordered target gene IDs from anchors."""
        return [a.target_gene_id for a in sorted(self.anchors, key=lambda a: a.target_idx)]


def chain_anchors_lis(
    anchors: List[GeneAnchor],
    max_gap: int = 2,
    min_size: int = 2,
    gap_penalty_scale: float = 0.0,
) -> List[List[GeneAnchor]]:
    """
    Find collinear chains using LIS-based dynamic programming.

    Tries both forward and reverse orientations and returns all
    chains meeting the minimum size requirement.

    Args:
        anchors: List of GeneAnchor objects for a single contig pair
        max_gap: Maximum gap (in genes) allowed between chain members.
                 Set to None to disable hard cutoff (use with gap_penalty_scale > 0).
        min_size: Minimum number of anchors required for a valid chain
        gap_penalty_scale: Concave gap penalty multiplier. Score reduction =
                          scale * log2(max(gap_q, gap_t) + 1). Default 0.0
                          preserves exact legacy behavior.

    Returns:
        List of anchor chains (each chain is a list of GeneAnchor)
    """
    if len(anchors) < min_size:
        return []

    # Deduplicate: keep best anchor per (query_idx, target_idx) pair
    best_per_pair: Dict[Tuple[int, int], GeneAnchor] = {}
    for a in anchors:
        key = (a.query_idx, a.target_idx)
        if key not in best_per_pair or a.similarity > best_per_pair[key].similarity:
            best_per_pair[key] = a
    anchors = list(best_per_pair.values())

    if len(anchors) < min_size:
        return []

    # Check if strand info is available (orientation != 0)
    has_strand = any(a.orientation != 0 for a in anchors)

    if has_strand:
        # Strand-aware partitioning: split anchors by relative orientation
        # before DP, instead of trying both directions
        partitions = []
        fwd = [a for a in anchors if a.orientation >= 0]  # same-strand or unknown
        rev = [a for a in anchors if a.orientation < 0]    # opposite-strand
        if len(fwd) >= min_size:
            partitions.append((fwd, False))  # forward target ordering
        if len(rev) >= min_size:
            partitions.append((rev, True))   # reverse target ordering
    else:
        # No strand info: try both orientations (legacy two-pass)
        partitions = [(anchors, False), (anchors, True)]

    all_chains = []

    for partition_anchors, reverse_target in partitions:
        # Deduplicate per query_idx (keep best match) for 1:1 mapping
        best_per_query: Dict[int, GeneAnchor] = {}
        for a in partition_anchors:
            if a.query_idx not in best_per_query or a.similarity > best_per_query[a.query_idx].similarity:
                best_per_query[a.query_idx] = a
        deduped_anchors = list(best_per_query.values())

        if len(deduped_anchors) < min_size:
            continue

        # Sort by query position
        sorted_anchors = sorted(deduped_anchors, key=lambda a: a.query_idx)

        # Compute comparison key for target position
        if reverse_target:
            cmp_vals = [-a.target_idx for a in sorted_anchors]
        else:
            cmp_vals = [a.target_idx for a in sorted_anchors]

        n = len(sorted_anchors)
        # dp[i] = (chain_length, prev_index, cumulative_score)
        dp: List[Tuple[int, int, float]] = [(1, -1, a.similarity) for a in sorted_anchors]

        for i in range(1, n):
            best_len, best_prev, best_score = 1, -1, sorted_anchors[i].similarity

            for j in range(i):
                # Check gap constraint on query side
                gap_q = sorted_anchors[i].query_idx - sorted_anchors[j].query_idx - 1
                if max_gap is not None and gap_q > max_gap:
                    continue

                # Check monotonicity on target side
                if cmp_vals[i] <= cmp_vals[j]:
                    continue

                # Check gap constraint on target side
                gap_t = abs(sorted_anchors[i].target_idx - sorted_anchors[j].target_idx) - 1
                if max_gap is not None and gap_t > max_gap:
                    continue

                # Compute gap cost
                if gap_penalty_scale > 0:
                    gap_cost = gap_penalty_scale * math.log2(max(gap_q, gap_t) + 1)
                else:
                    gap_cost = 0.0

                # Update if extending this chain is better
                new_len = dp[j][0] + 1
                new_score = dp[j][2] + sorted_anchors[i].similarity - gap_cost
                if new_len > best_len or (new_len == best_len and new_score > best_score):
                    best_len = new_len
                    best_prev = j
                    best_score = new_score

            dp[i] = (best_len, best_prev, best_score)

        # Backtrack to extract chains (greedy: longest first)
        used: Set[int] = set()
        indices = sorted(range(n), key=lambda i: (-dp[i][0], -dp[i][2]))

        for i in indices:
            if i in used or dp[i][0] < min_size:
                continue

            chain = []
            j = i
            while j >= 0:
                if j in used:
                    break
                chain.append(sorted_anchors[j])
                used.add(j)
                j = dp[j][1]

            if len(chain) >= min_size:
                chain.reverse()
                # Mark orientation
                orientation = -1 if reverse_target else 1
                for anchor in chain:
                    anchor.orientation = orientation
                all_chains.append(chain)

    return all_chains


def extract_nonoverlapping_chains(
    chains: List[List[GeneAnchor]],
    block_id_start: int = 0,
) -> List[ChainedBlock]:
    """
    Extract non-overlapping blocks from chains using greedy selection.

    Selects chains by score (length * mean_similarity), avoiding
    overlaps on both query and target sides.

    Args:
        chains: List of anchor chains from chain_anchors_lis
        block_id_start: Starting block ID for numbering

    Returns:
        List of ChainedBlock objects
    """
    if not chains:
        return []

    # Score each chain
    scored_chains = []
    for chain in chains:
        if not chain:
            continue
        mean_sim = sum(a.similarity for a in chain) / len(chain)
        score = len(chain) * mean_sim

        q_min = min(a.query_idx for a in chain)
        q_max = max(a.query_idx for a in chain)
        t_min = min(a.target_idx for a in chain)
        t_max = max(a.target_idx for a in chain)

        orientation = getattr(chain[0], 'orientation', 1)

        scored_chains.append((score, chain, q_min, q_max, t_min, t_max, orientation))

    # Sort by score descending
    scored_chains.sort(key=lambda x: -x[0])

    # Greedy selection avoiding overlaps
    blocks = []
    used_query: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    used_target: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

    def overlaps(intervals: List[Tuple[int, int]], start: int, end: int) -> bool:
        for s, e in intervals:
            if not (end < s or start > e):
                return True
        return False

    block_id = block_id_start
    for score, chain, q_min, q_max, t_min, t_max, orientation in scored_chains:
        if not chain:
            continue

        q_key = (chain[0].query_genome, chain[0].query_contig)
        t_key = (chain[0].target_genome, chain[0].target_contig)

        q_intervals = used_query.get(q_key, [])
        t_intervals = used_target.get(t_key, [])

        if overlaps(q_intervals, q_min, q_max):
            continue
        if overlaps(t_intervals, t_min, t_max):
            continue

        used_query.setdefault(q_key, []).append((q_min, q_max))
        used_target.setdefault(t_key, []).append((t_min, t_max))

        block = ChainedBlock(
            block_id=block_id,
            query_genome=chain[0].query_genome,
            target_genome=chain[0].target_genome,
            query_contig=chain[0].query_contig,
            target_contig=chain[0].target_contig,
            query_start=q_min,
            query_end=q_max,
            target_start=t_min,
            target_end=t_max,
            anchors=chain,
            chain_score=score,
            orientation=orientation,
        )
        blocks.append(block)
        block_id += 1

    return blocks
