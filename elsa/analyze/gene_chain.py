"""
Backward-compatibility shim for gene_chain module.

All implementations have moved to:
  - elsa.seed (GeneAnchor, find_cross_genome_anchors, group_anchors_by_contig_pair)
  - elsa.index (build_gene_index)
  - elsa.chain (ChainedBlock, chain_anchors_lis, extract_nonoverlapping_chains)
  - elsa.analyze.pipeline (_run_gene_chaining as run_gene_chaining)
"""

# Re-export everything from new locations
from elsa.seed import GeneAnchor, find_cross_genome_anchors, group_anchors_by_contig_pair
from elsa.index import build_gene_index
from elsa.chain import ChainedBlock, chain_anchors_lis, extract_nonoverlapping_chains
from elsa.analyze.pipeline import _run_gene_chaining as run_gene_chaining

__all__ = [
    'GeneAnchor',
    'ChainedBlock',
    'build_gene_index',
    'find_cross_genome_anchors',
    'group_anchors_by_contig_pair',
    'chain_anchors_lis',
    'extract_nonoverlapping_chains',
    'run_gene_chaining',
]
