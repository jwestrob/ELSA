# Genome Browser Redesign: From 15 Genomes to 5,000+

## Context

I'm building **ELSA** (Embedding Locus Search and Alignment), a tool that discovers conserved gene neighborhoods (syntenic blocks) across microbial genomes using protein language model embeddings. The pipeline:

1. Embeds all proteins with ESM2 (480D vectors)
2. Finds cross-genome anchor genes via FAISS kNN
3. Chains collinear anchors into syntenic blocks (like minimap2 for gene order)
4. Clusters overlapping blocks by Jaccard similarity
5. Merges contained clusters
6. Computes per-cluster architecture schemas (slot-based structural summaries)

**The scale problem**: ELSA was originally built for 6-15 genomes (~800 clusters). It now handles **3,500+ genomes with 3M+ blocks and 45,000+ meaningful clusters**. The current genome browser (Streamlit + D3.js clinker-style diagrams + SQLite) cannot handle this scale — rendering even a fraction of the clusters crashes the browser tab.

## What We Have

### Data artifacts per run

| Artifact | Description | DPANN scale |
|----------|-------------|-------------|
| `micro_chain_blocks.csv` | All syntenic blocks with cluster assignments | 3M rows |
| `micro_chain_clusters.csv` | Cluster summaries (size, genome support, genes) | 45k rows |
| `schema/cluster_architecture_summary.parquet` | Per-cluster: n_slots, coherence, arch_label | 45k rows |
| `schema/cluster_slot_assignments.parquet` | Gene-to-slot assignments per cluster | ~2M rows |
| `schema/cluster_slots.parquet` | Per-slot: occupancy, dispersion, centroid gene | ~200k rows |
| `schema/cluster_loci.parquet` | Deduplicated locus instances per cluster | ~900k rows |
| `genes.parquet` | All genes with 480D embeddings, coordinates, contigs | 3.4M rows |

### Current browser features (Streamlit)
- Cluster explorer cards with D3 clinker-style gene arrow diagrams
- Genome-contig navigation
- Block detail panels
- Architecture badges (coherent/variable/fragmented)
- Gene-level PFAM annotations when available

### CLI interface (for agents/programmatic use)
- `elsa search GENOME:CONTIG:START-END` — find syntenic regions matching a locus
- `elsa synteny` — run the full pipeline
- `elsa stats` — summary statistics

### Architecture schema (already computed)

Each cluster has a slot-based structural summary:
- **Reference locus**: highest-support locus defines N slots
- **Slot assignment**: other loci aligned to reference via embedding similarity + monotonic DP
- **Slot summaries**: occupancy, dispersion, centroid gene, core/variable classification
- **Architecture label**: coherent (>70% core, high occupancy), variable (replacement hotspots), moderate, fragmented

## Design Constraints

- **Two user classes**: (1) Humans exploring biology via GUI, (2) LLM agents querying via CLI
- **Runs on a laptop**: M4 Max MacBook Pro, 48GB unified memory. No cloud infra assumed.
- **Python ecosystem**: numpy, pandas, FAISS already in the stack. Open to adding frameworks.
- **Data is local**: All artifacts on disk. No remote API needed.
- **Clinker diagrams are the gold standard** for visualizing gene neighborhood conservation — horizontal gene arrows aligned across genomes, with similarity links between homologous genes. But they only work for ~2-20 loci at a time.

## The Question

**How should we redesign the genome browser to handle 45,000+ clusters across 5,000+ genomes?**

Specifically:

1. **Discovery/triage**: How does a user find the interesting clusters? Faceted search? Embedding-based similarity search? Pre-computed rankings? What filters matter (genome support, coherence, slot count, specific genomes, specific genes)?

2. **Visualization strategy**: Clinker diagrams can't render 45k clusters. What's the right hierarchy? Summary views → cluster detail → locus-level diagrams? What summary visualizations work at the 45k scale (heatmaps, network graphs, treemaps)?

3. **Architecture**: Streamlit + SQLite? FastAPI + DuckDB? Panel/Holoviz? Datasette? Something else? What handles 3M+ rows of block data interactively on a laptop?

4. **Agent interface**: The CLI already has `elsa search`. What additional query patterns do LLM agents need? Should the browser expose an API that agents can also use?

5. **Rendering strategy**: Pre-compute static views for top clusters? Lazy-render on demand? WebGL/Canvas instead of SVG for larger diagrams?

Please propose a concrete architecture with specific technology choices, UI/UX flow, and data flow. Assume I'm a computational biologist comfortable with Python but not a frontend specialist. I care more about biological insight than UI polish.
