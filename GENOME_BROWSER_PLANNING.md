# ELSA Genome Browser Planning Document

## Overview
Interactive Streamlit application for browsing genome diagrams with PFAM domain annotations overlaid on syntenic blocks. The browser will handle the bimodal size distribution observed in syntenic blocks and provide efficient navigation through large datasets.

## Architecture Design

### 1. Data Processing Pipeline

#### Input Data Sources
- **Syntenic Blocks**: `syntenic_analysis/syntenic_blocks.csv` (22,791 blocks)
- **Syntenic Clusters**: `syntenic_analysis/syntenic_clusters.csv` (8 clusters)  
- **Protein Sequences**: `test_data/genomes/*.faa` (FASTA format)
- **Gene Coordinates**: `test_data/genomes/*.gff` (GFF3 format)
- **Genome Sequences**: `test_data/genomes/*.fna` (optional, for sequence context)

#### PFAM Annotation Integration
Based on `../microbial_claude_matter` implementation:
- **Astra scan pipeline**: Use `04_astra_scan.py` approach with PyHMMer
- **Annotation processing**: Adapt `annotation_processors.py` `PfamProcessor` class
- **Domain format**: Join multiple PFAM domains per protein with `;` separator
- **Example output**: `PF00001.21;PF00002.15;PF00003.8` for a protein with 3 domains

### 2. Data Storage Strategy

#### SQLite Database Schema
```sql
-- Genome metadata
CREATE TABLE genomes (
    genome_id TEXT PRIMARY KEY,
    organism_name TEXT,
    total_contigs INTEGER,
    total_genes INTEGER,
    file_path TEXT
);

-- Gene/protein information  
CREATE TABLE genes (
    gene_id TEXT PRIMARY KEY,
    genome_id TEXT,
    contig_id TEXT,
    start_pos INTEGER,
    end_pos INTEGER,
    strand INTEGER, -- -1 or 1
    protein_sequence TEXT,
    pfam_domains TEXT, -- semicolon-separated
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id)
);

-- Syntenic blocks (indexed for fast lookup)
CREATE TABLE syntenic_blocks (
    block_id INTEGER PRIMARY KEY,
    query_locus TEXT,
    target_locus TEXT,
    length INTEGER,
    identity REAL,
    score REAL,
    n_query_windows INTEGER,
    n_target_windows INTEGER
);

-- Cluster assignments
CREATE TABLE cluster_assignments (
    block_id INTEGER,
    cluster_id INTEGER,
    FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id)
);

-- Indexes for performance
CREATE INDEX idx_genes_genome ON genes(genome_id);
CREATE INDEX idx_genes_location ON genes(contig_id, start_pos, end_pos);
CREATE INDEX idx_blocks_query ON syntenic_blocks(query_locus);
CREATE INDEX idx_blocks_target ON syntenic_blocks(target_locus);
CREATE INDEX idx_blocks_length ON syntenic_blocks(length);
```

### 3. Streamlit Application Architecture

#### Page Structure
```
📊 ELSA Genome Browser
├── 🏠 Home/Dashboard
├── 🔍 Syntenic Block Explorer  
├── 🧬 Genome Viewer
├── 🎯 Cluster Analysis
└── 📈 Statistics & QC
```

#### Component Design

**A. Navigation & Filtering Sidebar**
- Genome selector (multi-select)
- Block size range slider (log scale, handles bimodal distribution)
- Identity threshold slider
- Cluster filter (show/hide specific clusters)
- PFAM domain search box
- Results per page selector (10, 25, 50, 100)

**B. Main Content Area**
- **Block List View**: Paginated table with sortable columns
- **Genome Diagram View**: Interactive SVG/Plotly diagrams
- **Detailed Block View**: Expandable rows with gene details

**C. Performance Optimizations**
- **Pagination**: Limit to 100 blocks per page maximum
- **Lazy loading**: Load gene details only when blocks expanded
- **Caching**: Cache PFAM annotation results with `@st.cache_data`
- **Database queries**: Indexed queries with LIMIT/OFFSET for pagination

### 4. PFAM Annotation Workflow

#### Integration Strategy
1. **Reuse existing pipeline**: Adapt `../microbial_claude_matter/src/ingest/04_astra_scan.py`
2. **Batch processing**: Process all genomes in `test_data/genomes/` 
3. **Domain concatenation**: Join domains as `domain1.version;domain2.version;...`
4. **Caching**: Store results in SQLite for fast retrieval

#### Implementation Details
```python
# Adapted from microbial_claude_matter/src/build_kg/annotation_processors.py
class ELSAPfamProcessor:
    def process_protein_domains(self, protein_id: str, pfam_hits: pd.DataFrame) -> str:
        """Join PFAM domains with semicolon separator ordered by position."""
        # Filter significant hits (E-value <= 1e-5)
        significant_hits = pfam_hits[pfam_hits['evalue'] <= 1e-5]
        
        # Sort by envelope start position
        sorted_hits = significant_hits.sort_values('env_from')
        
        # Join domain names with versions
        domains = []
        for _, hit in sorted_hits.iterrows():
            domain_name = f"{hit['hmm_name']}"  # Already includes version
            domains.append(domain_name)
        
        return ';'.join(domains) if domains else ""
```

### 5. Bimodal Distribution Handling

#### Observed Pattern
- **Small blocks**: High frequency, shorter lengths (~1-10kb)
- **Large blocks**: Lower frequency, longer lengths (>25kb)
- **UI implications**: Need log-scale sliders and smart binning

#### UI Design Adaptations
- **Size filter**: Log-scale slider with preset ranges
  - Small: <5kb
  - Medium: 5-25kb  
  - Large: >25kb
- **Default view**: Show medium/large blocks first (more biologically interesting)
- **Performance**: Implement smart sampling for very large clusters

### 6. Hardware Constraints & Performance

#### Memory Management
- **Pagination**: Never load all 22,791 blocks at once
- **Streaming**: Use database cursors for large result sets
- **Caching**: Strategic caching of frequently accessed data
- **Batch processing**: Process PFAM annotations in chunks

#### Responsive Design
- **Progressive disclosure**: Summary → Details → Full annotation
- **Async loading**: Use Streamlit's async features where possible
- **Mobile-friendly**: Responsive layouts for tablet viewing

### 7. User Experience Flow

#### Primary Workflow
1. **Landing Page**: Overview statistics, quick filters
2. **Block Selection**: Browse/search syntenic blocks with pagination  
3. **Genome Context**: Click block → view genomic region with genes
4. **Annotation Details**: Hover/click genes → show PFAM domains
5. **Comparative View**: Side-by-side query/target loci comparison

#### Secondary Features
- **Export functionality**: Download filtered block lists
- **Bookmarking**: Save interesting blocks/clusters
- **Search**: Find blocks by genome, size, or PFAM domain
- **Help tooltips**: Explain syntenic blocks, PFAM domains, clustering

### 8. Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
- [ ] SQLite database setup and population
- [ ] Basic Streamlit app with navigation
- [ ] PFAM annotation pipeline integration  
- [ ] Simple block list with pagination

#### Phase 2: Visualization (Week 2)
- [ ] Genome diagram rendering (Plotly/matplotlib)
- [ ] Gene annotation overlays
- [ ] Interactive filtering and search
- [ ] Basic block detail views

#### Phase 3: Advanced Features (Week 3)
- [ ] Cluster visualization and analysis
- [ ] Comparative genome views
- [ ] Export/download functionality
- [ ] Performance optimizations
- [ ] Mobile responsiveness

#### Phase 4: Polish & Deploy (Week 4)
- [ ] UI/UX improvements
- [ ] Help documentation
- [ ] Error handling and validation
- [ ] Deployment preparation

### 9. Technical Stack

#### Core Technologies
- **Streamlit**: Web framework
- **SQLite**: Local database (fast, serverless)
- **Pandas**: Data manipulation
- **Plotly/Matplotlib**: Genome diagrams
- **PyHMMer/Astra**: PFAM annotation (from microbial_claude_matter)

#### File Structure
```
genome_browser/
├── app.py                    # Main Streamlit app
├── database/
│   ├── setup_db.py          # Database initialization
│   ├── populate_db.py       # Data ingestion pipeline  
│   └── queries.py           # SQL query functions
├── annotation/
│   ├── pfam_processor.py    # PFAM annotation pipeline
│   └── domain_utils.py      # Domain parsing utilities
├── visualization/
│   ├── genome_plots.py      # Genome diagram rendering
│   ├── block_plots.py       # Syntenic block visualization
│   └── cluster_plots.py     # Cluster analysis plots
├── utils/
│   ├── data_loader.py       # File I/O utilities
│   ├── pagination.py        # Pagination helpers
│   └── caching.py           # Streamlit caching strategies
└── static/
    ├── styles.css           # Custom CSS
    └── help/                # Help documentation
```

### 10. Success Metrics

#### Performance Targets
- **Page load time**: <3 seconds for any view
- **Block listing**: Handle 100+ blocks per page smoothly
- **Memory usage**: <2GB RAM for full dataset
- **Responsiveness**: Interactive filtering with <1 second response

#### User Experience Goals
- **Intuitive navigation**: Users can find syntenic blocks of interest quickly
- **Rich annotations**: PFAM domains clearly displayed and searchable
- **Scalable design**: Handle larger datasets without redesign
- **Export capability**: Users can download results for further analysis

## Next Steps

1. **Create database setup script** with schema and indexing
2. **Integrate PFAM annotation pipeline** from microbial_claude_matter
3. **Build basic Streamlit prototype** with block listing and pagination
4. **Implement genome visualization** with gene/domain overlays
5. **Add advanced filtering and search** capabilities
6. **Optimize performance** for large dataset handling

This architecture balances biological utility, technical feasibility, and user experience while respecting hardware constraints and the bimodal size distribution observed in the syntenic blocks.