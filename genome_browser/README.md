# ELSA Genome Browser

Interactive Streamlit application for exploring syntenic blocks with PFAM domain annotations. Handles large datasets efficiently with pagination and provides rich visualizations of genomic relationships.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- `astra` command-line tool (for PFAM annotation)
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone or navigate to the genome browser directory
cd genome_browser

# Install Python dependencies
pip install -r requirements.txt

# Set up the genome browser with your ELSA data
python setup_genome_browser.py \
    --genome-dir ../test_data/genomes \
    --blocks-file ../syntenic_analysis/syntenic_blocks.csv \
    --clusters-file ../syntenic_analysis/syntenic_clusters.csv

# Start the web application
streamlit run app.py
```

The genome browser will be available at `http://localhost:8501`

## 📁 Project Structure

```
genome_browser/
├── app.py                     # Main Streamlit application
├── setup_genome_browser.py    # Complete setup script
├── requirements.txt           # Python dependencies
├── database/
│   ├── setup_db.py           # Database schema creation
│   └── populate_db.py        # Data ingestion pipeline
├── annotation/
│   └── pfam_processor.py     # PFAM annotation using astra
├── visualization/
│   └── genome_plots.py       # Plotly genome diagrams
└── README.md                 # This file
```

## 🔧 Setup Options

### Basic Setup (No PFAM Annotations)
```bash
python setup_genome_browser.py \
    --genome-dir ../test_data/genomes \
    --blocks-file ../syntenic_analysis/syntenic_blocks.csv \
    --clusters-file ../syntenic_analysis/syntenic_clusters.csv \
    --skip-pfam
```

### Full Setup with PFAM Annotations
```bash
python setup_genome_browser.py \
    --genome-dir ../test_data/genomes \
    --blocks-file ../syntenic_analysis/syntenic_blocks.csv \
    --clusters-file ../syntenic_analysis/syntenic_clusters.csv \
    --threads 8 \
    --max-workers 2
```

### Custom Database Location
```bash
python setup_genome_browser.py \
    --genome-dir ../test_data/genomes \
    --blocks-file ../syntenic_analysis/syntenic_blocks.csv \
    --clusters-file ../syntenic_analysis/syntenic_clusters.csv \
    --db-path /path/to/custom/database.db
```

## 🎯 Features

### Dashboard
- **Overview Statistics**: Total genomes, syntenic blocks, clusters, genes
- **Size Distribution**: Handles bimodal distribution with smart categorization
- **Genome Comparison Matrix**: Heatmap of syntenic relationships

### Block Explorer
- **Smart Pagination**: Handle 22,791+ blocks efficiently (max 200 per page)
- **Advanced Filtering**:
  - Genome selection (multi-select)
  - Block size range (log scale for bimodal distribution)
  - Identity threshold slider
  - Block type categories (small <5kb, medium 5-25kb, large >25kb)
- **PFAM Domain Search**: Find blocks containing specific domains
- **Interactive Table**: Click blocks to view detailed information

### Genome Viewer
- **Interactive Diagrams**: Plotly-based genome visualization
- **Gene Arrows**: Strand-aware gene representation
- **PFAM Domain Track**: Visual domain annotations
- **Hover Details**: Rich tooltips with gene and domain information
- **Comparative View**: Side-by-side locus comparison (planned)

## 🎨 Visualization Features

### Genome Diagrams
- **Three-track layout**: Scale bar, genes, PFAM domains
- **Color coding**: 
  - Syntenic genes: Red/tomato
  - Non-syntenic genes: Blue/sky blue
  - Forward strand: Green
  - Reverse strand: Orange
  - PFAM domains: Color-coded by domain hash
- **Interactive elements**: Hover for details, zoom, pan

### Performance Optimizations
- **Efficient rendering**: Only visible elements loaded
- **Smart caching**: Streamlit `@st.cache_data` for database queries
- **Memory management**: Pagination prevents memory overload
- **Database indexing**: Optimized SQLite indexes for fast queries

## 📊 Data Format Requirements

### Input Files
Your data should be organized as follows:

```
test_data/genomes/
├── genome1.fna    # Nucleotide sequences (FASTA)
├── genome1.gff    # Gene annotations (GFF3)
├── genome1.faa    # Protein sequences (FASTA)
├── genome2.fna
├── genome2.gff
└── genome2.faa
```

### ELSA Analysis Files
```
syntenic_analysis/
├── syntenic_blocks.csv     # Block data with columns: block_id, query_locus, target_locus, length, identity, score
└── syntenic_clusters.csv   # Cluster data with columns: cluster_id, size, consensus_length, diversity
```

## 🔍 Usage Examples

### Finding High-Quality Syntenic Blocks
1. Go to **Block Explorer**
2. Set **Min Identity** to 0.9
3. Select **Block Type**: Large
4. Filter by specific genomes of interest

### Searching by PFAM Domain
1. In the sidebar, enter domain ID (e.g., "PF00001") or keyword (e.g., "kinase")
2. Browse results to find genes with those domains
3. Click blocks to view genome context

### Exploring Bimodal Size Distribution
1. Use the **Block Type** filter to explore small vs. large blocks
2. Adjust **Block Size Range** with log-scale slider
3. Notice different biological patterns in each size class

## 🛠️ Troubleshooting

### Database Issues
```bash
# Check database contents
python database/setup_db.py --info --db-path genome_browser.db

# Recreate database
python database/setup_db.py --force --db-path genome_browser.db
```

### PFAM Annotation Issues
```bash
# Check astra installation
astra --help

# Run annotation separately
python annotation/pfam_processor.py --protein-dir ../test_data/genomes --output-dir pfam_test
```

### Performance Issues
- Reduce **Results per page** in sidebar
- Use more restrictive filters to limit dataset size
- Ensure database indexes are created (automatic during setup)

## 📈 Scalability

The genome browser is designed to handle:
- **Genomes**: 10-100+ genomes
- **Syntenic blocks**: 100k+ blocks (with pagination)
- **Genes**: 1M+ genes (with indexing)
- **Memory usage**: <2GB RAM for typical datasets

For larger datasets, consider:
- Increasing pagination limits
- Using more selective default filters
- Database optimization (`VACUUM` and `ANALYZE`)

## 🔬 Biological Insights

The browser helps answer questions like:
- Which genome pairs have the most syntenic relationships?
- What PFAM domains are enriched in syntenic vs non-syntenic regions?
- How does the bimodal size distribution reflect different biological processes?
- Which gene clusters are most conserved across genomes?

## 🤝 Contributing

To extend the genome browser:
1. **Add new visualizations**: Extend `visualization/genome_plots.py`
2. **Add analysis features**: Modify `app.py` with new tabs/sections
3. **Enhance database schema**: Update `database/setup_db.py`
4. **Add new annotation types**: Extend `annotation/` directory

## 📝 Citation

If you use this genome browser in your research, please cite:
- The ELSA method paper
- Streamlit framework
- Plotly visualization library
- Any relevant bioinformatics tools (astra, PFAM, etc.)