#!/usr/bin/env python3
"""
SQLite database setup for ELSA genome browser.
Creates optimized schema for fast syntenic block and gene annotation queries.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_SCHEMA = """
-- Genome metadata
CREATE TABLE IF NOT EXISTS genomes (
    genome_id TEXT PRIMARY KEY,
    organism_name TEXT,
    total_contigs INTEGER,
    total_genes INTEGER,
    genome_size INTEGER,
    file_path_fna TEXT,
    file_path_faa TEXT,
    file_path_gff TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Contig information
CREATE TABLE IF NOT EXISTS contigs (
    contig_id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    contig_name TEXT,
    length INTEGER,
    gc_content REAL,
    gene_count INTEGER,
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id)
);

-- Gene/protein information with PFAM annotations
CREATE TABLE IF NOT EXISTS genes (
    gene_id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    contig_id TEXT NOT NULL,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    strand INTEGER NOT NULL, -- -1 or 1
    gene_length INTEGER,
    protein_id TEXT,
    protein_sequence TEXT,
    pfam_domains TEXT, -- semicolon-separated domain list
    pfam_count INTEGER DEFAULT 0, -- number of domains for quick filtering
    gc_content REAL,
    partial_gene BOOLEAN DEFAULT FALSE,
    confidence_score REAL,
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id),
    FOREIGN KEY (contig_id) REFERENCES contigs(contig_id)
);

-- Syntenic blocks from ELSA analysis
CREATE TABLE IF NOT EXISTS syntenic_blocks (
    block_id INTEGER PRIMARY KEY,
    cluster_id INTEGER DEFAULT 0, -- Cluster assignment from ELSA clustering
    query_locus TEXT NOT NULL,
    target_locus TEXT NOT NULL,
    query_genome_id TEXT,
    target_genome_id TEXT,
    query_contig_id TEXT,
    target_contig_id TEXT,
    length INTEGER NOT NULL,
    identity REAL NOT NULL,
    score REAL NOT NULL,
    n_query_windows INTEGER,
    n_target_windows INTEGER,
    query_window_start INTEGER, -- Start window index for query alignment
    query_window_end INTEGER,   -- End window index for query alignment  
    target_window_start INTEGER, -- Start window index for target alignment
    target_window_end INTEGER,   -- End window index for target alignment
    query_windows_json TEXT,    -- JSON array of aligned query window IDs
    target_windows_json TEXT,   -- JSON array of aligned target window IDs
    block_type TEXT, -- 'small', 'medium', 'large' based on length
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cluster assignments from ELSA clustering
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    size INTEGER NOT NULL, -- number of blocks in cluster
    consensus_length INTEGER,
    consensus_score REAL,
    diversity REAL,
    representative_query TEXT,
    representative_target TEXT,
    cluster_type TEXT -- based on size and diversity
);

CREATE TABLE IF NOT EXISTS cluster_assignments (
    block_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    PRIMARY KEY (block_id, cluster_id),
    FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

-- PFAM domain catalog for quick lookups
CREATE TABLE IF NOT EXISTS pfam_domains (
    pfam_id TEXT PRIMARY KEY,
    pfam_name TEXT,
    description TEXT,
    domain_type TEXT,
    protein_count INTEGER DEFAULT 0 -- how many proteins have this domain
);

-- Gene-to-block mappings for fast lookups
CREATE TABLE IF NOT EXISTS gene_block_mappings (
    gene_id TEXT,
    block_id INTEGER,
    block_role TEXT, -- 'query' or 'target'
    relative_position REAL, -- position within the block (0-1)
    PRIMARY KEY (gene_id, block_id),
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id),
    FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id)
);

-- Annotation statistics for dashboard
CREATE TABLE IF NOT EXISTS annotation_stats (
    genome_id TEXT PRIMARY KEY,
    total_genes INTEGER,
    annotated_genes INTEGER,
    total_pfam_domains INTEGER,
    unique_pfam_domains INTEGER,
    syntenic_genes INTEGER,
    non_syntenic_genes INTEGER,
    annotation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id)
);
"""

# Performance indexes
PERFORMANCE_INDEXES = """
-- Core performance indexes
CREATE INDEX IF NOT EXISTS idx_genes_genome ON genes(genome_id);
CREATE INDEX IF NOT EXISTS idx_genes_contig ON genes(contig_id);
CREATE INDEX IF NOT EXISTS idx_genes_location ON genes(contig_id, start_pos, end_pos);
CREATE INDEX IF NOT EXISTS idx_genes_strand ON genes(strand);
CREATE INDEX IF NOT EXISTS idx_genes_pfam_count ON genes(pfam_count);

-- Syntenic block indexes
CREATE INDEX IF NOT EXISTS idx_blocks_query_locus ON syntenic_blocks(query_locus);
CREATE INDEX IF NOT EXISTS idx_blocks_target_locus ON syntenic_blocks(target_locus);
CREATE INDEX IF NOT EXISTS idx_blocks_query_genome ON syntenic_blocks(query_genome_id);
CREATE INDEX IF NOT EXISTS idx_blocks_target_genome ON syntenic_blocks(target_genome_id);
CREATE INDEX IF NOT EXISTS idx_blocks_length ON syntenic_blocks(length);
CREATE INDEX IF NOT EXISTS idx_blocks_identity ON syntenic_blocks(identity);
CREATE INDEX IF NOT EXISTS idx_blocks_score ON syntenic_blocks(score);
CREATE INDEX IF NOT EXISTS idx_blocks_type ON syntenic_blocks(block_type);
CREATE INDEX IF NOT EXISTS idx_blocks_cluster ON syntenic_blocks(cluster_id);

-- Cluster indexes
CREATE INDEX IF NOT EXISTS idx_cluster_assignments_block ON cluster_assignments(block_id);
CREATE INDEX IF NOT EXISTS idx_cluster_assignments_cluster ON cluster_assignments(cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_size ON clusters(size);

-- Gene-block mapping indexes
CREATE INDEX IF NOT EXISTS idx_gene_blocks_gene ON gene_block_mappings(gene_id);
CREATE INDEX IF NOT EXISTS idx_gene_blocks_block ON gene_block_mappings(block_id);
CREATE INDEX IF NOT EXISTS idx_gene_blocks_role ON gene_block_mappings(block_role);

-- PFAM search indexes
CREATE INDEX IF NOT EXISTS idx_pfam_domains_name ON pfam_domains(pfam_name);
CREATE INDEX IF NOT EXISTS idx_pfam_domains_count ON pfam_domains(protein_count);

-- Full-text search for PFAM domains in genes (for domain search)
CREATE VIRTUAL TABLE IF NOT EXISTS genes_pfam_fts USING fts5(
    gene_id, pfam_domains, content='genes', content_rowid='rowid'
);
"""

def create_database(db_path: Path, force: bool = False) -> None:
    """
    Create SQLite database with optimized schema for genome browser.
    
    Args:
        db_path: Path to SQLite database file
        force: If True, recreate database even if it exists
    """
    if db_path.exists() and not force:
        logger.info(f"Database already exists: {db_path}")
        return
    
    if force and db_path.exists():
        logger.info(f"Removing existing database: {db_path}")
        db_path.unlink()
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating database: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set performance optimizations
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size = 10000")  # 10MB cache
        conn.execute("PRAGMA temp_store = MEMORY")  # Keep temp tables in memory
        
        # Create schema
        logger.info("Creating database schema...")
        conn.executescript(DATABASE_SCHEMA)
        
        # Create indexes
        logger.info("Creating performance indexes...")
        conn.executescript(PERFORMANCE_INDEXES)
        
        # Analyze tables for query optimization
        conn.execute("ANALYZE")
        
        conn.commit()
    
    logger.info(f"Database created successfully: {db_path}")

def get_database_info(db_path: Path) -> dict:
    """Get information about the database."""
    if not db_path.exists():
        return {"status": "not_exists"}
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get table counts
        tables_info = {}
        tables = ['genomes', 'contigs', 'genes', 'syntenic_blocks', 'clusters', 
                 'cluster_assignments', 'pfam_domains', 'gene_block_mappings', 'annotation_stats']
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                tables_info[table] = count
            except sqlite3.OperationalError:
                tables_info[table] = "table_not_exists"
        
        # Get database size
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            "status": "exists",
            "size_mb": round(size_mb, 2),
            "tables": tables_info
        }

def vacuum_database(db_path: Path) -> None:
    """Optimize database by vacuuming and reanalyzing."""
    logger.info(f"Optimizing database: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        conn.commit()
    
    logger.info("Database optimization complete")

def main():
    """Command line interface for database setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up ELSA genome browser database")
    parser.add_argument("--db-path", type=Path, default="genome_browser.db",
                       help="Path to SQLite database file")
    parser.add_argument("--force", action="store_true",
                       help="Recreate database even if it exists")
    parser.add_argument("--info", action="store_true",
                       help="Show database information")
    parser.add_argument("--vacuum", action="store_true",
                       help="Optimize existing database")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.info:
        info = get_database_info(args.db_path)
        print(f"\nDatabase Info: {args.db_path}")
        print(f"Status: {info['status']}")
        if info['status'] == 'exists':
            print(f"Size: {info['size_mb']} MB")
            print("\nTable Counts:")
            for table, count in info['tables'].items():
                print(f"  {table}: {count}")
    
    elif args.vacuum:
        if not args.db_path.exists():
            print(f"Database does not exist: {args.db_path}")
            return
        vacuum_database(args.db_path)
        print("Database optimized successfully")
    
    else:
        create_database(args.db_path, force=args.force)
        info = get_database_info(args.db_path)
        print(f"\nDatabase created: {args.db_path}")
        print(f"Size: {info['size_mb']} MB")

if __name__ == "__main__":
    main()