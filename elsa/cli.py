"""
ELSA command-line interface.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .params import ELSAConfig, load_config, create_default_config
from . import __version__

console = Console()


def print_banner():
    """Print ELSA banner."""
    console.print(f"""
[bold blue]ELSA[/bold blue] [dim]v{__version__}[/dim]
[italic]Embedding Locus Shingle Alignment[/italic]
""")


def setup_genome_browser_integration(config: ELSAConfig, analysis_output_dir: Path, 
                                   genome_browser_db: str, sequences_dir: Optional[str], 
                                   proteins_dir: Optional[str]) -> bool:
    """Set up genome browser with analysis results and existing PFAM annotations."""
    
    # Find the genome browser setup script
    genome_browser_dir = Path(__file__).parent.parent / "genome_browser"
    setup_script = genome_browser_dir / "setup_genome_browser.py"
    
    if not setup_script.exists():
        console.print(f"[red]Genome browser setup script not found: {setup_script}[/red]")
        return False
    
    # Check for required analysis output files
    blocks_file = analysis_output_dir / "syntenic_blocks.csv"
    clusters_file = analysis_output_dir / "syntenic_clusters.csv"
    
    if not blocks_file.exists():
        console.print(f"[red]Syntenic blocks file not found: {blocks_file}[/red]")
        return False
    
    if not clusters_file.exists():
        console.print(f"[red]Syntenic clusters file not found: {clusters_file}[/red]")
        return False
    
    # Auto-detect sequences and proteins directories if not provided
    if not sequences_dir or not proteins_dir:
        console.print("Auto-detecting genome data directories...")
        
        # Look for directory patterns - prioritize organized structure
        cwd = Path.cwd()
        
        # Use organized structure only - fail if not found
        sequences_patterns = ["data/genomes"]
        proteins_patterns = ["data/proteins"]
        
        if not sequences_dir:
            for candidate in sequences_patterns:
                candidate_path = cwd / candidate
                if candidate_path.exists():
                    fna_files = list(candidate_path.glob("*.fna"))
                    if fna_files:
                        sequences_dir = str(candidate_path.absolute())
                        console.print(f"Found nucleotide sequences: {sequences_dir} ({len(fna_files)} .fna files)")
                        break
        
        if not proteins_dir:
            for candidate in proteins_patterns:
                candidate_path = cwd / candidate
                if candidate_path.exists():
                    faa_files = list(candidate_path.glob("*.faa"))
                    if faa_files:
                        proteins_dir = str(candidate_path.absolute())
                        console.print(f"Found protein sequences: {proteins_dir} ({len(faa_files)} .faa files)")
                        break
    
    if not sequences_dir or not proteins_dir:
        console.print("[yellow]Could not auto-detect genome directories.[/yellow]")
        console.print(f"[dim]  sequences_dir: {sequences_dir}[/dim]")
        console.print(f"[dim]  proteins_dir: {proteins_dir}[/dim]")
        console.print(f"[dim]  current working directory: {Path.cwd()}[/dim]")
        
        # Show what directories exist for debugging
        cwd = Path.cwd()
        console.print("[dim]Available directories:[/dim]")
        all_patterns = list(set(sequences_patterns + proteins_patterns))
        for candidate in all_patterns:
            candidate_path = cwd / candidate
            if candidate_path.exists():
                fna_count = len(list(candidate_path.glob("*.fna")))
                faa_count = len(list(candidate_path.glob("*.faa")))
                console.print(f"[dim]  {candidate}: {fna_count} .fna, {faa_count} .faa files[/dim]")
        
        console.print("[yellow]Specify --sequences-dir and --proteins-dir for genome browser setup[/yellow]")
        return False
    
    # Check if directories exist and contain expected files (convert to absolute paths)
    sequences_path = Path(sequences_dir).absolute()
    proteins_path = Path(proteins_dir).absolute()
    
    if not sequences_path.exists():
        console.print(f"[red]Sequences directory not found: {sequences_path}[/red]")
        return False
    
    if not proteins_path.exists():
        console.print(f"[red]Proteins directory not found: {proteins_path}[/red]")
        return False
    
    # Check for existing PFAM annotations from elsa embed
    pfam_results_file = genome_browser_dir / "pfam_annotations" / "pfam_annotation_results.json"
    
    # Build setup command (use absolute paths for all files)
    cmd = [
        sys.executable, str(setup_script),
        "--sequences-dir", str(sequences_path),
        "--proteins-dir", str(proteins_path),
        "--blocks-file", str(blocks_file.absolute()),
        "--clusters-file", str(clusters_file.absolute()),
        "--db-path", str(Path(genome_browser_db).absolute()),
        "--force"  # Force regeneration to pick up new analysis results
    ]
    
    # Use existing PFAM annotations if available
    if pfam_results_file.exists():
        console.print(f"Using existing PFAM annotations: {pfam_results_file}")
        cmd.extend(["--skip-pfam", "--pfam-output", str(pfam_results_file.parent.absolute())])  # Skip PFAM regeneration but use existing results
    else:
        console.print("No existing PFAM annotations found - will generate fresh annotations")
    
    console.print(f"Setting up genome browser database: {genome_browser_db}")
    
    try:
        # Stream logs live to console so long-running steps don't appear hung
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=genome_browser_dir  # Run from genome_browser directory
        )
        console.print("✓ Genome browser setup completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Genome browser setup failed (return code {e.returncode})[/red]")
        # When capture_output=False, stdout/stderr already streamed; still show summaries if available
        if e.stdout:
            console.print(f"STDOUT: {e.stdout}")
        if e.stderr:
            console.print(f"STDERR: {e.stderr}")
        return False


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def main(ctx, version):
    """ELSA: Order-aware syntenic-block discovery from protein embeddings."""
    if version:
        click.echo(f"ELSA v{__version__}")
        sys.exit(0)
    
    if ctx.invoked_subcommand is None:
        print_banner()
        click.echo(ctx.get_help())


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml", 
              help="Configuration file path", type=click.Path())
@click.option("--force", is_flag=True, help="Overwrite existing files")
def init(config: str, force: bool):
    """Initialize ELSA workspace with default configuration."""
    console.print("[bold]Initializing ELSA workspace...[/bold]")
    
    config_path = Path(config)
    
    # Check if files exist
    if config_path.exists() and not force:
        console.print(f"[red]Configuration file already exists: {config_path}[/red]")
        console.print("Use --force to overwrite")
        sys.exit(1)
    
    # Create default config
    create_default_config(config_path)
    console.print(f"✓ Created configuration: [green]{config_path}[/green]")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Put genome FASTA files in a directory (e.g., data/)")
    console.print("2. Run: [cyan]elsa embed data/[/cyan]")
    console.print("3. Run: [cyan]elsa build[/cyan]")


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--fasta-pattern", default="*.fasta", help="Pattern for nucleotide FASTA files (default: *.fasta,*.fa,*.fna)")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
def embed(config: str, input_dir: str, fasta_pattern: str, resume: bool):
    """Generate protein embeddings from nucleotide FASTA files in a directory.
    
    This command runs the complete pipeline on all nucleotide FASTA files:
    1. Gene calling with Prodigal to identify protein-coding sequences
    2. Translation to amino acid sequences  
    3. Protein language model embedding (ESM2/ProtT5)
    4. PCA projection and window shingling for syntenic analysis
    """
    console.print("[bold]ELSA Embedding Pipeline[/bold]")
    
    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    input_path = Path(input_dir)
    
    # Auto-discover nucleotide FASTA files with multiple extensions
    fasta_files = []
    if fasta_pattern == "*.fasta":  # Default case - check all common extensions
        extensions = ["*.fasta", "*.fa", "*.fna"]
    else:
        extensions = [fasta_pattern]
    
    for pattern in extensions:
        fasta_files.extend(input_path.glob(pattern))
    
    fasta_files = sorted(set(fasta_files))  # Remove duplicates and sort
    
    if not fasta_files:
        console.print(f"[red]No FASTA files found in {input_dir}[/red]")
        console.print(f"Searched for: {', '.join(extensions)}")
        sys.exit(1)
    
    console.print(f"Found {len(fasta_files)} nucleotide FASTA files:")
    for f in fasta_files:
        console.print(f"  - {f.name}")
    
    # Create sample data - nucleotide FASTA only, no GFF or AA files
    sample_data = []
    for fasta_file in fasta_files:
        sample_id = fasta_file.stem
        sample_data.append((sample_id, fasta_file, None, None))  # No GFF, no AA files
    
    # Show configuration and discovered files
    console.print(f"\nGene caller: [green]{config_obj.ingest.gene_caller}[/green]")
    console.print(f"PLM Model: [green]{config_obj.plm.model}[/green]")
    console.print(f"Device: [green]{config_obj.plm.device}[/green]")
    console.print(f"Target dimension: [green]{config_obj.plm.project_to_D}[/green]")
    console.print(f"\nProcessing {len(sample_data)} samples with gene calling:")
    
    for sample_id, fasta_file, _, _ in sample_data:
        console.print(f"  - [cyan]{sample_id}[/cyan]: {fasta_file.name} → [dim]Prodigal gene calling[/dim]")
    
    # Run the pipeline
    try:
        from .ingest import ProteinIngester
        from .embeddings import ProteinEmbedder
        from .manifest import ELSAManifest
        
        # Initialize components
        manifest = ELSAManifest(config_obj.data.work_dir)
        manifest.set_config(config_obj)
        
        ingester = ProteinIngester(config_obj.ingest)
        # Use sliding window to avoid truncation
        from .embeddings import AggregationStrategy
        embedder = ProteinEmbedder(
            config_obj.plm, 
            window_size=1024, 
            overlap=256, 
            aggregation=AggregationStrategy.MAX_POOL
        )
        
        # Stage 1: Protein ingestion (gene calling/translation)
        console.print("\n[bold blue]Stage 1: Protein Ingestion[/bold blue]")
        sample_proteins = ingester.ingest_multiple(sample_data)
        
        # Stage 1.5: PFAM domain annotation (if enabled)
        if config_obj.ingest.run_pfam:
            console.print("\n[bold blue]Stage 1.5: PFAM Domain Annotation[/bold blue]")
            
            # Create standard PFAM output directory in genome_browser
            pfam_output_dir = Path("genome_browser/pfam_annotations")
            # Use system thread count from config - resolve "auto" to actual CPU count
            if isinstance(config_obj.system.jobs, int):
                threads = config_obj.system.jobs
            else:  # "auto"
                import os
                threads = os.cpu_count() or 4
            pfam_results_file = ingester.run_pfam_annotation(pfam_output_dir, threads)
                
            if pfam_results_file:
                console.print(f"✓ PFAM annotation completed: {pfam_results_file}")
            else:
                console.print("[yellow]PFAM annotation skipped or failed[/yellow]")
        else:
            console.print("\n[dim]Stage 1.5: PFAM annotation disabled[/dim]")
        
        # Stage 2: Embedding generation  
        console.print("\n[bold blue]Stage 2: Protein Embedding[/bold blue]")
        all_proteins = []
        for proteins in sample_proteins.values():
            all_proteins.extend(proteins)
        
        console.print(f"Total proteins to embed: {len(all_proteins):,}")
        
        # Generate embeddings
        embeddings = list(embedder.embed_sequences(all_proteins))
        
        console.print(f"\n✓ Generated {len(embeddings):,} embeddings")
        console.print(f"Embedding dimension: {embedder.embedding_dim}")
        
        # Stage 3: PCA projection and storage
        console.print("\n[bold blue]Stage 3: PCA Projection & Storage[/bold blue]")
        from .projection import ProjectionSystem
        
        projection_system = ProjectionSystem(config_obj.plm, config_obj.data.work_dir, manifest)
        
        # Create protein metadata for coordinate preservation
        protein_metadata = {}
        for proteins in sample_proteins.values():
            for protein in proteins:
                protein_metadata[protein.gene_id] = {
                    'contig_id': protein.contig_id,
                    'start': protein.start,
                    'end': protein.end,
                    'strand': protein.strand
                }
        
        # Process embeddings: fit PCA, project, and save
        projected_proteins = projection_system.process_embeddings(embeddings, protein_metadata)
        
        # Show projection statistics
        stats = projection_system.get_projection_stats()
        if stats:
            console.print(f"Explained variance: {stats['total_explained_variance']:.3f}")
            console.print(f"Projection: {stats['input_dim']}D → {stats['output_dim']}D")
        
        # Stage 4: Shingling (Window Embeddings)
        console.print("\n[bold blue]Stage 4: Window Shingling[/bold blue]")
        from .shingling import ShingleSystem
        
        shingle_system = ShingleSystem(config_obj.shingles, config_obj.data.work_dir, manifest)
        windows = shingle_system.process_proteins(projected_proteins)
        
        if windows:  # Only if we created new windows
            window_stats = shingle_system.get_window_stats(windows)
            console.print(f"Window size: {config_obj.shingles.n}, stride: {config_obj.shingles.stride}")
            console.print(f"Created {window_stats.get('total_windows', 0):,} windows from {window_stats.get('n_loci', 0)} loci")
        
        console.print(f"\n[bold green]Embedding pipeline completed![/bold green]")
        console.print(f"Ready for index building with: [cyan]elsa build[/cyan]")
        
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
def build(config: str, resume: bool):
    """Build discrete and continuous indexes."""
    console.print("[bold]Building indexes...[/bold]")
    
    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    # Run the indexing pipeline
    try:
        from .manifest import ELSAManifest
        from .indexing import IndexBuilder
        
        # Initialize components
        manifest = ELSAManifest(config_obj.data.work_dir)
        
        # Check if embeddings and shingles exist
        if not manifest.has_artifact('windows'):
            console.print("[red]No shingles found! Run 'elsa embed' first.[/red]")
            sys.exit(1)
        
        # Build indexes
        index_builder = IndexBuilder(config_obj, manifest)
        index_builder.build_indexes(resume=resume)
        
        console.print("[green]✓ Index building completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.argument("query_locus")
@click.option("--target-scope", default="all", help="Target search scope")
@click.option("--output", "-o", help="Output file for results")
def find(config: str, query_locus: str, target_scope: str, output: Optional[str]):
    """Find syntenic blocks similar to query locus."""
    console.print(f"[bold]Searching for blocks similar to: [green]{query_locus}[/green][/bold]")
    
    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    # Run the search pipeline
    try:
        from .manifest import ELSAManifest
        from .search import SearchEngine
        from dataclasses import asdict
        import json
        
        # Initialize components
        manifest = ELSAManifest(config_obj.data.work_dir)
        
        # Check if indexes exist
        if not manifest.has_artifact('indexes/discrete_index.json'):
            console.print("[red]No indexes found! Run 'elsa build' first.[/red]")
            sys.exit(1)
        
        # Execute search
        search_engine = SearchEngine(config_obj, manifest)
        blocks = search_engine.search(query_locus, max_results=50)
        
        if not blocks:
            console.print("[yellow]No syntenic blocks found for the query.[/yellow]")
            return
        
        # Display results
        console.print(f"\n[bold green]Found {len(blocks)} syntenic blocks:[/bold green]")
        
        for i, block in enumerate(blocks[:10], 1):  # Show top 10
            console.print(f"\n[bold]{i}. {block.target_locus}[/bold]")
            console.print(f"   Score: {block.chain_score:.3f}")
            console.print(f"   Identity: {block.identity:.1%}")
            console.print(f"   Length: {block.alignment_length} windows")
            console.print(f"   Windows: {len(block.query_windows)} query ↔ {len(block.target_windows)} target")
        
        # Save results if output specified
        if output:
            results_data = {
                'query_locus': query_locus,
                'total_blocks': len(blocks),
                'blocks': [asdict(block) for block in blocks]
            }
            
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            console.print(f"\n✓ Results saved to: {output_path}")
        
        console.print(f"\n[green]✓ Search completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.option("--min-windows", default=3, help="Minimum windows per locus")
@click.option("--min-similarity", default=0.7, help="Minimum similarity threshold")
@click.option("--output-dir", "-o", default="syntenic_analysis", help="Output directory")
@click.option("--setup-genome-browser/--no-setup-genome-browser", default=True,
              help="Set up genome browser after analysis (use --no-setup-genome-browser to skip)")
@click.option("--genome-browser-db", default="genome_browser/genome_browser.db", help="Genome browser database path")
@click.option("--sequences-dir", help="Directory containing nucleotide sequences (.fna)")
@click.option("--proteins-dir", help="Directory containing proteins (.faa)")
@click.option("--micro/--no-micro", default=False, help="Run micro-synteny (embedding-first, sidecar-only)")
def analyze(config: str, min_windows: int, min_similarity: float, output_dir: str, 
           setup_genome_browser: bool, genome_browser_db: str, sequences_dir: Optional[str], proteins_dir: Optional[str], micro: bool):
    """Perform comprehensive syntenic block analysis of entire dataset."""
    console.print("[bold]ELSA Comprehensive Syntenic Analysis[/bold]")
    
    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    # Run comprehensive analysis
    try:
        from .manifest import ELSAManifest
        from .analysis import SyntenicAnalyzer
        
        # Initialize components
        manifest = ELSAManifest(config_obj.data.work_dir)
        
        # Check if required data exists
        if not manifest.has_artifact('windows'):
            console.print("[red]No window data found! Run 'elsa embed' first.[/red]")
            sys.exit(1)
        
        # Execute comprehensive analysis
        analyzer = SyntenicAnalyzer(config_obj, manifest)
        landscape = analyzer.analyze(min_windows=min_windows, min_similarity=min_similarity)
        
        # Display summary
        console.print(f"\n[bold green]Analysis Complete![/bold green]")
        console.print(f"📊 Dataset Overview:")
        console.print(f"  • {landscape.total_loci:,} loci analyzed")
        console.print(f"  • {landscape.total_blocks:,} syntenic blocks found")
        console.print(f"  • {landscape.total_clusters:,} clusters identified")
        
        console.print(f"\n📈 Block Statistics:")
        stats = landscape.statistics['blocks']
        console.print(f"  • Mean length: {stats['mean_length']:.1f} windows")
        console.print(f"  • Mean identity: {stats['mean_identity']:.1%}")
        console.print(f"  • Length range: {stats['length_distribution']['min']}-{stats['length_distribution']['max']} windows")
        
        if landscape.total_clusters > 0:
            console.print(f"\n🧬 Cluster Statistics:")
            cluster_stats = landscape.statistics['clusters'] 
            console.print(f"  • Mean cluster size: {cluster_stats['mean_cluster_size']:.1f} blocks")
            console.print(f"  • Mean diversity: {cluster_stats['mean_diversity']:.3f}")
        
        # Save results
        output_path = Path(output_dir)
        analyzer.save_results(landscape, output_path)

        # Optional: post-clustering attach stage (PFAM-agnostic), driven by config
        attach_cfg = getattr(config_obj.analyze, 'attach', None)
        if attach_cfg and getattr(attach_cfg, 'enable', False):
            try:
                console.print("\n[bold]Post-process: Attaching sink blocks to clusters...[/bold]")
                blocks_csv = str(output_path / 'syntenic_blocks.csv')
                cmd = [
                    sys.executable, 'tools/attach_by_cluster_signatures.py',
                    '--config', str(Path(config).absolute()),
                    '--blocks', blocks_csv,
                    '--out', blocks_csv,
                    '--member_sample', str(attach_cfg.member_sample),
                    '--k1_method', str(attach_cfg.k1_method),
                    '--icws_r', str(attach_cfg.icws_r),
                    '--icws_bbit', str(attach_cfg.icws_bbit),
                    '--bandset_contain_tau', str(attach_cfg.bandset_contain_tau),
                    '--k1_contain_tau', str(attach_cfg.k1_contain_tau),
                    '--k1_inter_min', str(attach_cfg.k1_inter_min),
                    '--margin_min', str(attach_cfg.margin_min),
                    '--triangle_min', str(attach_cfg.triangle_min),
                    '--triangle_member_tau', str(attach_cfg.triangle_member_tau),
                    '--tiny_window_cap', str(attach_cfg.tiny_window_cap),
                    '--bandset_contain_tau_tiny', str(attach_cfg.bandset_contain_tau_tiny),
                    '--k1_contain_tau_tiny', str(attach_cfg.k1_contain_tau_tiny),
                    '--k1_inter_min_tiny', str(attach_cfg.k1_inter_min_tiny),
                    '--margin_min_tiny', str(attach_cfg.margin_min_tiny),
                    '--triangle_min_tiny', str(attach_cfg.triangle_min_tiny),
                    '--triangle_member_tau_tiny', str(attach_cfg.triangle_member_tau_tiny),
                ]
                if getattr(attach_cfg, 'enable_stitch', True):
                    cmd += ['--enable_stitch', '--stitch_gap', str(attach_cfg.stitch_gap), '--stitch_max_neighbors', str(attach_cfg.stitch_max_neighbors)]
                if getattr(attach_cfg, 'load_signatures', None):
                    cmd += ['--load_signatures', str(Path(attach_cfg.load_signatures))]
                    if getattr(attach_cfg, 'limit_member_sample', 0):
                        cmd += ['--limit_member_sample', str(attach_cfg.limit_member_sample)]

                # Run attach stage in repo root
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                console.print(result.stdout)
            except subprocess.CalledProcessError as e:
                console.print("[yellow]Attachment stage failed; proceeding without it.[/yellow]")
                if e.stdout:
                    console.print(f"STDOUT: {e.stdout}")
                if e.stderr:
                    console.print(f"STDERR: {e.stderr}")

        console.print(f"\n[green]✓ Comprehensive analysis completed successfully![/green]")
        
        # Set up genome browser if requested
        if setup_genome_browser:
            console.print(f"\n[bold blue]Setting up genome browser...[/bold blue]")
            try:
                success = setup_genome_browser_integration(
                    config_obj, output_path, genome_browser_db, 
                    sequences_dir, proteins_dir
                )
                if success:
                    console.print(f"\n[green]✓ Genome browser setup completed![/green]")
                    console.print(f"Database: {genome_browser_db}")
                    console.print(f"To start the browser: cd genome_browser && streamlit run app.py")
                    # If micro sidecar exists, import into primary tables and precompute consensus
                    if micro:
                        try:
                            import sqlite3, pandas as pd
                            from pathlib import Path as _P
                            conn = sqlite3.connect(str(genome_browser_db))
                            # Ensure micro CSVs exist
                            mdir = _P(output_path) / 'micro_gene'
                            mb_csv = mdir / 'micro_gene_blocks.csv'
                            mc_csv = mdir / 'micro_gene_clusters.csv'
                            if mb_csv.exists() and mc_csv.exists():
                                # Load micro CSVs
                                mb_df = pd.read_csv(mb_csv)
                                mc_df = pd.read_csv(mc_csv)
                                # Compute macro cluster offset (append after current max)
                                cur = conn.cursor()
                                row = cur.execute("SELECT COALESCE(MAX(cluster_id),0) FROM clusters").fetchone()
                                start_cid = int(row[0] or 0)
                                # Map only non-sink micro clusters (>0) and offset by number of macro clusters
                                non_sink = mc_df[mc_df['cluster_id'] > 0].sort_values('cluster_id')
                                cid_map = {int(r.cluster_id): (start_cid + int(r.cluster_id)) for r in non_sink.itertuples(index=False)}
                                # Insert clusters
                                rows = []
                                for r in non_sink.itertuples(index=False):
                                    rows.append((cid_map[int(r.cluster_id)], int(r.size), None, None, None, None, None, 'micro'))
                                cur.executemany(
                                    """
                                    INSERT OR REPLACE INTO clusters
                                        (cluster_id, size, consensus_length, consensus_score, diversity, representative_query, representative_target, cluster_type)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    rows,
                                )
                                # Insert micro blocks into syntenic_blocks with new block IDs
                                # Use high offset to avoid collisions
                                sb_rows = []
                                for r in mb_df.itertuples(index=False):
                                    new_bid = 1000000000 + int(r.block_id)
                                    q_locus = f"{r.genome_id}:{r.contig_id}#" + str(int(r.start_index)) + "-" + str(int(r.end_index))
                                    # Assign cluster: keep sink (0) as 0; offset non-sink via cid_map
                                    src_cid = int(r.cluster_id)
                                    dest_cid = 0 if src_cid == 0 else cid_map.get(src_cid, 0)
                                    sb_rows.append((
                                        new_bid,
                                        dest_cid,
                                        q_locus,
                                        '',
                                        str(r.genome_id),
                                        None,
                                        str(r.contig_id),
                                        None,
                                        int(1 + int(r.end_index) - int(r.start_index)),
                                        0.0,
                                        0.0,
                                        None, None,
                                        int(r.start_index), int(r.end_index),
                                        None, None,
                                        None, None,
                                        'micro'
                                    ))
                                cur.executemany(
                                    """
                                    INSERT OR REPLACE INTO syntenic_blocks (
                                        block_id, cluster_id, query_locus, target_locus,
                                        query_genome_id, target_genome_id,
                                        query_contig_id, target_contig_id,
                                        length, identity, score,
                                        n_query_windows, n_target_windows,
                                        query_window_start, query_window_end,
                                        target_window_start, target_window_end,
                                        query_windows_json, target_windows_json,
                                        block_type
                                    ) VALUES (
                                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                                    )
                                    """,
                                    sb_rows,
                                )
                                # Insert gene_block_mappings for micro blocks (query role)
                                # Build per-contig ordered gene lists cache
                                import collections
                                cache = {}
                                gbm_rows = []
                                for r in mb_df.itertuples(index=False):
                                    gid = str(r.genome_id); cid = str(r.contig_id)
                                    key = (gid, cid)
                                    if key not in cache:
                                        genes = [row[0] for row in conn.execute(
                                            "SELECT gene_id FROM genes WHERE genome_id = ? AND contig_id = ? ORDER BY start_pos, end_pos",
                                            (gid, cid)
                                        ).fetchall()]
                                        cache[key] = genes
                                    genes = cache[key]
                                    s = int(r.start_index); e = int(r.end_index)
                                    s = max(0, min(len(genes)-1, s)); e = max(0, min(len(genes)-1, e))
                                    if e < s:
                                        continue
                                    span = genes[s:e+1]
                                    L = max(1, len(span)-1)
                                    new_bid = 1000000000 + int(r.block_id)
                                    for idx, gene_id in enumerate(span):
                                        rel = (idx / L) if L > 0 else 0.0
                                        gbm_rows.append((gene_id, new_bid, 'query', rel))
                                # Ensure table exists
                                cur.execute(
                                    """
                                    CREATE TABLE IF NOT EXISTS gene_block_mappings (
                                        gene_id TEXT,
                                        block_id INTEGER,
                                        block_role TEXT,
                                        relative_position REAL,
                                        PRIMARY KEY (gene_id, block_id)
                                    )
                                    """
                                )
                                # Insert mappings in chunks
                                for i in range(0, len(gbm_rows), 1000):
                                    cur.executemany(
                                        "INSERT OR REPLACE INTO gene_block_mappings (gene_id, block_id, block_role, relative_position) VALUES (?, ?, ?, ?)",
                                        gbm_rows[i:i+1000]
                                    )
                                conn.commit()

                                # Also project paired micro alignments into syntenic_blocks and gene_block_mappings
                                try:
                                    mbp_csv = mdir / 'micro_block_pairs.csv'
                                    mgpm_csv = mdir / 'micro_gene_pair_mappings.csv'
                                    if mbp_csv.exists() and mgpm_csv.exists():
                                        mbp_df = pd.read_csv(mbp_csv)
                                        mgpm_df = pd.read_csv(mgpm_csv)
                                        if not mbp_df.empty:
                                            # Build counts per role for window stats
                                            counts = mgpm_df.groupby(['block_id','block_role']).size().unstack(fill_value=0)
                                            n_q = counts.get('query', pd.Series(dtype=int))
                                            n_t = counts.get('target', pd.Series(dtype=int))
                                            sb_pair_rows = []
                                            for r in mbp_df.itertuples(index=False):
                                                src_cid = int(r.cluster_id)
                                                dest_cid = 0 if src_cid == 0 else cid_map.get(src_cid, 0)
                                                q_locus = f"{r.query_genome_id}:{r.query_contig_id}:{int(r.query_start_bp)}-{int(r.query_end_bp)}"
                                                t_locus = f"{r.target_genome_id}:{r.target_contig_id}:{int(r.target_start_bp)}-{int(r.target_end_bp)}"
                                                q_len = int(r.query_end_bp) - int(r.query_start_bp)
                                                t_len = int(r.target_end_bp) - int(r.target_start_bp)
                                                length = q_len if q_len >= t_len else t_len
                                                nqw = int(n_q.get(int(r.block_id), 0)) if isinstance(n_q, pd.Series) else 0
                                                ntw = int(n_t.get(int(r.block_id), 0)) if isinstance(n_t, pd.Series) else 0
                                                sb_pair_rows.append((
                                                    int(r.block_id),
                                                    dest_cid,
                                                    q_locus,
                                                    t_locus,
                                                    str(r.query_genome_id),
                                                    str(r.target_genome_id),
                                                    str(r.query_contig_id),
                                                    str(r.target_contig_id),
                                                    int(length),
                                                    float(r.identity) if pd.notnull(r.identity) else 0.0,
                                                    float(r.score) if pd.notnull(r.score) else 0.0,
                                                    nqw, ntw,
                                                    None, None,
                                                    None, None,
                                                    None, None,
                                                    'micro'
                                                ))
                                            if sb_pair_rows:
                                                cur.executemany(
                                                    """
                                                    INSERT OR REPLACE INTO syntenic_blocks (
                                                        block_id, cluster_id, query_locus, target_locus,
                                                        query_genome_id, target_genome_id,
                                                        query_contig_id, target_contig_id,
                                                        length, identity, score,
                                                        n_query_windows, n_target_windows,
                                                        query_window_start, query_window_end,
                                                        target_window_start, target_window_end,
                                                        query_windows_json, target_windows_json,
                                                        block_type
                                                    ) VALUES (
                                                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                                                    )
                                                    """,
                                                    sb_pair_rows,
                                                )
                                            # Insert gene mappings for both roles with relative positions
                                            gbm_rows2 = []
                                            for (blk, role), g in mgpm_df.groupby(['block_id','block_role']):
                                                g2 = g.sort_values(['start_pos','end_pos'])
                                                n = len(g2)
                                                L = max(1, n-1)
                                                for idx, rr in enumerate(g2.itertuples(index=False)):
                                                    rel = (idx / L) if L > 0 else 0.0
                                                    gbm_rows2.append((str(rr.gene_id), int(blk), str(role), float(rel)))
                                            for i in range(0, len(gbm_rows2), 1000):
                                                cur.executemany(
                                                    "INSERT OR REPLACE INTO gene_block_mappings (gene_id, block_id, block_role, relative_position) VALUES (?, ?, ?, ?)",
                                                    gbm_rows2[i:i+1000]
                                                )
                                            conn.commit()
                                except Exception as e:
                                    # Non-fatal; pairs projection can be re-run later
                                    pass

                                # Precompute macro-style consensus for new micro clusters into cluster_consensus
                                try:
                                    # Robust import by file path
                                    import importlib.util as _ilu
                                    from pathlib import Path as _P
                                    mod_path = _P(__file__).parents[1] / 'genome_browser' / 'database' / 'cluster_content.py'
                                    spec = _ilu.spec_from_file_location('gb_cluster_content', str(mod_path))
                                    if spec and spec.loader:
                                        gbmod = _ilu.module_from_spec(spec)
                                        spec.loader.exec_module(gbmod)  # type: ignore
                                        compute_cluster_pfam_consensus = getattr(gbmod, 'compute_cluster_pfam_consensus')  # type: ignore
                                        # Create table if missing
                                        conn.execute("""
                                            CREATE TABLE IF NOT EXISTS cluster_consensus (
                                                cluster_id INTEGER PRIMARY KEY,
                                                consensus_json TEXT,
                                                agree_frac REAL,
                                                core_tokens INTEGER
                                            )
                                        """)
                                        import json as _json
                                        for old_cid, new_cid in cid_map.items():
                                            payload = compute_cluster_pfam_consensus(conn, int(new_cid), 0.5, 0.9, 12)  # type: ignore
                                            conn.execute(
                                                "INSERT OR REPLACE INTO cluster_consensus(cluster_id, consensus_json, agree_frac, core_tokens) VALUES (?, ?, ?, ?)",
                                                (int(new_cid), _json.dumps(payload), None, None)
                                            )
                                        conn.commit()
                                except Exception as e:
                                    console.print(f"[dim]Macro-style consensus precompute skipped for micro clusters: {e}[/dim]")
                        except Exception as e:
                            console.print(f"[yellow]Micro projection into browser DB failed: {e}[/yellow]")
                else:
                    console.print(f"\n[yellow]⚠ Genome browser setup skipped (see messages above)[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Genome browser setup failed: {e}[/red]")
                console.print("[yellow]Analysis results are still available - you can set up the browser manually[/yellow]")

        # Optional: run micro-synteny sidecar pipeline
        if micro:
            try:
                console.print("\n[bold blue]Running micro-synteny (embedding-first) sidecar...[/bold blue]")
                # Resolve genes.parquet
                from .manifest import ELSAManifest
                m = ELSAManifest(config_obj.data.work_dir)
                genes_path = None
                if m.has_artifact('genes'):
                    genes_path = Path(m.data['artifacts']['genes']['path'])
                else:
                    # Fallback common locations
                    candidates = [
                        Path('elsa_index/ingest/genes.parquet'),
                        Path.cwd() / 'elsa_index/ingest/genes.parquet',
                        Path(__file__).parents[2] / 'elsa_index/ingest/genes.parquet',
                    ]
                    for p in candidates:
                        if p.exists():
                            genes_path = p
                            break
                if not genes_path or not Path(genes_path).exists():
                    console.print("[yellow]genes.parquet not found; skipping micro-synteny[/yellow]")
                else:
                    from .analyze.micro_gene import run_micro_clustering
                    macro_cl = getattr(config_obj.analyze, 'clustering', None)
                    overrides = getattr(config_obj.analyze, 'micro_overrides', None)
                    # Base defaults derived from macro or sensible micro defaults
                    micro_j = float(getattr(macro_cl, 'jaccard_tau', 0.75)) if macro_cl else 0.75
                    micro_mk = int(getattr(macro_cl, 'mutual_k', 3)) if macro_cl else 3
                    micro_df = int(getattr(macro_cl, 'df_max', 30)) if macro_cl else 30
                    micro_gs = 3
                    # Apply micro-specific overrides if provided in config
                    try:
                        if overrides and getattr(overrides, 'jaccard_tau', None) is not None:
                            micro_j = float(overrides.jaccard_tau)
                        if overrides and getattr(overrides, 'mutual_k', None) is not None:
                            micro_mk = int(overrides.mutual_k)
                        if overrides and getattr(overrides, 'df_max', None) is not None:
                            micro_df = int(overrides.df_max)
                        if overrides and getattr(overrides, 'min_genome_support', None) is not None:
                            micro_gs = int(overrides.min_genome_support)
                    except Exception:
                        pass
                    micro_out = Path(output_path) / 'micro_gene'
                    db_path = Path(genome_browser_db) if setup_genome_browser and Path(genome_browser_db).exists() else None
                    # Defaults per AGENTS.md
                    blocks_df, clusters_df = run_micro_clustering(
                        Path(genes_path),
                        micro_out,
                        db_path=db_path,
                        k_values=(2, 3),
                        max_gap=1,
                        jaccard_tau=micro_j,
                        mutual_k=micro_mk,
                        df_max=micro_df,
                        min_genome_support=micro_gs,
                    )
                    # Post-write deduplication against macro spans (both-side containment)
                    if db_path and db_path.exists():
                        try:
                            from .analyze.micro_gene import deduplicate_micro_against_macro
                            stats = deduplicate_micro_against_macro(Path(db_path))
                            del_n = int(stats.get('deleted_blocks', 0))
                            aff_c = int(stats.get('affected_clusters', 0))
                            console.print(f"[dim]Micro post-dedup removed {del_n} blocks; {aff_c} clusters updated[/dim]")
                        except Exception as e:
                            console.print(f"[dim]Micro post-dedup skipped: {e}[/dim]")

                    # Precompute micro consensus for browser (robust import) after dedup
                    if db_path and db_path.exists():
                        import sqlite3
                        conn = sqlite3.connect(str(db_path))
                        try:
                            # Try package import first
                            try:
                                from genome_browser.database.cluster_content import precompute_all_micro_consensus  # type: ignore
                            except Exception:
                                # Fallback: load module directly from file path
                                import importlib.util as _ilu, sys as _sys
                                from pathlib import Path as _P
                                mod_path = _P(__file__).parents[1] / 'genome_browser' / 'database' / 'cluster_content.py'
                                spec = _ilu.spec_from_file_location('gb_cluster_content', str(mod_path))
                                if spec and spec.loader:
                                    gbmod = _ilu.module_from_spec(spec)
                                    spec.loader.exec_module(gbmod)  # type: ignore
                                    precompute_all_micro_consensus = getattr(gbmod, 'precompute_all_micro_consensus')  # type: ignore
                                else:
                                    raise ImportError('Could not load genome_browser.database.cluster_content')
                            nrows = precompute_all_micro_consensus(conn, 0.6, 0.9, 10)  # type: ignore
                            console.print(f"[dim]Precomputed micro consensus for {nrows} clusters[/dim]")
                        except Exception as e:
                            console.print(f"[dim]Micro consensus precompute skipped: {e}[/dim]")
                        finally:
                            conn.close()
                    # Report final micro counts
                    if db_path and db_path.exists():
                        import sqlite3
                        try:
                            conn = sqlite3.connect(str(db_path))
                            nb = conn.execute("SELECT COUNT(*) FROM micro_gene_blocks").fetchone()[0]
                            nc = conn.execute("SELECT COUNT(*) FROM micro_gene_clusters").fetchone()[0]
                            # Project paired micro alignments into syntenic_blocks and mappings (post-run, ensures availability)
                            try:
                                import pandas as pd
                                cur = conn.cursor()
                                # Load sidecars if present, else try DB tables
                                mdir = Path(output_path) / 'micro_gene'
                                mbp_csv = mdir / 'micro_block_pairs.csv'
                                mgpm_csv = mdir / 'micro_gene_pair_mappings.csv'
                                if mbp_csv.exists() and mgpm_csv.exists():
                                    mbp_df = pd.read_csv(mbp_csv)
                                    mgpm_df = pd.read_csv(mgpm_csv)
                                else:
                                    # Fallback: read from DB
                                    try:
                                        mbp_df = pd.read_sql_query("SELECT * FROM micro_block_pairs", conn)
                                        mgpm_df = pd.read_sql_query("SELECT * FROM micro_gene_pair_mappings", conn)
                                    except Exception:
                                        mbp_df = pd.DataFrame()
                                        mgpm_df = pd.DataFrame()
                                if not mbp_df.empty:
                                    # Build cluster offset map (if clusters from micro not yet inserted, leave as 0)
                                    row = conn.execute("SELECT COALESCE(MAX(cluster_id),0) FROM clusters").fetchone()
                                    start_cid = int(row[0] or 0)
                                    # Best-effort: compute non-sink micro IDs present in pairs
                                    mcids = sorted({int(c) for c in mbp_df['cluster_id'].unique() if int(c) > 0})
                                    cid_map = {c: (start_cid + c) for c in mcids}
                                    # Counts per role
                                    counts = mgpm_df.groupby(['block_id','block_role']).size().unstack(fill_value=0) if not mgpm_df.empty else None
                                    def role_count(series, bid, role):
                                        try:
                                            return int(series.get(role, pd.Series()).get(int(bid), 0))
                                        except Exception:
                                            return 0
                                    sb_pair_rows = []
                                    for r in mbp_df.itertuples(index=False):
                                        src_cid = int(r.cluster_id)
                                        dest_cid = 0 if src_cid == 0 else cid_map.get(src_cid, 0)
                                        q_locus = f"{r.query_genome_id}:{r.query_contig_id}:{int(r.query_start_bp)}-{int(r.query_end_bp)}"
                                        t_locus = f"{r.target_genome_id}:{r.target_contig_id}:{int(r.target_start_bp)}-{int(r.target_end_bp)}"
                                        q_len = int(r.query_end_bp) - int(r.query_start_bp)
                                        t_len = int(r.target_end_bp) - int(r.target_start_bp)
                                        length = q_len if q_len >= t_len else t_len
                                        nqw = role_count(counts, r.block_id, 'query') if counts is not None else None
                                        ntw = role_count(counts, r.block_id, 'target') if counts is not None else None
                                        sb_pair_rows.append((
                                            int(r.block_id), dest_cid, q_locus, t_locus,
                                            str(r.query_genome_id), str(r.target_genome_id),
                                            str(r.query_contig_id), str(r.target_contig_id),
                                            int(length), float(getattr(r, 'identity', 0.0) or 0.0), float(getattr(r, 'score', 0.0) or 0.0),
                                            nqw, ntw,
                                            None, None, None, None,
                                            None, None,
                                            'micro'
                                        ))
                                    if sb_pair_rows:
                                        cur.executemany(
                                            """
                                            INSERT OR REPLACE INTO syntenic_blocks (
                                                block_id, cluster_id, query_locus, target_locus,
                                                query_genome_id, target_genome_id,
                                                query_contig_id, target_contig_id,
                                                length, identity, score,
                                                n_query_windows, n_target_windows,
                                                query_window_start, query_window_end,
                                                target_window_start, target_window_end,
                                                query_windows_json, target_windows_json,
                                                block_type
                                            ) VALUES (
                                                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                                            )
                                            """,
                                            sb_pair_rows,
                                        )
                                    # Mappings into gene_block_mappings with relative positions
                                    if not mgpm_df.empty:
                                        gbm_rows2 = []
                                        for (blk, role), g in mgpm_df.groupby(['block_id','block_role']):
                                            g2 = g.sort_values(['start_pos','end_pos'])
                                            n = len(g2)
                                            L = max(1, n-1)
                                            for idx, rr in enumerate(g2.itertuples(index=False)):
                                                rel = (idx / L) if L > 0 else 0.0
                                                gbm_rows2.append((str(rr.gene_id), int(blk), str(role), float(rel)))
                                        for i in range(0, len(gbm_rows2), 1000):
                                            cur.executemany(
                                                "INSERT OR REPLACE INTO gene_block_mappings (gene_id, block_id, block_role, relative_position) VALUES (?, ?, ?, ?)",
                                                gbm_rows2[i:i+1000]
                                            )
                                    conn.commit()
                            except Exception:
                                pass
                        finally:
                            conn.close()
                        console.print(f"[bold]Micro clusters:[/bold] {nc} (blocks: {nb})")
                    else:
                        # No DB: report counts from pre-dedup DataFrames
                        console.print(f"[bold]Micro clusters:[/bold] {len(clusters_df)} (blocks: {len(blocks_df)})")
                    console.print("[green]✓ Micro-synteny results written to sidecar and DB (if available)[/green]")
            except Exception as e:
                console.print(f"[yellow]Micro-synteny pass failed/skipped: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.argument("block_id")
def explain(config: str, block_id: str):
    """Explain a syntenic block with detailed alignment info."""
    console.print(f"[bold]Explaining block: [green]{block_id}[/green][/bold]")
    
    # TODO: Implement explanation
    console.print("[yellow]Block explanation not yet implemented![/yellow]")


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
def stats(config: str):
    """Show QC statistics and index information."""
    console.print("[bold]ELSA Index Statistics[/bold]")
    
    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    # Create a nice table showing configuration
    table = Table(title="Current Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Parameter", style="white")  
    table.add_column("Value", style="green")
    
    # Add key config items
    table.add_row("PLM", "model", config_obj.plm.model)
    table.add_row("PLM", "device", config_obj.plm.device)
    table.add_row("PLM", "project_to_D", str(config_obj.plm.project_to_D))
    table.add_row("Discrete", "K", str(config_obj.discrete.K))
    table.add_row("Discrete", "minhash_hashes", str(config_obj.discrete.minhash_hashes))
    table.add_row("Continuous", "srp_bits", str(config_obj.continuous.srp_bits))
    table.add_row("System", "rng_seed", str(config_obj.system.rng_seed))
    
    console.print(table)
    
    # TODO: Add actual index statistics
    console.print("\n[yellow]Index statistics not yet available![/yellow]")


if __name__ == "__main__":
    main()
