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
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=genome_browser_dir  # Run from genome_browser directory
        )
        console.print("✓ Genome browser setup completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Genome browser setup failed (return code {e.returncode})[/red]")
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
def analyze(config: str, min_windows: int, min_similarity: float, output_dir: str, 
           setup_genome_browser: bool, genome_browser_db: str, sequences_dir: Optional[str], proteins_dir: Optional[str]):
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
                else:
                    console.print(f"\n[yellow]⚠ Genome browser setup skipped (see messages above)[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Genome browser setup failed: {e}[/red]")
                console.print("[yellow]Analysis results are still available - you can set up the browser manually[/yellow]")
        
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
