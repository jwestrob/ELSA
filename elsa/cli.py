"""
ELSA command-line interface.
"""

import sys
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
@click.option("--force-pfam", is_flag=True, help="Force PFAM annotation even if embeddings exist")
def embed(config: str, input_dir: str, fasta_pattern: str, resume: bool, force_pfam: bool):
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
        
        # Stage 1.5: PFAM Annotation (if weighted sketching enabled)
        use_weighted_sketching = (
            hasattr(config_obj, 'phase2') and 
            config_obj.phase2.enable and 
            config_obj.phase2.weighted_sketch
        )
        
        if use_weighted_sketching:
            console.print("\n[bold blue]Stage 1.5: PFAM Annotation[/bold blue]")
            from .pfam_annotation import run_pfam_annotation_pipeline
            
            # Handle 'auto' threads properly
            if config_obj.system.jobs == 'auto':
                import os
                pfam_threads = os.cpu_count() or 8
            else:
                pfam_threads = config_obj.system.jobs
            
            pfam_summary = run_pfam_annotation_pipeline(
                config_obj, manifest, config_obj.data.work_dir, 
                threads=pfam_threads,
                force_annotation=force_pfam
            )
            
            if pfam_summary:
                # Handle both result formats
                if 'summary' in pfam_summary:
                    successful = pfam_summary['summary']['successful_genomes']
                    total = pfam_summary['summary']['total_genomes']
                    hits = pfam_summary['summary']['total_domains']
                else:
                    successful = pfam_summary.get('successful_samples', 0)
                    total = pfam_summary.get('total_samples', 0)
                    hits = pfam_summary.get('total_hits', 0)
                
                console.print(f"✓ PFAM annotation: {successful}/{total} samples")
                console.print(f"  Total domains found: {hits:,}")
            else:
                console.print("⚠️  PFAM annotation skipped or failed")
        
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
        
        shingle_system = ShingleSystem(config_obj.shingles, config_obj.data.work_dir, manifest, config_obj)
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
def analyze(config: str, min_windows: int, min_similarity: float, output_dir: str):
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
        
        # Automatically set up genome browser
        console.print(f"\n[bold blue]🧬 Setting up Genome Browser...[/bold blue]")
        setup_genome_browser(config_obj, output_path)
        
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


def setup_genome_browser(config_obj: ELSAConfig, results_path: Path):
    """Automatically set up genome browser after analysis."""
    try:
        import subprocess
        import sys
        
        # Paths
        genome_browser_dir = Path("genome_browser")
        setup_script = genome_browser_dir / "setup_genome_browser.py"
        
        if not setup_script.exists():
            console.print("[yellow]⚠️  Genome browser setup script not found, skipping[/yellow]")
            return
        
        # Find required files
        blocks_file = results_path / "syntenic_blocks.csv"
        clusters_file = results_path / "syntenic_clusters.csv"
        sequences_dir = Path("test_data/genomes")
        proteins_dir = Path("test_data/proteins")
        
        if not all([blocks_file.exists(), clusters_file.exists(), sequences_dir.exists(), proteins_dir.exists()]):
            console.print("[yellow]⚠️  Missing required files for genome browser setup[/yellow]")
            console.print(f"     Blocks: {blocks_file.exists()}")
            console.print(f"     Clusters: {clusters_file.exists()}")
            console.print(f"     Sequences: {sequences_dir.exists()}")
            console.print(f"     Proteins: {proteins_dir.exists()}")
            return
        
        # Prepare command
        cmd = [
            sys.executable,
            str(setup_script),
            "--sequences-dir", str(sequences_dir),
            "--proteins-dir", str(proteins_dir),
            "--blocks-file", str(blocks_file),
            "--clusters-file", str(clusters_file),
            "--skip-pfam"  # Use existing ELSA PFAM results
        ]
        
        console.print(f"Running: {' '.join(cmd)}")
        
        # Run setup
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            console.print("[green]✅ Genome browser setup completed successfully![/green]")
            console.print(f"\n[bold cyan]🚀 Ready to start genome browser:[/bold cyan]")
            console.print(f"  cd genome_browser")
            console.print(f"  streamlit run app.py")
        else:
            console.print(f"[red]❌ Genome browser setup failed (exit {result.returncode})[/red]")
            if result.stderr:
                console.print(f"Error: {result.stderr[:200]}...")
                
    except Exception as e:
        console.print(f"[red]❌ Genome browser setup failed: {e}[/red]")


if __name__ == "__main__":
    main()