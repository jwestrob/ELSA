"""
ELSA v2 command-line interface.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .params import ELSAConfig, load_config, create_default_config
from . import __version__

console = Console()


def print_banner():
    """Print ELSA banner."""
    console.print(f"""
[bold blue]ELSA[/bold blue] [dim]v{__version__}[/dim]
[italic]Embedding Locus Search and Alignment[/italic]
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

        cwd = Path.cwd()
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
        console.print("[yellow]Specify --sequences-dir and --proteins-dir for genome browser setup[/yellow]")
        return False

    sequences_path = Path(sequences_dir).absolute()
    proteins_path = Path(proteins_dir).absolute()

    if not sequences_path.exists():
        console.print(f"[red]Sequences directory not found: {sequences_path}[/red]")
        return False

    if not proteins_path.exists():
        console.print(f"[red]Proteins directory not found: {proteins_path}[/red]")
        return False

    # Check for existing PFAM annotations
    local_pfam = Path("genome_browser/pfam_annotations/pfam_annotation_results.json")
    pfam_results_file = local_pfam if local_pfam.exists() else genome_browser_dir / "pfam_annotations" / "pfam_annotation_results.json"

    cmd = [
        sys.executable, str(setup_script),
        "--sequences-dir", str(sequences_path),
        "--proteins-dir", str(proteins_path),
        "--blocks-file", str(blocks_file.absolute()),
        "--clusters-file", str(clusters_file.absolute()),
        "--db-path", str(Path(genome_browser_db).absolute()),
        "--force"
    ]

    if pfam_results_file.exists():
        console.print(f"Using existing PFAM annotations: {pfam_results_file}")
        cmd.extend(["--skip-pfam", "--pfam-output", str(pfam_results_file.parent.absolute())])
    else:
        console.print("No existing PFAM annotations found - will generate fresh annotations")

    console.print(f"Setting up genome browser database: {genome_browser_db}")

    try:
        subprocess.run(cmd, check=True, text=True, cwd=genome_browser_dir)
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
    """ELSA: Syntenic-block discovery from protein embeddings."""
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

    if config_path.exists() and not force:
        console.print(f"[red]Configuration file already exists: {config_path}[/red]")
        console.print("Use --force to overwrite")
        sys.exit(1)

    create_default_config(config_path)
    console.print(f"✓ Created configuration: [green]{config_path}[/green]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Put genome FASTA files in a directory (e.g., data/)")
    console.print("2. Run: [cyan]elsa embed data/[/cyan]")
    console.print("3. Run: [cyan]elsa analyze -o syntenic_analysis[/cyan]")


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--fasta-pattern", default="*.fasta", help="Pattern for nucleotide FASTA files")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
@click.option("--save-raw", is_flag=True, help="Save raw (unprojected) embeddings for later combined projection")
@click.option("--jobs", "-j", type=int, default=None,
              help="Max threads (overrides config system.jobs; default: all cores)")
def embed(config: str, input_dir: str, fasta_pattern: str, resume: bool, save_raw: bool, jobs: Optional[int]):
    """Generate protein embeddings from nucleotide FASTA files.

    Runs: gene calling -> PLM embedding -> PCA projection.
    Use --save-raw for cross-species workflows.
    """
    console.print("[bold]ELSA Embedding Pipeline[/bold]")

    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    input_path = Path(input_dir)

    # Auto-discover nucleotide FASTA files
    if fasta_pattern == "*.fasta":
        extensions = ["*.fasta", "*.fa", "*.fna"]
    else:
        extensions = [fasta_pattern]

    fasta_files = []
    for pattern in extensions:
        fasta_files.extend(input_path.glob(pattern))
    fasta_files = sorted(set(fasta_files))

    if not fasta_files:
        console.print(f"[red]No FASTA files found in {input_dir}[/red]")
        console.print(f"Searched for: {', '.join(extensions)}")
        sys.exit(1)

    console.print(f"Found {len(fasta_files)} nucleotide FASTA files:")
    for f in fasta_files:
        console.print(f"  - {f.name}")

    # Create sample data with optional precomputed annotations / proteins
    sample_data = []
    gff_hint_dir = Path("data/annotations")
    proteins_hint_dir = Path("data/proteins")
    # Also check sibling proteins/ dir relative to input_dir
    sibling_proteins_dir = input_path.parent / "proteins"

    console.print("\nResolved inputs:")
    for fasta_file in fasta_files:
        sample_id = fasta_file.stem

        aa_candidates = [
            sibling_proteins_dir / f"{sample_id}.faa",
            proteins_hint_dir / f"{sample_id}.faa",
            fasta_file.with_suffix(".faa"),
        ]
        gff_candidates = [
            gff_hint_dir / f"{sample_id}.gff",
            fasta_file.with_suffix(".gff"),
        ]

        aa_fasta_path = next((p for p in aa_candidates if p.exists()), None)
        gff_path = next((p for p in gff_candidates if p.exists()), None)

        sample_data.append((sample_id, fasta_file, gff_path, aa_fasta_path))

        if aa_fasta_path is not None:
            console.print(f"  - [cyan]{sample_id}[/cyan]: reuse proteins [green]{aa_fasta_path.name}[/green]")
        elif gff_path is not None:
            console.print(f"  - [cyan]{sample_id}[/cyan]: translate CDS from [green]{gff_path.name}[/green]")
        else:
            console.print(f"  - [cyan]{sample_id}[/cyan]: [dim]Prodigal gene calling[/dim]")

    console.print(f"\nGene caller: [green]{config_obj.ingest.gene_caller}[/green]")
    console.print(f"PLM Model: [green]{config_obj.plm.model}[/green]")
    console.print(f"Device: [green]{config_obj.plm.device}[/green]")
    console.print(f"Target dimension: [green]{config_obj.plm.project_to_D}[/green]")
    console.print(f"\nProcessing {len(sample_data)} samples...")

    try:
        from .ingest import ProteinIngester
        from .embeddings import ProteinEmbedder
        from .manifest import ELSAManifest

        manifest = ELSAManifest(config_obj.data.work_dir)
        manifest.set_config(config_obj)

        ingester = ProteinIngester(config_obj.ingest)
        from .embeddings import AggregationStrategy
        embedder = ProteinEmbedder(
            config_obj.plm,
            window_size=1024,
            overlap=256,
            aggregation=AggregationStrategy.MAX_POOL
        )

        # Stage 1: Protein ingestion
        console.print("\n[bold blue]Stage 1: Protein Ingestion[/bold blue]")
        sample_proteins = ingester.ingest_multiple(sample_data)

        # Stage 1.5: PFAM annotation
        if config_obj.ingest.run_pfam:
            console.print("\n[bold blue]Stage 1.5: PFAM Domain Annotation[/bold blue]")
            pfam_output_dir = Path("genome_browser/pfam_annotations")
            from .params import resolve_jobs
            threads = jobs if jobs is not None else resolve_jobs(config_obj.system.jobs)
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

        progress_file = Path(config_obj.data.work_dir) / "embed_progress.txt"
        console.print(f"Progress file: {progress_file}")
        embeddings = list(embedder.embed_sequences(all_proteins, progress_file=progress_file))

        console.print(f"\n✓ Generated {len(embeddings):,} embeddings")
        console.print(f"Embedding dimension: {embedder.embedding_dim}")

        # Stage 3: PCA projection and storage
        console.print("\n[bold blue]Stage 3: PCA Projection & Storage[/bold blue]")
        from .projection import ProjectionSystem

        projection_system = ProjectionSystem(config_obj.plm, config_obj.data.work_dir, manifest)

        protein_metadata = {}
        for proteins in sample_proteins.values():
            for protein in proteins:
                protein_metadata[protein.gene_id] = {
                    'contig_id': protein.contig_id,
                    'start': protein.start,
                    'end': protein.end,
                    'strand': protein.strand
                }

        if save_raw:
            console.print("\n[bold yellow]Saving raw (unprojected) embeddings...[/bold yellow]")
            raw_path = projection_system.save_raw_embeddings(embeddings, protein_metadata)
            console.print(f"[dim]Raw embeddings saved to: {raw_path}[/dim]")

        projection_system.process_embeddings(embeddings, protein_metadata)

        stats = projection_system.get_projection_stats()
        if stats:
            console.print(f"Explained variance: {stats['total_explained_variance']:.3f}")
            console.print(f"Projection: {stats['input_dim']}D -> {stats['output_dim']}D")

        console.print("\n[bold green]Embedding pipeline completed![/bold green]")
        console.print("Ready for analysis with: [cyan]elsa analyze -o syntenic_analysis[/cyan]")

    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.option("--raw", required=True, type=click.Path(exists=True),
              help="Path to raw embeddings parquet (genes_raw.parquet)")
def project(config: str, raw: str):
    """Project raw embeddings through PCA and save to genes.parquet.

    Use this to combine multiple separately-embedded datasets into a
    unified PCA space for cross-species comparison.
    """
    console.print("[bold]ELSA Projection Pipeline[/bold]")
    console.print("[dim]Projecting raw embeddings through PCA[/dim]")

    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    raw_path = Path(raw)

    try:
        from .projection import ProjectionSystem
        from .manifest import ELSAManifest

        manifest = ELSAManifest(config_obj.data.work_dir)
        manifest.set_config(config_obj)

        projection_system = ProjectionSystem(config_obj.plm, config_obj.data.work_dir, manifest)

        console.print(f"\n[bold blue]Loading raw embeddings from {raw_path}[/bold blue]")
        embeddings, protein_metadata = projection_system.load_raw_embeddings(raw_path)

        console.print(f"Loaded {len(embeddings):,} embeddings")
        console.print(f"Raw embedding dimension: {embeddings[0].embedding.shape[0]}")

        console.print(f"\n[bold blue]Fitting PCA projection to {config_obj.plm.project_to_D}D[/bold blue]")

        if projection_system.pca_model_path.exists():
            console.print("[yellow]Removing existing PCA model to fit fresh projection[/yellow]")
            projection_system.pca_model_path.unlink()
        if projection_system.scaler_path.exists():
            projection_system.scaler_path.unlink()

        projection_system.process_embeddings(embeddings, protein_metadata)

        stats = projection_system.get_projection_stats()
        if stats:
            console.print(f"\nExplained variance: {stats['total_explained_variance']:.3f}")
            console.print(f"Projection: {stats['input_dim']}D -> {stats['output_dim']}D")

        console.print("\n[bold green]Projection pipeline completed![/bold green]")
        console.print(f"Projected genes: {projection_system.genes_parquet_path}")
        console.print("Ready for analysis with: [cyan]elsa analyze -o syntenic_analysis[/cyan]")

    except Exception as e:
        console.print(f"\n[red]Projection failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="syntenic_analysis", help="Output directory")
@click.option("--genome-browser-db", default=None,
              help="Genome browser database path (disabled by default)")
@click.option("--sequences-dir", help="Directory containing nucleotide sequences (.fna)")
@click.option("--proteins-dir", help="Directory containing proteins (.faa)")
@click.option("--jobs", "-j", type=int, default=None,
              help="Max threads (overrides config system.jobs; default: all cores)")
def analyze(config: str, output_dir: str, genome_browser_db: str,
           sequences_dir: Optional[str], proteins_dir: Optional[str],
           jobs: Optional[int]):
    """Discover syntenic blocks via gene-level anchor chaining."""
    console.print("[bold]ELSA Syntenic Analysis[/bold]")

    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    try:
        from .manifest import ELSAManifest

        manifest = ELSAManifest(config_obj.data.work_dir)

        # Find genes.parquet
        genes_path = manifest.get_artifact_path('genes')
        if not genes_path or not Path(genes_path).exists():
            # Fallback paths
            for candidate in [
                Path(config_obj.data.work_dir) / "ingest" / "genes.parquet",
                Path(config_obj.data.work_dir) / "genes.parquet",
            ]:
                if candidate.exists():
                    genes_path = str(candidate)
                    break

        if not genes_path or not Path(genes_path).exists():
            console.print("[red]genes.parquet not found! Run 'elsa embed' first.[/red]")
            sys.exit(1)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run chain pipeline
        console.print("\n[bold blue]Running gene-level anchor chaining pipeline...[/bold blue]")
        from .analyze.micro_chain import run_micro_chain_pipeline, MicroChainConfig

        chain_dir = output_path / 'micro_chain'
        chain_cfg = config_obj.chain

        from .params import resolve_jobs
        _n_jobs = jobs if jobs is not None else resolve_jobs(config_obj.system.jobs)
        mc_config = MicroChainConfig(
            index_backend=chain_cfg.index_backend,
            faiss_nprobe=chain_cfg.faiss_nprobe,
            hnsw_k=chain_cfg.hnsw_k,
            hnsw_m=chain_cfg.hnsw_m,
            hnsw_ef_construction=chain_cfg.hnsw_ef_construction,
            hnsw_ef_search=chain_cfg.hnsw_ef_search,
            similarity_threshold=chain_cfg.similarity_threshold,
            max_gap_genes=chain_cfg.max_gap_genes,
            min_chain_size=chain_cfg.min_chain_size,
            jaccard_tau=chain_cfg.jaccard_tau,
            mutual_k=chain_cfg.mutual_k,
            df_max=chain_cfg.df_max,
            min_genome_support=chain_cfg.min_genome_support,
            gap_penalty_scale=chain_cfg.gap_penalty_scale,
            n_jobs=_n_jobs,
        )

        summary = run_micro_chain_pipeline(
            Path(genes_path),
            chain_dir,
            config=mc_config,
            db_path=Path(genome_browser_db) if genome_browser_db else None,
        )

        console.print(
            f"\n[bold green]Analysis complete![/bold green]\n"
            f"  genes={summary.num_genes} anchors={summary.num_anchors} "
            f"blocks={summary.num_blocks} clusters={summary.num_clusters} "
            f"singletons={summary.num_singletons} "
            f"genome_support_median={summary.genome_support_median} "
            f"mean_block_size={summary.mean_block_size:.1f}"
        )

        # Genome browser integration
        if genome_browser_db:
            # Create symlinks for genome browser compatibility
            blocks_csv = chain_dir / "micro_chain_blocks.csv"
            clusters_csv = chain_dir / "micro_chain_clusters.csv"

            out_blocks = output_path / "syntenic_blocks.csv"
            out_clusters = output_path / "syntenic_clusters.csv"

            if blocks_csv.exists():
                import shutil
                shutil.copy2(blocks_csv, out_blocks)
                shutil.copy2(clusters_csv, out_clusters)

            console.print("\n[bold blue]Setting up genome browser...[/bold blue]")
            success = setup_genome_browser_integration(
                config_obj, output_path, genome_browser_db,
                sequences_dir, proteins_dir
            )
            if success:
                console.print("[green]✓ Genome browser ready![/green]")
                console.print("  Run: cd genome_browser && streamlit run app.py")

    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("query_locus")
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
@click.option("--max-results", default=50, help="Maximum results to return")
@click.option("--fasta", default=None, type=click.Path(exists=True),
              help="Nucleotide FASTA file (required when query is a GFF)")
@click.option("--query-name", default=None,
              help="Override query genome name (default: filename stem)")
def search(query_locus: str, config: str, max_results: int,
           fasta: Optional[str], query_name: Optional[str]):
    """Search for syntenic blocks matching a query locus or file.

    QUERY_LOCUS can be:

    \b
      A protein FASTA (.faa/.fa/.fasta)        — embed on-the-fly
      A GFF file (.gff/.gff3) with --fasta     — translate + embed
      A locus string genome:contig:start-end   — use existing index

    Examples:

    \b
      elsa search query.faa -c config.yaml
      elsa search query.gff --fasta query.fna -c config.yaml
      elsa search "genome:contig:0-20" -c config.yaml
    """
    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    try:
        from .manifest import ELSAManifest
        from .index import build_gene_index
        from .search import search_locus as _search_locus

        manifest = ELSAManifest(config_obj.data.work_dir)

        # Find genes.parquet
        genes_path = manifest.get_artifact_path('genes')
        if not genes_path or not Path(genes_path).exists():
            for candidate in [
                Path(config_obj.data.work_dir) / "ingest" / "genes.parquet",
                Path(config_obj.data.work_dir) / "genes.parquet",
            ]:
                if candidate.exists():
                    genes_path = str(candidate)
                    break

        if not genes_path or not Path(genes_path).exists():
            console.print("[red]genes.parquet not found! Run 'elsa embed' first.[/red]")
            sys.exit(1)

        import pandas as pd
        import numpy as np

        # Load target genes
        genes_df = pd.read_parquet(genes_path)
        emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]

        genes_df = genes_df.sort_values(['sample_id', 'contig_id', 'start', 'end'])
        genes_df['position_index'] = genes_df.groupby(['sample_id', 'contig_id']).cumcount()

        # --- Determine query mode: file vs locus string ---
        PROTEIN_EXTENSIONS = {'.faa', '.fa', '.fasta'}
        GFF_EXTENSIONS = {'.gff', '.gff3'}
        FILE_EXTENSIONS = PROTEIN_EXTENSIONS | GFF_EXTENSIONS

        query_path = Path(query_locus)
        is_file_query = query_path.suffix.lower() in FILE_EXTENSIONS

        if is_file_query:
            # --- File-based query ---
            if not query_path.exists():
                console.print(f"[red]Query file not found: {query_locus}[/red]")
                sys.exit(1)

            name = query_name or query_path.stem
            ext = query_path.suffix.lower()
            console.print(
                f"[bold]File-based search: [green]{query_path.name}[/green]"
                f" (query name: {name})[/bold]"
            )

            from .ingest import ProteinIngester, GFFParser
            from .embeddings import ProteinSequence
            from .params import IngestConfig

            parse_config = IngestConfig(
                gene_caller="none",
                min_cds_aa=config_obj.ingest.min_cds_aa,
            )

            if ext in PROTEIN_EXTENSIONS:
                ingester = ProteinIngester(parse_config)
                proteins = ingester._parse_protein_fasta(query_path, name)
            elif ext in GFF_EXTENSIONS:
                if not fasta:
                    console.print("[red]GFF query requires --fasta <nucleotide.fna>[/red]")
                    sys.exit(1)
                parser = GFFParser(parse_config)
                genes = parser.parse_gff_with_fasta(query_path, Path(fasta), name)
                proteins = [
                    ProteinSequence(
                        sample_id=g.sample_id, contig_id=g.contig_id,
                        gene_id=g.gene_id, start=g.start, end=g.end,
                        strand=g.strand, sequence=g.sequence,
                    )
                    for g in genes
                ]
            else:
                console.print(f"[red]Unsupported file type: {ext}[/red]")
                sys.exit(1)

            if not proteins:
                console.print("[red]No proteins parsed from query file.[/red]")
                sys.exit(1)

            console.print(f"Parsed {len(proteins)} query proteins")

            # Embed and project query proteins
            from .search import embed_query_proteins
            query_genes = embed_query_proteins(
                proteins, config_obj.plm,
                str(config_obj.data.work_dir), name,
            )
            console.print(f"Embedded {len(query_genes)} query genes")

        else:
            # --- Locus string mode ---
            console.print(
                f"[bold]Searching for blocks matching:"
                f" [green]{query_locus}[/green][/bold]"
            )

            parts = query_locus.split(":")
            if len(parts) != 3 or "-" not in parts[2]:
                console.print("[red]Invalid locus format. Use: genome:contig:start-end[/red]")
                sys.exit(1)

            q_genome, q_contig, q_range = parts
            q_start, q_end = map(int, q_range.split("-"))

            query_mask = (
                (genes_df['sample_id'] == q_genome) &
                (genes_df['contig_id'] == q_contig) &
                (genes_df['position_index'] >= q_start) &
                (genes_df['position_index'] <= q_end)
            )
            query_genes = genes_df[query_mask].copy()

            if query_genes.empty:
                console.print(f"[yellow]No genes found for locus {query_locus}[/yellow]")
                console.print(f"Available genomes: {sorted(genes_df['sample_id'].unique())}")
                sys.exit(1)

            console.print(f"Query locus: {len(query_genes)} genes")

        # --- Common: build index and search ---
        embeddings = genes_df[emb_cols].values.astype(np.float32)
        console.print(f"Building index ({config_obj.chain.index_backend}) over {len(genes_df)} genes...")
        index_tuple = build_gene_index(
            embeddings,
            m=config_obj.chain.hnsw_m,
            ef_construction=config_obj.chain.hnsw_ef_construction,
            ef_search=config_obj.chain.hnsw_ef_search,
            index_backend=config_obj.chain.index_backend,
            faiss_nprobe=config_obj.chain.faiss_nprobe,
        )

        blocks = _search_locus(
            query_genes=query_genes,
            index_tuple=index_tuple,
            target_genes=genes_df.reset_index(drop=True),
            target_embeddings=embeddings,
            k=config_obj.chain.hnsw_k,
            similarity_threshold=config_obj.chain.similarity_threshold,
            max_gap=config_obj.chain.max_gap_genes,
            min_chain_size=config_obj.chain.min_chain_size,
            gap_penalty_scale=config_obj.chain.gap_penalty_scale,
            max_results=max_results,
        )

        if not blocks:
            console.print("[yellow]No matching syntenic blocks found.[/yellow]")
            return

        console.print(f"\n[bold green]Found {len(blocks)} syntenic blocks:[/bold green]")

        for i, block in enumerate(blocks[:20], 1):
            console.print(
                f"  {i}. {block.target_genome}:{block.target_contig}"
                f"  genes [{block.target_start}-{block.target_end}]"
                f"  anchors={block.n_anchors}"
                f"  score={block.chain_score:.2f}"
                f"  orient={'fwd' if block.orientation > 0 else 'inv'}"
            )

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default="elsa.config.yaml",
              help="Configuration file", type=click.Path(exists=True))
def stats(config: str):
    """Show configuration and index information."""
    console.print("[bold]ELSA Configuration[/bold]")

    try:
        config_obj = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    table = Table(title="Current Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Parameter", style="white")
    table.add_column("Value", style="green")

    table.add_row("PLM", "model", config_obj.plm.model)
    table.add_row("PLM", "device", config_obj.plm.device)
    table.add_row("PLM", "project_to_D", str(config_obj.plm.project_to_D))
    table.add_row("Chain", "similarity_threshold", str(config_obj.chain.similarity_threshold))
    table.add_row("Chain", "max_gap_genes", str(config_obj.chain.max_gap_genes))
    table.add_row("Chain", "min_chain_size", str(config_obj.chain.min_chain_size))
    table.add_row("Chain", "gap_penalty_scale", str(config_obj.chain.gap_penalty_scale))
    table.add_row("System", "rng_seed", str(config_obj.system.rng_seed))

    console.print(table)


@main.command()
@click.option("--db", type=click.Path(exists=True),
              help="Sharur DuckDB database path (protein metadata)")
@click.option("--proteins", type=click.Path(exists=True, file_okay=False),
              help="Directory of .faa protein FASTA files (alternative to --db)")
@click.option("--embeddings", type=click.Path(exists=True),
              help="HDF5 file with protein embeddings (Sharur format)")
@click.option("--embeddings-parquet", type=click.Path(exists=True),
              help="Parquet file with emb_* embedding columns")
@click.option("--store", type=click.Path(),
              help="Persistent FAISS store directory (load or create)")
@click.option("--add-db", type=click.Path(exists=True),
              help="DuckDB with new genomes to add to existing --store")
@click.option("--add-embeddings", type=click.Path(exists=True),
              help="HDF5 with new embeddings to add to existing --store")
@click.option("--output-dir", "-o", default="syntenic_output", help="Output directory")
@click.option("--similarity-threshold", default=0.85, help="Cosine similarity threshold for anchors")
@click.option("--max-gap", default=2, help="Maximum gene gap in chains")
@click.option("--min-chain-size", default=2, help="Minimum anchors per chain")
@click.option("--min-genome-support", default=2, help="Minimum genomes per cluster")
@click.option("--gap-penalty-scale", default=0.0, help="Concave gap penalty (0=off)")
@click.option("--jaccard-tau", default=0.3, help="Jaccard threshold for cluster overlap")
@click.option("--index-backend", default="auto", help="Index backend (auto/faiss/hnswlib)")
@click.option("--hnsw-k", default=50, help="k for HNSW neighbor search")
@click.option("--no-normalize", is_flag=True, help="Skip L2 normalization of embeddings")
@click.option("--jobs", "-j", type=int, required=True,
              help="Max threads for FAISS search and clustering (required)")
@click.option("--annotations-db", type=click.Path(exists=True),
              help="Sharur DuckDB with annotations table (loads PFAM + all sources)")
@click.option("--genome-browser-db", type=click.Path(), default=None,
              help="Genome browser DB path (disabled by default; pass path to enable)")
def synteny(db: Optional[str], proteins: Optional[str],
            embeddings: Optional[str], embeddings_parquet: Optional[str],
            store: Optional[str],
            add_db: Optional[str], add_embeddings: Optional[str],
            output_dir: str, similarity_threshold: float, max_gap: int,
            min_chain_size: int, min_genome_support: int,
            gap_penalty_scale: float, jaccard_tau: float,
            index_backend: str, hnsw_k: int, no_normalize: bool,
            jobs: int,
            annotations_db: Optional[str], genome_browser_db: Optional[str]):
    """Discover syntenic blocks from external embeddings.

    Accepts protein metadata from a Sharur DuckDB or FASTA files,
    and embeddings from HDF5 or parquet. No ELSA config file needed.

    Use --store for persistent FAISS index that survives across runs.
    Use --add-db / --add-embeddings to append new genomes to a store.

    \b
    Examples:
      elsa synteny --db sharur.duckdb --embeddings proteins.h5 -o results/
      elsa synteny --db sharur.duckdb --embeddings proteins.h5 --store ./my_store -o results/
      elsa synteny --store ./my_store -o results/
      elsa synteny --store ./my_store --add-db new.duckdb --add-embeddings new.h5 -o results/
    """
    console.print("[bold]ELSA Synteny Discovery[/bold]")

    try:
        from .analyze.pipeline import run_chain_pipeline, ChainConfig

        genes_df = None
        prebuilt_index = None

        # --- Store mode ---
        if store:
            from .store import SyntenyStore

            store_path = Path(store)
            config_exists = (store_path / "config.json").exists()

            if config_exists and not (db or proteins or embeddings or embeddings_parquet):
                # Load existing store, no new data
                console.print(f"Loading store: [green]{store}[/green]")
                ss = SyntenyStore.load(store_path)

                # Handle --add-db / --add-embeddings
                if add_db and add_embeddings:
                    from .adapter import (
                        load_proteins_from_duckdb,
                        load_embeddings_h5,
                        build_genes_dataframe,
                    )
                    console.print(f"Adding genomes from: [green]{add_db}[/green]")
                    new_proteins = load_proteins_from_duckdb(add_db)
                    new_emb = load_embeddings_h5(add_embeddings)
                    new_genes = build_genes_dataframe(
                        new_proteins, new_emb, normalize=not no_normalize,
                    )
                    ss.add_genes(new_genes)

                # Pass metadata and embeddings SEPARATELY to avoid
                # building a combined DataFrame that doubles memory
                genes_df = ss._metadata.copy()
                genes_df["strand"] = genes_df.get("strand", 0)
                store_embeddings = ss._embeddings
                prebuilt_index = ss.get_index_tuple()
                # Free the store's reference to embeddings (pipeline owns it now)
                ss._embeddings = None
                console.print(
                    f"  {ss.n_vectors:,} genes, {ss.dim}D, "
                    f"{len(ss.genomes)} genomes"
                )

            else:
                # Build new store from provided data
                genes_df = _load_genes_from_sources(
                    db, proteins, embeddings, embeddings_parquet,
                    no_normalize, console,
                )
                console.print(f"Creating store: [green]{store}[/green]")
                ss = SyntenyStore.create(store_path, genes_df)
                prebuilt_index = ss.get_index_tuple()

        else:
            # --- No store: one-shot mode ---
            if not (db or proteins):
                console.print("[red]Provide --db, --proteins, or --store[/red]")
                sys.exit(1)
            if not (embeddings or embeddings_parquet):
                console.print("[red]Provide --embeddings or --embeddings-parquet[/red]")
                sys.exit(1)

            genes_df = _load_genes_from_sources(
                db, proteins, embeddings, embeddings_parquet,
                no_normalize, console,
            )

        # Run pipeline
        config = ChainConfig(
            index_backend=index_backend,
            hnsw_k=hnsw_k,
            similarity_threshold=similarity_threshold,
            max_gap_genes=max_gap,
            min_chain_size=min_chain_size,
            gap_penalty_scale=gap_penalty_scale,
            jaccard_tau=jaccard_tau,
            min_genome_support=min_genome_support,
            n_jobs=max(1, jobs),
        )

        console.print(f"\n[bold blue]Running chain pipeline...[/bold blue]")
        # Pass embeddings separately if available (avoids DataFrame copy)
        _store_emb = locals().get("store_embeddings", None)
        summary = run_chain_pipeline(
            output_dir=Path(output_dir),
            config=config,
            genes_df=genes_df,
            prebuilt_index=prebuilt_index,
            embeddings=_store_emb,
        )

        console.print(
            f"\n[bold green]Synteny discovery complete![/bold green]\n"
            f"  genes={summary.num_genes} anchors={summary.num_anchors} "
            f"blocks={summary.num_blocks} clusters={summary.num_clusters} "
            f"singletons={summary.num_singletons} "
            f"genome_support_median={summary.genome_support_median} "
            f"mean_block_size={summary.mean_block_size:.1f}"
        )
        console.print(f"  Output: [green]{output_dir}[/green]")

        # Load annotations from Sharur DuckDB if provided
        annotations_all_df = None
        ann_db = annotations_db or db  # fall back to --db if it has annotations
        if annotations_db:
            from .adapter import load_annotations_from_duckdb, load_all_annotations_from_duckdb

            console.print(f"\n[bold blue]Loading annotations from DuckDB...[/bold blue]")

            # Phase 1: PFAM → merge into genes_df for pfam_domains columns
            pfam_df = load_annotations_from_duckdb(annotations_db, source="pfam")
            if not pfam_df.empty and genes_df is not None:
                # Drop existing pfam columns if any, then merge
                for col in ["pfam_domains", "pfam_count", "primary_pfam"]:
                    if col in genes_df.columns:
                        genes_df = genes_df.drop(columns=[col])
                genes_df = genes_df.merge(pfam_df, on="gene_id", how="left")
                n_ann = genes_df["pfam_domains"].notna().sum()
                console.print(f"  PFAM: {n_ann:,}/{len(genes_df):,} genes annotated")

            # Phase 2: all sources → annotations_multi table
            annotations_all_df = load_all_annotations_from_duckdb(annotations_db)

        # Genome browser DB — opt-in via --genome-browser-db flag
        if genome_browser_db and summary.num_blocks > 0 and genes_df is not None:
            from .browser import populate_browser_db

            browser_db = Path(genome_browser_db)
            _out = Path(output_dir)
            blocks_csv = _out / "micro_chain_blocks.csv"
            clusters_csv = _out / "micro_chain_clusters.csv"

            if blocks_csv.exists() and clusters_csv.exists():
                console.print(f"\n[bold blue]Populating genome browser DB...[/bold blue]")
                populate_browser_db(
                    browser_db, genes_df, blocks_csv, clusters_csv,
                    annotations_df=annotations_all_df,
                )
                console.print(f"  [green]{browser_db}[/green]")

    except Exception as e:
        console.print(f"[red]Synteny discovery failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--store", type=click.Path(exists=True), default=None,
              help="SyntenyStore directory (for gene metadata)")
@click.option("--genes-parquet", type=click.Path(exists=True), default=None,
              help="genes.parquet with metadata (alternative to --store)")
@click.option("--db-path", type=click.Path(), default=None,
              help="Output DB path (default: <output_dir>/genome_browser.db)")
@click.option("--annotations-db", type=click.Path(exists=True), default=None,
              help="Sharur DuckDB with annotations table (loads PFAM + all sources)")
def browser(output_dir: str, store: Optional[str],
            genes_parquet: Optional[str], db_path: Optional[str],
            annotations_db: Optional[str]):
    """Populate a genome browser DB from existing pipeline output.

    Reads micro_chain_blocks.csv and micro_chain_clusters.csv from
    OUTPUT_DIR and populates a SQLite database for the genome browser.

    Gene metadata comes from --store or --genes-parquet.
    Annotations come from --annotations-db (Sharur DuckDB with annotations table).

    \b
    Examples:
      elsa browser results/ --store ./my_store
      elsa browser results/ --store ./my_store --annotations-db sharur.duckdb
      elsa browser results/ --genes-parquet elsa_index/ingest/genes.parquet
    """
    out = Path(output_dir)
    blocks_csv = out / "micro_chain_blocks.csv"
    clusters_csv = out / "micro_chain_clusters.csv"

    if not blocks_csv.exists():
        console.print(f"[red]Not found: {blocks_csv}[/red]")
        sys.exit(1)
    if not clusters_csv.exists():
        console.print(f"[red]Not found: {clusters_csv}[/red]")
        sys.exit(1)

    import pandas as pd

    if store:
        from .store import SyntenyStore
        ss = SyntenyStore.load(Path(store))
        genes_df = ss.get_genes_df()
    elif genes_parquet:
        genes_df = pd.read_parquet(genes_parquet)
    else:
        console.print("[red]Provide --store or --genes-parquet for gene metadata[/red]")
        sys.exit(1)

    # Load annotations from Sharur DuckDB if provided
    annotations_all_df = None
    if annotations_db:
        from .adapter import load_annotations_from_duckdb, load_all_annotations_from_duckdb

        console.print(f"Loading annotations from: [green]{annotations_db}[/green]")

        # PFAM → merge into genes_df
        pfam_df = load_annotations_from_duckdb(annotations_db, source="pfam")
        if not pfam_df.empty:
            for col in ["pfam_domains", "pfam_count", "primary_pfam"]:
                if col in genes_df.columns:
                    genes_df = genes_df.drop(columns=[col])
            genes_df = genes_df.merge(pfam_df, on="gene_id", how="left")
            n_ann = genes_df["pfam_domains"].notna().sum()
            console.print(f"  PFAM: {n_ann:,}/{len(genes_df):,} genes annotated")

        # All sources → annotations_multi
        annotations_all_df = load_all_annotations_from_duckdb(annotations_db)

    from .browser import populate_browser_db

    target_db = Path(db_path) if db_path else out / "genome_browser.db"
    console.print(f"[bold blue]Populating genome browser DB...[/bold blue]")
    populate_browser_db(
        target_db, genes_df, blocks_csv, clusters_csv,
        annotations_df=annotations_all_df,
    )
    console.print(f"  [green]{target_db}[/green]")


def _load_genes_from_sources(
    db: Optional[str],
    proteins: Optional[str],
    embeddings: Optional[str],
    embeddings_parquet: Optional[str],
    no_normalize: bool,
    console,
) -> "pd.DataFrame":
    """Load and merge protein metadata + embeddings from CLI sources."""
    from .adapter import (
        load_proteins_from_duckdb,
        load_proteins_from_fasta,
        load_embeddings_h5,
        load_embeddings_parquet as _load_emb_pq,
        build_genes_dataframe,
    )

    if db:
        console.print(f"Loading proteins from DuckDB: [green]{db}[/green]")
        proteins_df = load_proteins_from_duckdb(db)
    else:
        console.print(f"Loading proteins from FASTA: [green]{proteins}[/green]")
        proteins_df = load_proteins_from_fasta(proteins)

    n_genomes = proteins_df["sample_id"].nunique()
    console.print(f"  {len(proteins_df):,} proteins from {n_genomes} genomes")

    if embeddings:
        console.print(f"Loading embeddings from HDF5: [green]{embeddings}[/green]")
        emb_df = load_embeddings_h5(embeddings)
    else:
        console.print(f"Loading embeddings from parquet: [green]{embeddings_parquet}[/green]")
        emb_df = _load_emb_pq(embeddings_parquet)

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    console.print(f"  {len(emb_df):,} vectors, {len(emb_cols)}D")

    genes_df = build_genes_dataframe(
        proteins_df, emb_df, normalize=not no_normalize,
    )
    console.print(f"  Merged: {len(genes_df):,} genes with embeddings")
    return genes_df


if __name__ == "__main__":
    main()
