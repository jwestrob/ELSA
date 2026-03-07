#!/usr/bin/env python3
"""
Download and parse operon databases for ELSA benchmarking.

Sources:
- B. subtilis: SubtiWiki (http://subtiwiki.uni-goettingen.de/)
- E. coli: RegulonDB (https://regulondb.ccg.unam.mx/)

Usage:
    python benchmarks/scripts/download_operons.py

Output:
    benchmarks/operons/bsubtilis/operons.tsv
    benchmarks/operons/ecoli/operons.tsv
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional
import ssl
import urllib3

# Suppress SSL warnings for RegulonDB (has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
OPERONS_DIR = BENCHMARKS_DIR / "operons"


def download_file(url: str, output_path: Path, verify_ssl: bool = True) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, verify=verify_ssl, timeout=60)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        print(f"  Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_subtiwiki_operons() -> Optional[Path]:
    """
    Download B. subtilis operons from SubtiWiki.

    SubtiWiki provides downloadable tables in various formats.
    We'll try the gene-operon mapping from their downloads page.
    """
    output_dir = OPERONS_DIR / "bsubtilis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # SubtiWiki URLs to try
    urls = [
        # Gene table with operon info
        ("http://subtiwiki.uni-goettingen.de/v4/exports/gene.json", "genes.json"),
        # Try operon-specific export
        ("http://subtiwiki.uni-goettingen.de/v4/exports/operon.json", "operons_raw.json"),
    ]

    downloaded = []
    for url, filename in urls:
        output_path = output_dir / filename
        if download_file(url, output_path):
            downloaded.append(output_path)

    if not downloaded:
        print("Warning: Could not download SubtiWiki data. Using fallback operon list.")
        return create_fallback_bsubtilis_operons(output_dir)

    # Parse the downloaded data
    return parse_subtiwiki_data(output_dir, downloaded)


def parse_subtiwiki_data(output_dir: Path, downloaded_files: list) -> Path:
    """Parse SubtiWiki JSON exports into standardized TSV."""
    operons = {}

    for filepath in downloaded_files:
        if filepath.name == "genes.json":
            try:
                data = json.loads(filepath.read_text())
                # SubtiWiki gene export has operon field
                for gene in data:
                    if isinstance(gene, dict):
                        operon_name = gene.get('operon') or gene.get('operonName')
                        gene_name = gene.get('name') or gene.get('locus')
                        if operon_name and gene_name:
                            if operon_name not in operons:
                                operons[operon_name] = {'genes': [], 'evidence': 'SubtiWiki'}
                            operons[operon_name]['genes'].append(gene_name)
            except Exception as e:
                print(f"  Error parsing {filepath}: {e}")

        elif filepath.name == "operons_raw.json":
            try:
                data = json.loads(filepath.read_text())
                for operon in data:
                    if isinstance(operon, dict):
                        operon_name = operon.get('name') or operon.get('id')
                        genes = operon.get('genes', [])
                        if isinstance(genes, str):
                            genes = [g.strip() for g in genes.split(',')]
                        if operon_name and genes:
                            operons[operon_name] = {
                                'genes': genes,
                                'evidence': 'SubtiWiki'
                            }
            except Exception as e:
                print(f"  Error parsing {filepath}: {e}")

    if not operons:
        print("  No operons parsed from SubtiWiki data")
        return create_fallback_bsubtilis_operons(output_dir)

    # Write standardized TSV
    output_path = output_dir / "operons.tsv"
    with open(output_path, 'w') as f:
        f.write("operon_id\tgenes\tgene_count\tevidence\n")
        for operon_id, info in sorted(operons.items()):
            genes = ','.join(info['genes'])
            gene_count = len(info['genes'])
            evidence = info['evidence']
            f.write(f"{operon_id}\t{genes}\t{gene_count}\t{evidence}\n")

    print(f"  Wrote {len(operons)} operons to {output_path}")
    return output_path


def create_fallback_bsubtilis_operons(output_dir: Path) -> Path:
    """
    Create fallback operon list for B. subtilis from well-characterized operons.

    These are highly conserved operons that should be present across strains.
    Source: Literature and NCBI gene records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Well-characterized B. subtilis operons
    # Format: operon_name, [gene_list], evidence_type
    operons = [
        # Ribosomal protein operons (highly conserved)
        ("rpsJ-rplC", ["rpsJ", "rplC", "rplD", "rplW", "rplB", "rpsS", "rplV", "rpsC", "rplP", "rpmC", "rpsQ"], "ribosomal"),
        ("rpsL-rpsG", ["rpsL", "rpsG", "fusA", "tuf"], "ribosomal"),
        ("rpmI-rplT", ["rpmI", "rplT"], "ribosomal"),
        ("rpsA", ["rpsA"], "ribosomal"),
        ("rpsB-tsf", ["rpsB", "tsf"], "ribosomal"),

        # Sporulation operons
        ("spoIIA", ["spoIIAA", "spoIIAB", "sigF"], "sporulation"),
        ("spoIIG", ["spoIIGA", "sigE"], "sporulation"),
        ("spoIIE", ["spoIIE"], "sporulation"),
        ("spoVA", ["spoVAA", "spoVAB", "spoVAC", "spoVAD", "spoVAEa", "spoVAEb", "spoVAF"], "sporulation"),
        ("cotE", ["cotE"], "sporulation"),

        # Competence operons
        ("comK", ["comK"], "competence"),
        ("comG", ["comGA", "comGB", "comGC", "comGD", "comGE", "comGF", "comGG"], "competence"),
        ("comE", ["comEA", "comEB", "comEC"], "competence"),
        ("comF", ["comFA", "comFB", "comFC"], "competence"),

        # Amino acid biosynthesis
        ("trpEDCFBA", ["trpE", "trpD", "trpC", "trpF", "trpB", "trpA"], "tryptophan"),
        ("hisGDBHAFIE", ["hisG", "hisD", "hisB", "hisH", "hisA", "hisF", "hisI", "hisE"], "histidine"),
        ("ilvBHC-leuABCD", ["ilvB", "ilvH", "ilvC", "leuA", "leuB", "leuC", "leuD"], "BCAA"),
        ("argCJBD-carAB-argF", ["argC", "argJ", "argB", "argD", "carA", "carB", "argF"], "arginine"),
        ("lysC", ["lysC"], "lysine"),

        # Cell division
        ("ftsAZ", ["ftsA", "ftsZ"], "division"),
        ("divIVA", ["divIVA"], "division"),
        ("minCD", ["minC", "minD"], "division"),

        # DNA replication
        ("dnaA-dnaN", ["dnaA", "dnaN", "recF", "gyrB"], "replication"),
        ("dnaE", ["dnaE"], "replication"),
        ("polC", ["polC"], "replication"),

        # Energy metabolism
        ("atpIBEFHAGDC", ["atpI", "atpB", "atpE", "atpF", "atpH", "atpA", "atpG", "atpD", "atpC"], "ATP_synthase"),
        ("cydABCD", ["cydA", "cydB", "cydC", "cydD"], "cytochrome"),
        ("qcrABC", ["qcrA", "qcrB", "qcrC"], "cytochrome"),
        ("ctaABCDE", ["ctaA", "ctaB", "ctaC", "ctaD", "ctaE"], "cytochrome"),

        # Cell wall
        ("murBCDEF", ["murB", "murC", "murD", "murE", "murF"], "peptidoglycan"),
        ("pbpA", ["pbpA"], "peptidoglycan"),
        ("pbpB", ["pbpB"], "peptidoglycan"),
        ("tagDEF", ["tagD", "tagE", "tagF"], "teichoic_acid"),

        # Motility and chemotaxis
        ("fla-che", ["flgB", "flgC", "fliE", "fliF", "fliG", "fliH", "fliI", "fliJ", "ylxF", "fliK", "ylxG", "flgE", "flgK", "fliL", "fliM", "fliY", "cheY", "fliZ", "fliP", "fliQ", "fliR", "flhB", "flhA", "flhF", "ylxH", "cheB", "cheA", "cheW", "cheC", "cheD", "sigD", "swrB"], "motility"),
        ("hag", ["hag"], "flagellin"),
        ("motAB", ["motA", "motB"], "motility"),

        # Secretion
        ("secA", ["secA"], "secretion"),
        ("secYEG", ["secY", "secE", "secG"], "secretion"),
        ("secDF-yajC", ["secD", "secF", "yajC"], "secretion"),

        # Toxin-antitoxin (small cassettes)
        ("bsrG-SR4", ["bsrG", "SR4"], "TA_system"),
        ("yonT-yoyJ", ["yonT", "yoyJ"], "TA_system"),

        # Stress response
        ("groESL", ["groES", "groEL"], "chaperone"),
        ("dnaKJ", ["dnaK", "dnaJ", "grpE"], "chaperone"),
        ("clpPX", ["clpP", "clpX"], "protease"),
        ("sigB-rsbVWX", ["rsbV", "rsbW", "sigB", "rsbX"], "stress"),

        # Carbon metabolism
        ("pdhABCD", ["pdhA", "pdhB", "pdhC", "pdhD"], "pyruvate_dehydrogenase"),
        ("citZ-icd-mdh", ["citZ", "icd", "mdh"], "TCA"),
        ("gapA-pgk-tpiA-pgm-eno", ["gapA", "pgk", "tpiA", "pgm", "eno"], "glycolysis"),
        ("ptsGHI", ["ptsG", "ptsH", "ptsI"], "PTS"),

        # Additional well-characterized operons
        ("spo0A", ["spo0A"], "sporulation"),
        ("kinA", ["kinA"], "sporulation"),
        ("sigA", ["sigA"], "sigma"),
        ("sigH", ["sigH"], "sigma"),
        ("sinIR", ["sinI", "sinR"], "biofilm"),
        ("epsA-O", ["epsA", "epsB", "epsC", "epsD", "epsE", "epsF", "epsG", "epsH", "epsI", "epsJ", "epsK", "epsL", "epsM", "epsN", "epsO"], "biofilm"),
    ]

    output_path = output_dir / "operons.tsv"
    with open(output_path, 'w') as f:
        f.write("operon_id\tgenes\tgene_count\tevidence\n")
        for operon_id, genes, evidence in operons:
            f.write(f"{operon_id}\t{','.join(genes)}\t{len(genes)}\t{evidence}\n")

    print(f"  Created fallback B. subtilis operon list with {len(operons)} operons")
    return output_path


def download_regulondb_operons() -> Optional[Path]:
    """
    Download E. coli operons from RegulonDB.

    RegulonDB provides clean TSV downloads of operon data.
    """
    output_dir = OPERONS_DIR / "ecoli"
    output_dir.mkdir(parents=True, exist_ok=True)

    # RegulonDB download URLs (verify=False due to certificate issues)
    urls = [
        ("https://regulondb.ccg.unam.mx/menu/download/datasets/files/OperonSet.txt", "OperonSet.txt"),
        ("https://regulondb.ccg.unam.mx/menu/download/datasets/files/GeneProductSet.txt", "GeneProductSet.txt"),
    ]

    downloaded = []
    for url, filename in urls:
        output_path = output_dir / filename
        if download_file(url, output_path, verify_ssl=False):
            downloaded.append(output_path)

    if not downloaded:
        print("Warning: Could not download RegulonDB data. Using fallback operon list.")
        return create_fallback_ecoli_operons(output_dir)

    # Parse RegulonDB format
    return parse_regulondb_data(output_dir, downloaded)


def parse_regulondb_data(output_dir: Path, downloaded_files: list) -> Path:
    """Parse RegulonDB operon data into standardized TSV."""
    operons = {}
    gene_info = {}  # gene_id -> gene_name mapping

    # First, parse gene information if available
    for filepath in downloaded_files:
        if filepath.name == "GeneProductSet.txt":
            try:
                with open(filepath, 'r', errors='replace') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gene_id = parts[0]
                            gene_name = parts[1] if len(parts) > 1 else parts[0]
                            gene_info[gene_id] = gene_name
            except Exception as e:
                print(f"  Error parsing {filepath}: {e}")

    # Parse operon file
    for filepath in downloaded_files:
        if filepath.name == "OperonSet.txt":
            try:
                with open(filepath, 'r', errors='replace') as f:
                    header = None
                    for line in f:
                        if line.startswith('#'):
                            # Try to extract header
                            if 'operon' in line.lower() or 'gene' in line.lower():
                                header = line.strip('#').strip().split('\t')
                            continue
                        if not line.strip():
                            continue

                        parts = line.strip().split('\t')
                        if len(parts) < 2:
                            continue

                        # RegulonDB format: operon_name, first_gene, last_gene, strand, ...
                        # or: operon_id, operon_name, genes, ...
                        operon_name = parts[0]

                        # Try to extract genes (format varies by RegulonDB version)
                        genes = []
                        evidence = "RegulonDB"

                        # Check for genes column
                        for i, part in enumerate(parts):
                            # Look for comma-separated gene list
                            if ',' in part and not part.startswith('ECK'):
                                genes = [g.strip() for g in part.split(',') if g.strip()]
                            # Look for evidence
                            if part.lower() in ['strong', 'weak', 'confirmed', 'predicted']:
                                evidence = part

                        # If no gene list found, try first/last gene approach
                        if not genes and len(parts) >= 3:
                            first_gene = parts[1]
                            last_gene = parts[2]
                            # For simplicity, just use first gene (full list requires genome lookup)
                            genes = [first_gene]
                            if first_gene != last_gene:
                                genes.append(last_gene)

                        if operon_name and genes:
                            operons[operon_name] = {
                                'genes': genes,
                                'evidence': evidence
                            }
            except Exception as e:
                print(f"  Error parsing {filepath}: {e}")

    if not operons:
        print("  No operons parsed from RegulonDB data")
        return create_fallback_ecoli_operons(output_dir)

    # Write standardized TSV
    output_path = output_dir / "operons.tsv"
    with open(output_path, 'w') as f:
        f.write("operon_id\tgenes\tgene_count\tevidence\n")
        for operon_id, info in sorted(operons.items()):
            genes = ','.join(info['genes'])
            gene_count = len(info['genes'])
            evidence = info['evidence']
            f.write(f"{operon_id}\t{genes}\t{gene_count}\t{evidence}\n")

    print(f"  Wrote {len(operons)} operons to {output_path}")
    return output_path


def create_fallback_ecoli_operons(output_dir: Path) -> Path:
    """
    Create fallback operon list for E. coli from well-characterized operons.

    These are classic E. coli operons used in textbooks and verified experimentally.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Well-characterized E. coli operons (K-12 MG1655 gene names)
    operons = [
        # Classic operons (textbook examples)
        ("lacZYA", ["lacZ", "lacY", "lacA"], "experimental"),
        ("trpEDCBA", ["trpE", "trpD", "trpC", "trpB", "trpA"], "experimental"),
        ("araBAD", ["araB", "araA", "araD"], "experimental"),
        ("araFGH", ["araF", "araG", "araH"], "experimental"),
        ("galETKM", ["galE", "galT", "galK", "galM"], "experimental"),
        ("hisGDCBHAFIE", ["hisG", "hisD", "hisC", "hisB", "hisH", "hisA", "hisF", "hisI", "hisE"], "experimental"),
        ("thrABC", ["thrA", "thrB", "thrC"], "experimental"),
        ("leuABCD", ["leuA", "leuB", "leuC", "leuD"], "experimental"),
        ("ilvGMEDA", ["ilvG", "ilvM", "ilvE", "ilvD", "ilvA"], "experimental"),
        ("ilvBN", ["ilvB", "ilvN"], "experimental"),
        ("pheA", ["pheA"], "experimental"),
        ("tyrA", ["tyrA"], "experimental"),

        # Ribosomal protein operons
        ("rpsJ", ["rpsJ", "rplC", "rplD", "rplW", "rplB", "rpsS", "rplV", "rpsC", "rplP", "rpmC", "rpsQ"], "ribosomal"),
        ("rpsL", ["rpsL", "rpsG", "fusA", "tufA"], "ribosomal"),
        ("rpoBC", ["rpoB", "rpoC"], "transcription"),
        ("rplKAJL", ["rplK", "rplA", "rplJ", "rplL"], "ribosomal"),
        ("rpmI", ["rpmI", "rplT"], "ribosomal"),

        # DNA replication
        ("dnaA", ["dnaA", "dnaN", "recF", "gyrB"], "replication"),
        ("oriC", ["gidA", "gidB", "mioC"], "replication"),

        # Cell division
        ("ftsQAZ", ["ftsQ", "ftsA", "ftsZ"], "division"),
        ("minCDE", ["minC", "minD", "minE"], "division"),

        # Energy metabolism
        ("atpIBEFHAGDC", ["atpI", "atpB", "atpE", "atpF", "atpH", "atpA", "atpG", "atpD", "atpC"], "ATP_synthase"),
        ("cydAB", ["cydA", "cydB"], "cytochrome"),
        ("cyoABCDE", ["cyoA", "cyoB", "cyoC", "cyoD", "cyoE"], "cytochrome"),
        ("nuoABCDEFGHIJKLMN", ["nuoA", "nuoB", "nuoC", "nuoD", "nuoE", "nuoF", "nuoG", "nuoH", "nuoI", "nuoJ", "nuoK", "nuoL", "nuoM", "nuoN"], "NADH_dehydrogenase"),

        # Carbohydrate metabolism
        ("malEFG", ["malE", "malF", "malG"], "maltose_transport"),
        ("malK-lamB-malM", ["malK", "lamB", "malM"], "maltose_transport"),
        ("manXYZ", ["manX", "manY", "manZ"], "mannose"),
        ("ptsHI-crr", ["ptsH", "ptsI", "crr"], "PTS"),
        ("glpFKX", ["glpF", "glpK", "glpX"], "glycerol"),
        ("glpTQ", ["glpT", "glpQ"], "glycerol"),
        ("glpABC", ["glpA", "glpB", "glpC"], "glycerol"),
        ("aceEF-lpdA", ["aceE", "aceF", "lpdA"], "pyruvate_dehydrogenase"),
        ("gapA", ["gapA"], "glycolysis"),
        ("pgk", ["pgk"], "glycolysis"),

        # Amino acid transport
        ("argT-hisJQMP", ["argT", "hisJ", "hisQ", "hisM", "hisP"], "histidine_transport"),
        ("livKHMGF", ["livK", "livH", "livM", "livG", "livF"], "BCAA_transport"),
        ("livJKHMGF", ["livJ"], "BCAA_transport"),

        # Stress response
        ("groESL", ["groES", "groEL"], "chaperone"),
        ("dnaKJ", ["dnaK", "dnaJ"], "chaperone"),
        ("clpPX", ["clpP", "clpX"], "protease"),
        ("clpA", ["clpA"], "protease"),
        ("lon", ["lon"], "protease"),
        ("rpoH", ["rpoH"], "heat_shock"),
        ("rpoS", ["rpoS"], "stationary"),
        ("rpoE-rseABC", ["rpoE", "rseA", "rseB", "rseC"], "envelope_stress"),

        # Cell wall
        ("murEFG", ["murE", "murF", "murG"], "peptidoglycan"),
        ("murBCD", ["murB", "murC", "murD"], "peptidoglycan"),
        ("mraYW-murD", ["mraY", "mraW"], "peptidoglycan"),
        ("pbpA-rodA", ["pbpA", "rodA"], "peptidoglycan"),

        # Iron metabolism
        ("entCEBAH", ["entC", "entE", "entB", "entA", "entH"], "enterobactin"),
        ("fepA", ["fepA"], "iron_transport"),
        ("fhuABCD", ["fhuA", "fhuB", "fhuC", "fhuD"], "ferrichrome"),
        ("tonB-exbBD", ["tonB", "exbB", "exbD"], "iron_transport"),

        # Flagella and motility
        ("flgBCDEFGHIJ", ["flgB", "flgC", "flgD", "flgE", "flgF", "flgG", "flgH", "flgI", "flgJ"], "flagella"),
        ("flgKL", ["flgK", "flgL"], "flagella"),
        ("fliAZY", ["fliA", "fliZ", "fliY"], "flagella"),
        ("fliE", ["fliE"], "flagella"),
        ("fliFGHIJK", ["fliF", "fliG", "fliH", "fliI", "fliJ", "fliK"], "flagella"),
        ("fliLMNOPQR", ["fliL", "fliM", "fliN", "fliO", "fliP", "fliQ", "fliR"], "flagella"),
        ("motAB-cheAW", ["motA", "motB", "cheA", "cheW"], "motility"),
        ("cheRBYZ", ["cheR", "cheB", "cheY", "cheZ"], "chemotaxis"),
        ("tar-tap-cheRBYZ", ["tar", "tap"], "chemotaxis"),

        # Secretion
        ("secYEG", ["secY", "secE", "secG"], "secretion"),
        ("secDF-yajC", ["secD", "secF", "yajC"], "secretion"),
        ("secA", ["secA"], "secretion"),

        # SOS response
        ("lexA", ["lexA"], "SOS"),
        ("recA", ["recA"], "SOS"),
        ("sulA", ["sulA"], "SOS"),
        ("uvrABC", ["uvrA", "uvrB", "uvrC"], "nucleotide_excision"),

        # Toxin-antitoxin systems (small cassettes)
        ("mazEF", ["mazE", "mazF"], "TA_system"),
        ("relBE", ["relB", "relE"], "TA_system"),
        ("hipBA", ["hipB", "hipA"], "TA_system"),
        ("chpABIK", ["chpA", "chpB", "chpI", "chpK"], "TA_system"),
        ("mqsRA", ["mqsR", "mqsA"], "TA_system"),
        ("dinJ-yafQ", ["dinJ", "yafQ"], "TA_system"),

        # Outer membrane proteins
        ("ompF", ["ompF"], "porin"),
        ("ompC", ["ompC"], "porin"),
        ("ompA", ["ompA"], "outer_membrane"),

        # Additional metabolic operons
        ("aceBAK", ["aceB", "aceA", "aceK"], "glyoxylate"),
        ("fumAC", ["fumA", "fumC"], "TCA"),
        ("sdhCDAB", ["sdhC", "sdhD", "sdhA", "sdhB"], "TCA"),
        ("sucABCD", ["sucA", "sucB", "sucC", "sucD"], "TCA"),
        ("icdA", ["icdA"], "TCA"),
        ("mdh", ["mdh"], "TCA"),
        ("gltBD", ["gltB", "gltD"], "glutamate"),
    ]

    output_path = output_dir / "operons.tsv"
    with open(output_path, 'w') as f:
        f.write("operon_id\tgenes\tgene_count\tevidence\n")
        for operon_id, genes, evidence in operons:
            f.write(f"{operon_id}\t{','.join(genes)}\t{len(genes)}\t{evidence}\n")

    print(f"  Created fallback E. coli operon list with {len(operons)} operons")
    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Downloading operon databases for ELSA benchmarking")
    print("=" * 60)

    # Download B. subtilis operons
    print("\n[1/2] B. subtilis operons (SubtiWiki)")
    print("-" * 40)
    bsub_path = download_subtiwiki_operons()
    if bsub_path:
        print(f"  B. subtilis operons: {bsub_path}")

    # Download E. coli operons
    print("\n[2/2] E. coli operons (RegulonDB)")
    print("-" * 40)
    ecoli_path = download_regulondb_operons()
    if ecoli_path:
        print(f"  E. coli operons: {ecoli_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    # Summary
    for org, path in [("B. subtilis", bsub_path), ("E. coli", ecoli_path)]:
        if path and path.exists():
            count = sum(1 for line in open(path) if not line.startswith('operon_id'))
            print(f"  {org}: {count} operons")


if __name__ == "__main__":
    main()
