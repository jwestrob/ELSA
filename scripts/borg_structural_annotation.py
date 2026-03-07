#!/usr/bin/env python3
"""
Borg Backbone Structural Annotation Pipeline

Uses ESM3 Forge API to fold backbone proteins, then searches against
Foldseek databases (ESM Atlas, AlphaFold) for structural homologs.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from Bio import SeqIO
from tqdm import tqdm

# ESM3 imports
from esm.sdk import client
from esm.sdk.api import ESMProtein, GenerationConfig


@dataclass
class FoldseekHit:
    """A structural homolog hit from Foldseek."""
    target_id: str
    target_description: str
    evalue: float
    score: float
    identity: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int


class BorgStructuralAnnotator:
    """Pipeline for structural annotation of Borg proteins."""

    # Foldseek API endpoints
    FOLDSEEK_API = "https://search.foldseek.com/api"

    # Available databases (from https://search.foldseek.com/api/databases)
    DATABASES = {
        "esmatlas": "esm30_folddisco",  # ESM Atlas 30% clustered (metagenomic)
        "afdb": "afdb50",               # AlphaFold DB 50% clustered
        "afdb_swissprot": "afdb-swissprot",  # AlphaFold SwissProt (curated)
        "pdb": "pdb100",                # PDB 100%
    }

    def __init__(
        self,
        output_dir: Path,
        esm_api_key: Optional[str] = None,
        databases: list[str] = ["afdb", "afdb_swissprot"],
        num_steps: int = 8,  # ESM3 folding steps
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.pdb_dir = self.output_dir / "pdbs"
        self.pdb_dir.mkdir(exist_ok=True)
        self.results_dir = self.output_dir / "foldseek_results"
        self.results_dir.mkdir(exist_ok=True)

        # ESM3 Forge client
        token = esm_api_key or os.environ.get("ESM_API_KEY")
        if not token:
            raise ValueError("ESM_API_KEY not found in environment")
        self.forge = client(model="esm3-open-2024-03", token=token)
        self.fold_config = GenerationConfig(track="structure", num_steps=num_steps)

        # Foldseek databases
        self.databases = [self.DATABASES[db] for db in databases if db in self.DATABASES]

    def fold_protein(self, seq_id: str, sequence: str) -> Optional[Path]:
        """Fold a protein using ESM3 and save PDB."""
        pdb_path = self.pdb_dir / f"{seq_id}.pdb"

        # Skip if already folded
        if pdb_path.exists():
            return pdb_path

        # Clean sequence (remove stop codons, etc.)
        sequence = sequence.replace("*", "").replace("X", "A")

        if len(sequence) > 2000:
            print(f"  Warning: {seq_id} has {len(sequence)} aa, truncating to 2000")
            sequence = sequence[:2000]

        try:
            protein = ESMProtein(sequence=sequence)
            result = self.forge.generate(protein, self.fold_config)

            if result.coordinates is None:
                print(f"  Warning: No coordinates generated for {seq_id}")
                return None

            # Save PDB and ensure proper termination
            pdb_str = result.to_pdb_string()
            # Add TER and END if missing (required by Foldseek)
            if not pdb_str.strip().endswith("END"):
                if "TER" not in pdb_str:
                    pdb_str = pdb_str.rstrip() + "\nTER\n"
                pdb_str = pdb_str.rstrip() + "\nEND\n"

            with open(pdb_path, 'w') as f:
                f.write(pdb_str)

            return pdb_path

        except Exception as e:
            print(f"  Error folding {seq_id}: {e}")
            return None

    def submit_foldseek(self, pdb_path: Path) -> Optional[str]:
        """Submit PDB to Foldseek and return ticket ID."""
        url = f"{self.FOLDSEEK_API}/ticket"

        with open(pdb_path, 'rb') as f:
            files = {'q': f}
            data = {
                'mode': '3diaa',
                'database[]': self.databases,
            }
            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            return result.get('id')
        else:
            print(f"  Foldseek submit error: {response.status_code}")
            return None

    def check_foldseek_status(self, ticket_id: str) -> str:
        """Check status of Foldseek job."""
        url = f"{self.FOLDSEEK_API}/ticket/{ticket_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('status', 'UNKNOWN')
        return 'ERROR'

    def get_foldseek_results(self, ticket_id: str) -> Optional[dict]:
        """Get results from completed Foldseek job."""
        url = f"{self.FOLDSEEK_API}/result/{ticket_id}/0"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    def parse_foldseek_results(self, results: dict) -> list[FoldseekHit]:
        """Parse Foldseek results into structured hits."""
        hits = []

        if not results or 'results' not in results:
            return hits

        for db_result in results.get('results', []):
            for alignment in db_result.get('alignments', []):
                for aln in alignment:
                    try:
                        hit = FoldseekHit(
                            target_id=aln.get('target', ''),
                            target_description=aln.get('tDescription', ''),
                            evalue=float(aln.get('eval', 1e10)),
                            score=float(aln.get('score', 0)),
                            identity=float(aln.get('seqId', 0)),
                            query_start=int(aln.get('qStartPos', 0)),
                            query_end=int(aln.get('qEndPos', 0)),
                            target_start=int(aln.get('tStartPos', 0)),
                            target_end=int(aln.get('tEndPos', 0)),
                        )
                        hits.append(hit)
                    except (ValueError, KeyError) as e:
                        continue

        # Sort by e-value
        hits.sort(key=lambda h: h.evalue)
        return hits

    def search_structure(self, pdb_path: Path, seq_id: str) -> list[FoldseekHit]:
        """Search a structure against Foldseek databases."""
        results_path = self.results_dir / f"{seq_id}.json"

        # Check cache
        if results_path.exists():
            with open(results_path) as f:
                cached = json.load(f)
            return [FoldseekHit(**h) for h in cached]

        # Submit job
        ticket_id = self.submit_foldseek(pdb_path)
        if not ticket_id:
            return []

        # Poll for completion
        max_wait = 300  # 5 minutes max
        wait_time = 0
        while wait_time < max_wait:
            status = self.check_foldseek_status(ticket_id)
            if status == 'COMPLETE':
                break
            elif status == 'ERROR':
                print(f"  Foldseek error for {seq_id}")
                return []
            time.sleep(5)
            wait_time += 5

        if wait_time >= max_wait:
            print(f"  Foldseek timeout for {seq_id}")
            return []

        # Get results
        results = self.get_foldseek_results(ticket_id)
        hits = self.parse_foldseek_results(results)

        # Cache results
        with open(results_path, 'w') as f:
            json.dump([vars(h) for h in hits], f, indent=2)

        return hits

    def run(self, fasta_path: Path, max_proteins: Optional[int] = None) -> Path:
        """Run the full annotation pipeline."""
        print(f"Loading sequences from {fasta_path}")
        records = list(SeqIO.parse(fasta_path, 'fasta'))

        if max_proteins:
            records = records[:max_proteins]

        print(f"Processing {len(records)} proteins...")

        all_results = {}

        for record in tqdm(records, desc="Annotating"):
            seq_id = record.id.replace("/", "_").replace(" ", "_")[:50]
            sequence = str(record.seq)

            # Step 1: Fold
            pdb_path = self.fold_protein(seq_id, sequence)
            if not pdb_path:
                continue

            # Step 2: Search
            hits = self.search_structure(pdb_path, seq_id)

            # Store results
            all_results[seq_id] = {
                'sequence_length': len(sequence),
                'description': record.description,
                'num_hits': len(hits),
                'top_hits': [vars(h) for h in hits[:10]],
            }

            # Brief status
            if hits:
                top = hits[0]
                tqdm.write(f"  {seq_id}: {len(hits)} hits, top: {top.target_id} (E={top.evalue:.1e})")
            else:
                tqdm.write(f"  {seq_id}: no hits")

        # Save summary
        summary_path = self.output_dir / "annotation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Generate report
        report_path = self.output_dir / "annotation_report.md"
        self._generate_report(all_results, report_path)

        print(f"\nResults saved to {self.output_dir}")
        print(f"  Summary: {summary_path}")
        print(f"  Report: {report_path}")

        return summary_path

    def _generate_report(self, results: dict, output_path: Path):
        """Generate a markdown report of annotations."""
        with open(output_path, 'w') as f:
            f.write("# Borg Backbone Structural Annotation Report\n\n")

            # Summary stats
            total = len(results)
            with_hits = sum(1 for r in results.values() if r['num_hits'] > 0)

            f.write(f"## Summary\n\n")
            f.write(f"- Total proteins analyzed: {total}\n")
            f.write(f"- Proteins with structural hits: {with_hits} ({100*with_hits/total:.1f}%)\n")
            f.write(f"- Proteins with no hits: {total - with_hits}\n\n")

            # Top hits table
            f.write("## Top Hits by Protein\n\n")
            f.write("| Protein | Length | Hits | Top Hit | E-value | Description |\n")
            f.write("|---------|--------|------|---------|---------|-------------|\n")

            for seq_id, data in sorted(results.items()):
                if data['top_hits']:
                    top = data['top_hits'][0]
                    desc = top['target_description'][:40] + "..." if len(top['target_description']) > 40 else top['target_description']
                    f.write(f"| {seq_id[:30]} | {data['sequence_length']} | {data['num_hits']} | {top['target_id']} | {top['evalue']:.1e} | {desc} |\n")
                else:
                    f.write(f"| {seq_id[:30]} | {data['sequence_length']} | 0 | - | - | No structural homologs |\n")

            f.write("\n\n## Interpretation\n\n")
            f.write("Proteins with E-value < 1e-5 likely have true structural homologs.\n")
            f.write("Hits from ESM Atlas (metagenomic) may indicate related proteins from other uncultured organisms.\n")
            f.write("Hits from AlphaFold DB indicate homologs with known UniProt entries.\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Structural annotation of Borg backbone proteins")
    parser.add_argument("fasta", type=Path, help="Input FASTA file")
    parser.add_argument("-o", "--output", type=Path, default=Path("borg_structural_annotation"),
                        help="Output directory")
    parser.add_argument("-n", "--max-proteins", type=int, default=None,
                        help="Maximum proteins to process (for testing)")
    parser.add_argument("--databases", nargs="+", default=["afdb", "afdb_swissprot"],
                        choices=["esmatlas", "afdb", "afdb_swissprot", "pdb"],
                        help="Foldseek databases to search")
    parser.add_argument("--num-steps", type=int, default=8,
                        help="ESM3 folding steps (more = better but slower)")

    args = parser.parse_args()

    annotator = BorgStructuralAnnotator(
        output_dir=args.output,
        databases=args.databases,
        num_steps=args.num_steps,
    )

    annotator.run(args.fasta, max_proteins=args.max_proteins)


if __name__ == "__main__":
    main()
