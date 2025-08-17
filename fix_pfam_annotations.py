#!/usr/bin/env python3
"""
Quick fix to reprocess PFAM TSV results into proper genome_annotations format.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def extract_genome_id(protein_id, genome_files):
    """Extract genome ID from protein ID by matching against actual genome files."""
    # For patterns like 'accn|1313.30775.con.0001_10', extract the part after '|' and before '.con'
    if '|' in protein_id:
        after_pipe = protein_id.split('|', 1)[1]
        
        # Find the longest matching genome file (handles nested names)
        best_match = None
        best_length = 0
        
        for genome_file in genome_files:
            if after_pipe.startswith(genome_file) and len(genome_file) > best_length:
                best_match = genome_file
                best_length = len(genome_file)
        
        if best_match:
            return best_match
    
    # Fallback: try to extract before the first '.con' or '_'
    if '.con' in protein_id:
        before_con = protein_id.split('.con')[0]
        if '|' in before_con:
            return before_con.split('|', 1)[1]
        return before_con
    
    return protein_id.split('_')[0]  # ultimate fallback

def reprocess_pfam_results(tsv_file, json_file, genome_files):
    """Reprocess PFAM TSV into proper JSON format with genome_annotations."""
    
    print(f"Loading PFAM hits from: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"Found {len(df)} PFAM hits")
    
    print(f"Using genome files for ID mapping: {genome_files}")
    
    # Group by protein and collect domains
    protein_domains = defaultdict(list)
    
    for _, row in df.iterrows():
        protein_id = row['sequence_id']
        domain = row['hmm_name']
        protein_domains[protein_id].append(domain)
    
    # Group proteins by genome
    genome_annotations = defaultdict(dict)
    
    for protein_id, domains in protein_domains.items():
        genome_id = extract_genome_id(protein_id, genome_files)
        domain_string = ';'.join(sorted(set(domains)))  # Remove duplicates and sort
        genome_annotations[genome_id][protein_id] = domain_string
    
    print(f"Processed annotations for {len(genome_annotations)} genomes:")
    for genome_id in sorted(genome_annotations.keys()):
        protein_count = len(genome_annotations[genome_id])
        print(f"  {genome_id}: {protein_count} proteins")
    
    # Load existing JSON file and update it
    print(f"Loading existing JSON: {json_file}")
    with open(json_file, 'r') as f:
        existing_data = json.load(f)
    
    # Add genome_annotations field
    existing_data["genome_annotations"] = dict(genome_annotations)
    
    # Write updated JSON
    print(f"Writing updated JSON: {json_file}")
    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print("✓ PFAM annotations reprocessed successfully!")
    
    # Show sample
    first_genome = next(iter(genome_annotations.keys()))
    sample_proteins = list(genome_annotations[first_genome].keys())[:3]
    print(f"\nSample annotations for genome '{first_genome}':")
    for protein_id in sample_proteins:
        domains = genome_annotations[first_genome][protein_id]
        print(f"  {protein_id}: {domains}")

if __name__ == "__main__":
    # Use paths in the test directory
    base_dir = Path("genome_browser/pfam_annotations")
    tsv_file = base_dir / "PFAM_hits_df.tsv"
    json_file = base_dir / "pfam_annotation_results.json"
    
    if not tsv_file.exists():
        print(f"Error: TSV file not found: {tsv_file}")
        exit(1)
    
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        exit(1)
    
    # Discover genome files to use for ID mapping
    genomes_dir = Path("data/genomes")
    if genomes_dir.exists():
        genome_files = [f.stem for f in genomes_dir.glob("*.fna")]  # Remove .fna extension
        print(f"Found genome files: {genome_files}")
    else:
        print(f"Warning: Genomes directory not found: {genomes_dir}")
        genome_files = []
    
    reprocess_pfam_results(tsv_file, json_file, genome_files)