#!/usr/bin/env python3
"""
Data quality audit for ELSA pre-submission.

Checks:
1. Cosine similarity > 1.0 in output files (float16 rounding artifact)
2. Zero-valued orthogroup CSVs (ID mapping issues)
3. Cross-check key numbers against canonical output files

Usage:
    python benchmarks/scripts/audit_data_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_cosine_similarity(output_dir: Path) -> list:
    """Check for cosine similarity > 1.0 in chain scores and embeddings."""
    issues = []

    # Check genes.parquet embedding norms (should be ~1.0 after L2 normalization)
    parquet_paths = [
        output_dir / "benchmarks" / "elsa_output" / "cross_species" / "elsa_index" / "ingest" / "genes.parquet",
        output_dir / "benchmarks" / "elsa_output" / "ecoli" / "elsa_index" / "ingest" / "genes.parquet",
        output_dir / "elsa_index" / "ingest" / "genes.parquet",
    ]

    for pp in parquet_paths:
        if not pp.exists():
            continue
        df = pd.read_parquet(pp)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            continue

        embeddings = df[emb_cols].values.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1)

        # Check norms
        over_1 = np.sum(norms > 1.001)  # small tolerance for float16
        max_norm = float(np.max(norms))
        min_norm = float(np.min(norms))
        mean_norm = float(np.mean(norms))

        if over_1 > 0:
            issues.append({
                "file": str(pp.relative_to(output_dir)),
                "issue": f"{over_1} genes have L2 norm > 1.001 (max={max_norm:.6f})",
                "severity": "WARNING",
                "detail": f"Norms: mean={mean_norm:.6f}, min={min_norm:.6f}, max={max_norm:.6f}",
            })
        else:
            issues.append({
                "file": str(pp.relative_to(output_dir)),
                "issue": "PASS — no embedding norms > 1.001",
                "severity": "OK",
                "detail": f"Norms: mean={mean_norm:.6f}, min={min_norm:.6f}, max={max_norm:.6f}",
            })

        # Check for pairwise cosine similarities > 1.0 (sample)
        n_sample = min(1000, len(embeddings))
        idx = np.random.RandomState(42).choice(len(embeddings), n_sample, replace=False)
        sample = embeddings[idx]
        sample_norm = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9)
        dots = sample_norm @ sample_norm.T
        max_dot = float(np.max(dots))
        over_1_dots = int(np.sum(dots > 1.001))  # excluding self-similarity

        if over_1_dots > 0:
            issues.append({
                "file": str(pp.relative_to(output_dir)),
                "issue": f"Sampled pairwise cosine: {over_1_dots} pairs > 1.001 (max={max_dot:.6f})",
                "severity": "WARNING",
            })

    # Check chain_score in blocks CSVs
    # chain_score is CUMULATIVE (sum of per-anchor similarities), NOT a single cosine
    blocks_paths = list((output_dir / "benchmarks" / "results").glob("**/micro_chain_blocks.csv"))

    for bp in blocks_paths:
        df = pd.read_csv(bp, nrows=100)
        if "chain_score" in df.columns and "n_anchors" in df.columns:
            # Per-anchor similarity = chain_score / n_anchors
            per_anchor = df["chain_score"] / df["n_anchors"].clip(lower=1)
            max_per_anchor = float(per_anchor.max())
            over_1_pa = int((per_anchor > 1.001).sum())

            if over_1_pa > 0:
                issues.append({
                    "file": str(bp.relative_to(output_dir)),
                    "issue": f"Per-anchor similarity > 1.001 in {over_1_pa}/100 blocks (max={max_per_anchor:.6f})",
                    "severity": "WARNING",
                    "detail": "chain_score/n_anchors should be <= 1.0 for cosine similarity",
                })
            else:
                issues.append({
                    "file": str(bp.relative_to(output_dir)),
                    "issue": "PASS — per-anchor similarity <= 1.001",
                    "severity": "OK",
                    "detail": f"Max per-anchor similarity: {max_per_anchor:.6f}",
                })

    return issues


def check_correspondence_csvs(output_dir: Path) -> list:
    """Check orthogroup correspondence CSVs for all-zero issues."""
    issues = []

    files = {
        "elsa_correspondence.csv": "ELSA block orthogroup correspondence",
        "mcscanx_correspondence.csv": "MCScanX block orthogroup correspondence",
        "anchor_orthogroup_precision.csv": "Anchor orthogroup precision",
    }

    for fname, desc in files.items():
        fp = output_dir / "benchmarks" / "evaluation" / fname
        if not fp.exists():
            issues.append({
                "file": fname,
                "issue": f"File not found",
                "severity": "MISSING",
            })
            continue

        df = pd.read_csv(fp)
        issues.append({
            "file": fname,
            "issue": f"Loaded: {len(df)} rows, {len(df.columns)} columns",
            "severity": "INFO",
            "detail": f"Columns: {list(df.columns)}",
        })

        # Check for numeric columns and whether they're all zeros
        for col in df.select_dtypes(include=[np.number]).columns:
            n_nonzero = int((df[col] != 0).sum())
            n_total = len(df[col].dropna())
            pct_nonzero = n_nonzero / max(1, n_total) * 100

            if n_nonzero == 0:
                issues.append({
                    "file": fname,
                    "issue": f"Column '{col}' is ALL ZEROS ({n_total} rows)",
                    "severity": "CRITICAL",
                })
            elif pct_nonzero < 5:
                issues.append({
                    "file": fname,
                    "issue": f"Column '{col}': only {n_nonzero}/{n_total} non-zero ({pct_nonzero:.1f}%)",
                    "severity": "WARNING",
                })

    return issues


def cross_check_numbers(output_dir: Path) -> list:
    """Cross-check key reported numbers against actual data files."""
    issues = []

    checks = [
        {
            "claim": "76,954 total blocks (30-genome Enterobacteriaceae)",
            "expected": 76954,
            "file": "benchmarks/results/cross_species_chain/micro_chain/micro_chain_blocks.csv",
        },
        {
            "claim": "19,279 blocks (20-genome E. coli)",
            "expected": 19279,
            "file": "benchmarks/results/ecoli_chain/micro_chain_blocks.csv",
        },
    ]

    for check in checks:
        fp = output_dir / check["file"]
        if not fp.exists():
            issues.append({
                "claim": check["claim"],
                "issue": f"File not found: {check['file']}",
                "severity": "MISSING",
            })
            continue

        df = pd.read_csv(fp)
        actual = len(df)
        expected = check["expected"]

        if actual == expected:
            issues.append({
                "claim": check["claim"],
                "issue": f"MATCH — actual={actual:,}",
                "severity": "OK",
            })
        else:
            diff = actual - expected
            issues.append({
                "claim": check["claim"],
                "issue": f"MISMATCH — expected={expected:,}, actual={actual:,}, diff={diff:+,}",
                "severity": "CRITICAL" if abs(diff) > 10 else "WARNING",
            })

    # Check 8,069 ELSA-only cross-genus blocks
    # This requires loading both ELSA and MCScanX blocks and comparing
    cross_species_path = output_dir / "benchmarks" / "results" / "cross_species_chain" / "micro_chain" / "micro_chain_blocks.csv"
    if cross_species_path.exists():
        df = pd.read_csv(cross_species_path)
        # Count cross-genus blocks
        n_cross = 0
        for _, row in df.iterrows():
            qg = classify_genome(str(row["query_genome"]))
            tg = classify_genome(str(row["target_genome"]))
            if qg != tg:
                n_cross += 1

        issues.append({
            "claim": "53,271 cross-genus blocks",
            "issue": f"Counted: {n_cross:,} cross-genus blocks",
            "severity": "OK" if abs(n_cross - 53271) <= 10 else "WARNING",
        })

    # Check operon recall numbers by loading existing evaluation
    operon_eval_paths = [
        output_dir / "benchmarks" / "evaluation" / "operon_recall_comparison.md",
        output_dir / "benchmarks" / "evaluation" / "ELSA_vs_MCScanX_FULL_REPORT.md",
    ]
    for ep in operon_eval_paths:
        if ep.exists():
            issues.append({
                "claim": "Operon recall numbers (82.6% independent, 98.4% any, 47.2% strict)",
                "issue": f"Source file exists: {ep.name}",
                "severity": "INFO",
                "detail": "Manual verification recommended — numbers should match evaluation scripts",
            })
            break

    # Check 92.4% mean ortholog fraction
    block_val = output_dir / "benchmarks" / "evaluation" / "BLOCK_VALIDATION_RESULTS.md"
    if block_val.exists():
        with open(block_val) as f:
            content = f.read()
        if "92.4%" in content or "92.4" in content:
            issues.append({
                "claim": "92.4% mean ortholog fraction",
                "issue": "FOUND in BLOCK_VALIDATION_RESULTS.md",
                "severity": "OK",
            })
        else:
            issues.append({
                "claim": "92.4% mean ortholog fraction",
                "issue": "NOT FOUND in BLOCK_VALIDATION_RESULTS.md — verify manually",
                "severity": "WARNING",
            })

    return issues


# Reuse from search benchmark
def classify_genome(genome_id: str) -> str:
    # NCBI-verified taxonomy (21 E. coli, 5 Salmonella, 4 Klebsiella)
    SALMONELLA = ["GCF_000006945", "GCF_000007545", "GCF_000009505", "GCF_000022165", "GCF_000195995"]
    KLEBSIELLA = ["GCF_000016305", "GCF_000240185", "GCF_000733495", "GCF_000742755"]
    prefix = genome_id.rsplit(".", 1)[0]
    for p in SALMONELLA:
        if prefix.startswith(p):
            return "Salmonella"
    for p in KLEBSIELLA:
        if prefix.startswith(p):
            return "Klebsiella"
    return "E_coli"


def write_audit_report(
    cosine_issues: list,
    correspondence_issues: list,
    number_issues: list,
    output_dir: Path,
):
    """Write the data quality audit report."""
    md_path = output_dir / "benchmarks" / "evaluation" / "data_quality_audit.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    import time

    with open(md_path, "w") as f:
        f.write("# Data Quality Audit\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary counts
        all_issues = cosine_issues + correspondence_issues + number_issues
        n_critical = sum(1 for i in all_issues if i.get("severity") == "CRITICAL")
        n_warning = sum(1 for i in all_issues if i.get("severity") == "WARNING")
        n_ok = sum(1 for i in all_issues if i.get("severity") == "OK")
        n_missing = sum(1 for i in all_issues if i.get("severity") == "MISSING")

        f.write("## Summary\n\n")
        f.write(f"- CRITICAL: {n_critical}\n")
        f.write(f"- WARNING: {n_warning}\n")
        f.write(f"- OK: {n_ok}\n")
        f.write(f"- MISSING: {n_missing}\n\n")

        # Section 1: Cosine similarity
        f.write("## 1. Cosine Similarity Range Check\n\n")
        f.write("Checks that embedding norms and per-anchor similarities do not exceed 1.0.\n\n")
        f.write("**Note**: `chain_score` in blocks CSVs is a CUMULATIVE score (sum of per-anchor\n")
        f.write("similarities), NOT a single cosine value. Values like 867.0 for 867 anchors are\n")
        f.write("expected. The relevant metric is `chain_score / n_anchors` (per-anchor average).\n\n")

        for issue in cosine_issues:
            severity = issue.get("severity", "INFO")
            marker = {"OK": "PASS", "WARNING": "WARN", "CRITICAL": "FAIL"}.get(severity, severity)
            f.write(f"- **[{marker}]** `{issue.get('file', 'N/A')}`: {issue['issue']}\n")
            if "detail" in issue:
                f.write(f"  - {issue['detail']}\n")

        # Section 2: Correspondence CSVs
        f.write("\n## 2. Orthogroup Correspondence CSVs\n\n")
        for issue in correspondence_issues:
            severity = issue.get("severity", "INFO")
            marker = {"OK": "PASS", "WARNING": "WARN", "CRITICAL": "FAIL",
                       "MISSING": "MISS", "INFO": "INFO"}.get(severity, severity)
            f.write(f"- **[{marker}]** `{issue.get('file', 'N/A')}`: {issue['issue']}\n")
            if "detail" in issue:
                f.write(f"  - {issue['detail']}\n")

        f.write("\n### ID Mapping Fix (applied)\n\n")
        f.write("The MCScanX correspondence CSV previously showed 99.2% zeros due to two bugs:\n\n")
        f.write("1. **ID namespace mismatch**: MCScanX collinearity uses OrthoFinder internal IDs\n")
        f.write("   (e.g. `0_912`), but orthogroup lookup used protein accessions (e.g. `NP_*`).\n")
        f.write("   **Fix**: Translate internal IDs via `SequenceIDs.txt` before OG lookup.\n\n")
        f.write("2. **Collinearity parser bug**: Tab-delimited lines were split on whitespace,\n")
        f.write("   causing the block-pair index (`0:`) to be treated as a gene ID.\n")
        f.write("   **Fix**: Use tab-split instead of whitespace-split.\n\n")
        f.write("3. **ELSA annotation gap**: E. coli genomes (Prodigal) had no GFF annotations\n")
        f.write("   on this system, so coordinate-based matching only covered 10/30 genomes.\n")
        f.write("   **Fix**: Fall back to Prodigal FASTA headers for coordinate extraction.\n\n")
        f.write("Scripts fixed: `validate_pairs_orthogroups.py`, `analyze_gene_correspondence.py`\n")

        # Section 3: Number validation
        f.write("\n## 3. Key Number Cross-Check\n\n")
        for issue in number_issues:
            severity = issue.get("severity", "INFO")
            marker = {"OK": "PASS", "WARNING": "WARN", "CRITICAL": "FAIL",
                       "MISSING": "MISS", "INFO": "INFO"}.get(severity, severity)
            claim = issue.get("claim", "N/A")
            f.write(f"- **[{marker}]** {claim}: {issue['issue']}\n")
            if "detail" in issue:
                f.write(f"  - {issue['detail']}\n")

        # Section 4: What still needs attention
        f.write("\n## 4. Remaining Issues\n\n")

        if n_critical > 0:
            f.write("### Critical Issues (must fix before submission)\n\n")
            for issue in all_issues:
                if issue.get("severity") == "CRITICAL":
                    f.write(f"- {issue.get('file', '')}: {issue['issue']}\n")

        if n_warning > 0:
            f.write("\n### Warnings (should investigate)\n\n")
            for issue in all_issues:
                if issue.get("severity") == "WARNING":
                    f.write(f"- {issue.get('file', '')}: {issue['issue']}\n")

    print(f"[Audit] Wrote {md_path}", file=sys.stderr, flush=True)


def main():
    print("[Audit] Starting data quality audit...", file=sys.stderr, flush=True)

    cosine_issues = check_cosine_similarity(PROJECT_ROOT)
    print(f"[Audit] Cosine check: {len(cosine_issues)} items", file=sys.stderr, flush=True)

    correspondence_issues = check_correspondence_csvs(PROJECT_ROOT)
    print(f"[Audit] Correspondence check: {len(correspondence_issues)} items",
          file=sys.stderr, flush=True)

    number_issues = cross_check_numbers(PROJECT_ROOT)
    print(f"[Audit] Number check: {len(number_issues)} items", file=sys.stderr, flush=True)

    write_audit_report(cosine_issues, correspondence_issues, number_issues, PROJECT_ROOT)

    # Print summary
    all_issues = cosine_issues + correspondence_issues + number_issues
    n_critical = sum(1 for i in all_issues if i.get("severity") == "CRITICAL")
    n_warning = sum(1 for i in all_issues if i.get("severity") == "WARNING")
    print(f"\n=== Audit Summary ===", file=sys.stderr, flush=True)
    print(f"  CRITICAL: {n_critical}", file=sys.stderr, flush=True)
    print(f"  WARNING: {n_warning}", file=sys.stderr, flush=True)
    print(f"  Total checks: {len(all_issues)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
