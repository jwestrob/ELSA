#!/usr/bin/env python3
"""Build v3 manuscript figures from canonical artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
MANUSCRIPT_DIR = BENCHMARKS_DIR / "evaluation" / "manuscript"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"

sys.path.append(str(SCRIPT_DIR))
from benchmark_utils import load_species_map, species_pair, attach_species


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig1_pipeline_overview(output_dir: Path) -> None:
    """Figure 1: Method overview schematic (gene-level chaining only)."""
    script = SCRIPT_DIR / "render_pipeline_figure.py"
    subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output_dir)],
        check=True,
    )


def fig2_divergence_scaling(
    elsa_blocks: pd.DataFrame, mcscanx_blocks: pd.DataFrame, output_dir: Path
) -> None:
    """Figure 2: Divergence scaling (blocks by species pair)."""
    species_map = load_species_map()
    elsa = attach_species(elsa_blocks, species_map)
    mc = attach_species(mcscanx_blocks, species_map)

    elsa["species_pair"] = elsa.apply(
        lambda r: species_pair(r["query_species"], r["target_species"]), axis=1
    )
    mc["species_pair"] = mc.apply(
        lambda r: species_pair(r["query_species"], r["target_species"]), axis=1
    )

    order = [
        ("ecoli-ecoli", "E. coli ↔ E. coli"),
        ("salmonella-salmonella", "Salmonella ↔ Salmonella"),
        ("klebsiella-klebsiella", "Klebsiella ↔ Klebsiella"),
        ("ecoli-salmonella", "E. coli ↔ Salmonella"),
        ("ecoli-klebsiella", "E. coli ↔ Klebsiella"),
        ("klebsiella-salmonella", "Klebsiella ↔ Salmonella"),
    ]

    elsa_counts = [int((elsa["species_pair"] == k).sum()) for k, _ in order]
    mc_counts = [int((mc["species_pair"] == k).sum()) for k, _ in order]
    ratio = [e / m if m else np.nan for e, m in zip(elsa_counts, mc_counts)]

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, elsa_counts, width, label="ELSA", color="#2ecc71")
    ax.bar(x + width / 2, mc_counts, width, label="MCScanX", color="#3498db")
    ax.set_ylabel("Block count")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in order], rotation=20, ha="right")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(x, ratio, color="#2c3e50", marker="o", linewidth=1.5)
    ax2.set_ylabel("ELSA / MCScanX")
    ax2.set_ylim(0, max(r for r in ratio if not np.isnan(r)) * 1.25)

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_divergence_scaling.png", dpi=300)
    fig.savefig(output_dir / "fig2_divergence_scaling.pdf")
    plt.close(fig)


def fig3_cosine_vs_identity(data_path: Path, output_dir: Path) -> bool:
    """Figure 3: Embedding cosine vs sequence identity."""
    if not data_path.exists():
        print(f"[skip] missing cosine/identity data: {data_path}")
        return False

    df = pd.read_csv(data_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        df["sequence_identity"],
        df["cosine_similarity"],
        c=df["same_orthogroup"].astype(int),
        cmap="coolwarm",
        alpha=0.6,
        s=12,
    )
    ax.set_xlabel("Sequence identity")
    ax.set_ylabel("Embedding cosine similarity")
    ax.set_title("Embeddings recover homology below alignment thresholds")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=["Different OG", "Same OG"],
        title="Orthogroup",
        loc="lower right",
    )
    ax.add_artist(legend)

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_cosine_vs_identity.png", dpi=300)
    fig.savefig(output_dir / "fig3_cosine_vs_identity.pdf")
    plt.close(fig)
    return True


def fig4_correspondence_density(
    density_path: Path, correspondence_path: Path, output_dir: Path
) -> bool:
    """Figure 4: Correspondence density / accidental span."""
    if not density_path.exists():
        print(f"[skip] missing correspondence data: {density_path}")
        return False

    df = pd.read_csv(density_path)
    has_corr = correspondence_path.exists()

    if has_corr:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.boxplot(
        [df["elsa_anchor_density"], df["mcscanx_anchor_density"]],
        tick_labels=["ELSA", "MCScanX"],
        patch_artist=True,
        boxprops=dict(facecolor="#2ecc71"),
        medianprops=dict(color="#2c3e50"),
    )
    ax.set_ylabel("Anchor density (anchors / span)")
    ax.set_title("Anchor density")

    if has_corr:
        corr = pd.read_csv(correspondence_path)
        strict = corr[corr["has_strict_block"] == True]
        order = [
            "accidental_span",
            "weak_correspondence",
            "partial_correspondence",
            "true_correspondence",
        ]
        counts = strict["classification"].value_counts().reindex(order, fill_value=0)
        ax2 = axes[1]
        ax2.bar(order, counts.values, color="#95a5a6")
        ax2.set_title("MCScanX strict hits (correspondence)")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_correspondence_density.png", dpi=300)
    fig.savefig(output_dir / "fig4_correspondence_density.pdf")
    plt.close(fig)
    return True


def fig5_genome_browser(image_path: Path, output_dir: Path) -> bool:
    if not image_path.exists():
        print(f"[skip] missing genome browser image: {image_path}")
        return False
    target = output_dir / "fig5_genome_browser.png"
    if image_path.resolve() != target.resolve():
        target.write_bytes(image_path.read_bytes())
    return True


def fig6_threshold_sweep(ecoli_path: Path, borg_path: Path, output_dir: Path) -> bool:
    """Figure 6: Threshold calibration sweep (E. coli + Borgs)."""
    if not ecoli_path.exists():
        print(f"[skip] missing threshold sweep summary: {ecoli_path}")
        return False

    ecoli = pd.read_csv(ecoli_path).sort_values("tau")
    if ecoli.empty:
        print(f"[skip] threshold sweep summary empty: {ecoli_path}")
        return False

    borg = pd.read_csv(borg_path).sort_values("tau") if borg_path.exists() else None

    if borg is not None and not borg.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_rec = axes[0, 0]
        ax_counts_ec = axes[0, 1]
        ax_counts_borg = axes[1, 0]
        ax_size_borg = axes[1, 1]

        ax_rec.plot(ecoli["tau"], ecoli["strict_recall"], marker="o", label="Strict")
        ax_rec.plot(ecoli["tau"], ecoli["independent_recall"], marker="o", label="Independent")
        ax_rec.plot(ecoli["tau"], ecoli["any_recall"], marker="o", label="Any")
        ax_rec.set_xlabel("Cosine threshold (tau)")
        ax_rec.set_ylabel("Operon recall")
        ax_rec.set_ylim(0, 1.0)
        ax_rec.set_title("E. coli recall vs τ")
        ax_rec.legend(frameon=False)

        ax_counts_ec.bar(ecoli["tau"], ecoli["n_blocks"], width=0.012, color="#95a5a6", label="Blocks")
        ax_counts_ec.set_xlabel("Cosine threshold (tau)")
        ax_counts_ec.set_ylabel("Block count")
        ax_counts_ec.set_title("E. coli blocks vs τ")
        ax_counts_ec.tick_params(axis="x", rotation=0)
        ax_counts_ec2 = ax_counts_ec.twinx()
        ax_counts_ec2.plot(ecoli["tau"], ecoli["n_anchors"], color="#2c3e50", marker="o", label="Anchors")
        ax_counts_ec2.set_ylabel("Anchor count")

        ax_counts_borg.bar(borg["tau"], borg["n_blocks"], width=0.012, color="#95a5a6", label="Blocks")
        ax_counts_borg.set_xlabel("Cosine threshold (tau)")
        ax_counts_borg.set_ylabel("Block count")
        ax_counts_borg.set_title("Borg blocks vs τ")
        ax_counts_borg.tick_params(axis="x", rotation=0)
        ax_counts_borg2 = ax_counts_borg.twinx()
        ax_counts_borg2.plot(borg["tau"], borg["n_anchors"], color="#2c3e50", marker="o", label="Anchors")
        ax_counts_borg2.set_ylabel("Anchor count")

        ax_size_borg.plot(borg["tau"], borg["median_block_size"], marker="o", color="#27ae60")
        ax_size_borg.set_xlabel("Cosine threshold (tau)")
        ax_size_borg.set_ylabel("Median block size (anchors)")
        ax_size_borg.set_title("Borg median block size vs τ")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_rec = axes[0]
        ax_counts_ec = axes[1]

        ax_rec.plot(ecoli["tau"], ecoli["strict_recall"], marker="o", label="Strict")
        ax_rec.plot(ecoli["tau"], ecoli["independent_recall"], marker="o", label="Independent")
        ax_rec.plot(ecoli["tau"], ecoli["any_recall"], marker="o", label="Any")
        ax_rec.set_xlabel("Cosine threshold (tau)")
        ax_rec.set_ylabel("Operon recall")
        ax_rec.set_ylim(0, 1.0)
        ax_rec.set_title("Recall vs similarity threshold")
        ax_rec.legend(frameon=False)

        ax_counts_ec.bar(ecoli["tau"], ecoli["n_blocks"], width=0.012, color="#95a5a6", label="Blocks")
        ax_counts_ec.set_xlabel("Cosine threshold (tau)")
        ax_counts_ec.set_ylabel("Block count")
        ax_counts_ec.set_title("Block count vs threshold")
        ax_counts_ec.tick_params(axis="x", rotation=0)
        ax_counts_ec2 = ax_counts_ec.twinx()
        ax_counts_ec2.plot(ecoli["tau"], ecoli["n_anchors"], color="#2c3e50", marker="o", label="Anchors")
        ax_counts_ec2.set_ylabel("Anchor count")

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_threshold_sweep.png", dpi=300)
    fig.savefig(output_dir / "fig6_threshold_sweep.pdf")
    plt.close(fig)
    return True


def build_manifest(
    output_dir: Path,
    inputs: dict[str, Path],
    manifest_path: Path,
    config_manifest: dict,
) -> None:
    artifacts = {name: str(path) for name, path in inputs.items() if path.exists()}
    hashes = {name: sha256_file(path) for name, path in inputs.items() if path.exists()}

    params = {
        "model": config_manifest["config"]["plm"]["model"],
        "embedding_input_dim": config_manifest["artifacts"]["pca_model"]["metadata"][
            "input_dim"
        ],
        "projection_dim": config_manifest["artifacts"]["pca_model"]["metadata"][
            "output_dim"
        ],
        "micro_chain_similarity_threshold": config_manifest["config"]["analyze"][
            "micro_chain"
        ]["similarity_threshold"],
    }

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "inputs": artifacts,
        "input_hashes": hashes,
        "params": params,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elsa-blocks",
        type=Path,
        default=BENCHMARKS_DIR
        / "results"
        / "cross_species_chain"
        / "micro_chain"
        / "micro_chain_blocks.csv",
    )
    parser.add_argument(
        "--mcscanx-blocks",
        type=Path,
        default=BENCHMARKS_DIR / "results" / "mcscanx_comparison" / "mcscanx_blocks_v2.csv",
    )
    parser.add_argument(
        "--cosine-identity",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "cosine_vs_identity.csv",
    )
    parser.add_argument(
        "--correspondence",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "anchor_density_summary.csv",
    )
    parser.add_argument(
        "--operon-correspondence",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "operon_correspondence_analysis.csv",
    )
    parser.add_argument(
        "--browser-image",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "figures" / "genome_browser.png",
    )
    parser.add_argument(
        "--threshold-sweep",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "threshold_sweep_summary.csv",
    )
    parser.add_argument(
        "--threshold-sweep-borg",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "threshold_sweep_borg_summary.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BENCHMARKS_DIR / "elsa_output" / "cross_species" / "MANIFEST.json",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    elsa_blocks = pd.read_csv(args.elsa_blocks)
    mcscanx_blocks = pd.read_csv(args.mcscanx_blocks)

    fig1_pipeline_overview(FIGURES_DIR)
    fig2_divergence_scaling(elsa_blocks, mcscanx_blocks, FIGURES_DIR)
    fig3_cosine_vs_identity(args.cosine_identity, FIGURES_DIR)
    fig4_correspondence_density(args.correspondence, args.operon_correspondence, FIGURES_DIR)
    fig5_genome_browser(args.browser_image, FIGURES_DIR)
    fig6_threshold_sweep(args.threshold_sweep, args.threshold_sweep_borg, FIGURES_DIR)

    if args.manifest.exists():
        config_manifest = load_manifest(args.manifest)
        inputs = {
            "elsa_blocks": args.elsa_blocks,
            "mcscanx_blocks": args.mcscanx_blocks,
            "cosine_identity": args.cosine_identity,
            "correspondence": args.correspondence,
            "operon_correspondence": args.operon_correspondence,
            "browser_image": args.browser_image,
            "threshold_sweep": args.threshold_sweep,
            "threshold_sweep_borg": args.threshold_sweep_borg,
            "manifest": args.manifest,
        }
        build_manifest(
            FIGURES_DIR,
            inputs,
            FIGURES_DIR / "manifest.json",
            config_manifest,
        )
    else:
        print(f"  Skipping manifest (not found: {args.manifest})")

    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
