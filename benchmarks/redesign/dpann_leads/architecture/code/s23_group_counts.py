#!/usr/bin/env python3
"""Step 23: group-level filter counts appended to screen_counts_groups.tsv (groups / genes / genomes at each step)."""
import pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
G = pd.read_parquet("data/groups.parquet")
K = pd.read_parquet("data/known_form.parquet")
T = pd.read_csv("candidates_ranked.tsv", sep="\t")
C = pd.read_csv("candidates_curated.tsv", sep="\t")
def row(step, reps, note=""):
    a = A[A.rep.isin(reps)]
    return dict(step=step, groups=len(set(reps)), genes=len(a), genomes=a.bin_id.nunique(), note=note)
rows = [row("MMseqs2 groups of screen candidates (bidirectional coverage >= 0.7)", G.rep),
        row(">= 3 genomes in >= 2 genera", K.rep),
        row("  known form: DPANN GA carriers share the full-length architecture at >= 0.5x the labeled rate (dropped)", K[K.known_form].rep),
        row("known-form control passed", K[~K.known_form].rep, "of which the architecture is also in >= max(5, half as many) GA carriers: %d" % int((~K.known_form & K.arch_annotated_in_carriers).sum())),
        row(">= 3 sequence clusters at 95% identity", T[T.independent].rep),
        row("gene-call artifact control (separate adjacent extension-sized gene in <= 10% of short-form genomes and of members)", T[T.pass_all].rep),
        row("extra region hit in >= 50% of members (weak full Pfam i-E <= 1e-3 or CDD)", T[T.pass_all & T.extra_annotated].rep)]
for cat in ["JOIN", "DUP", "SAMEFAM", "DARK", "KNOWN", "REG", "MGE", "SURF"]:
    rows.append(row(f"curated category {cat}", C[C.category == cat].rep))
pd.DataFrame(rows).to_csv("screen_counts_groups.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
