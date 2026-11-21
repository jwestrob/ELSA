#!/usr/bin/env python3
"""Frame-critic test: is the ~560-aa family functionally split by HMGCR status? MSA (mafft FFT-NS-2) of all family
members (480-700 aa) plus the curated HMGS records; residues at the E. faecalis HMGS (Q9FD71) catalytic and
HMG-CoA/CoA-binding positions (UniProt FT; CoA positions from the archaeal record A0A384E143 mapped onto Q9FD71), tabulated
per genus against the genome's HMGCR status. Also: thiolase step by Pfam Thiolase_N (GA) among lipid genomes."""
import subprocess, duckdb, pandas as pd
from collections import OrderedDict, Counter
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
G = pd.read_parquet(f"{W}/genome_steps.parquet")
fam = con.execute("""SELECT DISTINCT f.protein_id, pr.bin_id, pr.sequence FROM d.protein_families f JOIN d.proteins pr USING (protein_id)
                     WHERE f.family_id IN ('fam:5ad133d455:DAIDBP010000029.1_20','fam:5ad133d455:DAHWNI010000033.1_6') AND pr.sequence_length BETWEEN 480 AND 700""").df()
refs = ["Q9FD71", "A0A384E143", "Q01581", "P40830", "Q6DNE9"]
with open(f"{W}/fam_msa_in.faa", "w") as fh:
    for a in refs:
        fh.write(f">{a}\n" + "".join(l.strip() for l in open(f"{W}/ref/{a}.faa") if not l.startswith(">")) + "\n")
    for r in fam.itertuples():
        fh.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
out = subprocess.run(["mafft", "--auto", "--thread", "4", "--quiet", f"{W}/fam_msa_in.faa"], capture_output=True, text=True).stdout
aln = OrderedDict(); cur = None
for l in out.splitlines():
    if l.startswith(">"): cur = l[1:].split()[0]; aln[cur] = ""
    else: aln[cur] += l.strip()
open(f"{W}/fam_msa.aln.faa", "w").write("".join(f">{k}\n{v}\n" for k, v in aln.items()))
def col(seqid, n):
    k = 0
    for i, c in enumerate(aln[seqid]):
        if c != "-":
            k += 1
            if k == n: return i
POS = OrderedDict([("E79_cat", 79), ("C111_cat", 111), ("H233_cat", 233), ("D29_hmg", 29), ("T152_hmg", 152), ("S201_hmg", 201),
                   ("P199_CoA(=arch R203)", 199), ("K238_CoA(=arch K243)", 238), ("K242_hmg(=arch K247)", 242), ("N275_hmg", 275), ("S308_hmg", 308)])
cols = {k: col("Q9FD71", v) for k, v in POS.items()}
print("reference residues at these columns:")
for a in refs:
    print(a, " ".join(f"{k.split('_')[0]}:{aln[a][c]}" for k, c in cols.items()))
fam["genus"] = fam.bin_id.map(G.genus); fam["fam"] = fam.bin_id.map(G.fam); fam["HMGCR"] = fam.bin_id.map(G["3_HMGCR"])
for k, c in cols.items():
    fam[k] = [aln[p][c] for p in fam.protein_id]
fam.drop(columns=["sequence"]).to_csv(f"{W}/pocket_residues_family.tsv", sep="\t", index=False)
print("\n# residue composition at each position, all members, split by genome HMGCR status")
for k in cols:
    a = Counter(fam[fam.HMGCR][k]); b = Counter(fam[~fam.HMGCR][k])
    print(f"{k:24s} HMGCR+ (n={fam.HMGCR.sum()}): {dict(a.most_common(4))}   HMGCR- (n={(~fam.HMGCR).sum()}): {dict(b.most_common(4))}")
print("\n# per genus (>= 5 members): modal residue at each position, HMGCR genomes / members")
gg = fam.groupby("genus").filter(lambda x: len(x) >= 5).groupby("genus")
T = gg.agg(members=("protein_id", "size"), HMGCR=("HMGCR", "sum"))
for k in cols:
    T[k.split("_")[0]] = gg[k].agg(lambda s: f"{s.mode().iloc[0]}{round(100 * (s == s.mode().iloc[0]).mean())}")
print(T.sort_values("members", ascending=False).to_string())
# thiolase step by Pfam Thiolase_N (GA), lipid genomes completeness >= 70
th = set(con.execute("SELECT DISTINCT pr.bin_id FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam' AND a.name='Thiolase_N'").df().bin_id)
hq = G[(G.completeness >= 70) & G.lipid_specific_any].copy(); hq["thiolase_pfam"] = hq.index.isin(th)
print("\n# thiolase step among lipid genomes (completeness >= 70): K00626 vs Pfam Thiolase_N (GA), percent")
print(hq.groupby("phylum").agg(n=("thiolase_pfam", "size"), K00626=("1_thiolase_AACT", lambda s: round(100 * s.mean())), Thiolase_N=("thiolase_pfam", lambda s: round(100 * s.mean()))).to_string())
