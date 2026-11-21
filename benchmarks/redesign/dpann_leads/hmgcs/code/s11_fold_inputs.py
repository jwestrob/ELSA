#!/usr/bin/env python3
"""Fold inputs. Single chains (plain ESMFold2): the 6 distinct lead sequences (3 sequence clusters among the 17), 3
GA-annotated members of the same ~560-aa Micrarchaeota family, 2 canonical DPANN HMGCS at GA (positive controls), 1
DPANN thiolase K00626 from a lead genome (same-superfamily control), 2 unrelated ~575-aa DPANN enzymes (negative
controls). Co-folds: homodimer + 2 acetyl-CoA for two leads and one canonical positive control, each with a family MSA
(mafft FFT-NS-2 of family members, written to msa/)."""
import os, subprocess, duckdb, pandas as pd
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
neg = con.execute("""SELECT DISTINCT a.protein_id, a.name FROM d.annotations a JOIN d.proteins pr USING(protein_id)
                     WHERE pr.bin_id='GCF_016806735.1' AND a.source='pfam' AND a.name IN ('Carbam_trans_N','ILVD_EDD_N') AND pr.sequence_length BETWEEN 560 AND 600""").df()
neg = neg.drop_duplicates("name").protein_id.tolist()
sets = {
    "lead": ["OJT94780.1", "JAJZWZ010000028.1_6", "DAIDNJ010000063.1_6", "MCL5680306.1", "JAJZNR010000018.1_8", "DAHXDO010000058.1_11"],
    "family_GA_pos": ["DAIDLN010000011.1_2", "DAHXET010000043.1_3", "DAHXQX010000030.1_19"],
    "canonical_HMGCS_pos": ["CAISET010000016.1_25", "AJF61242.1"],
    "thiolase_superfamily_ctrl": ["WP_203134943.1"],
    "unrelated_neg": neg,
}
ids = [p for v in sets.values() for p in v]
seq = con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))", [ids]).df().set_index("protein_id").sequence
os.makedirs(f"{W}/fold_in", exist_ok=True)
with open(f"{W}/fold_in/single.faa", "w") as fh:
    for p in ids:
        fh.write(f">{p}\n{seq[p].rstrip('*')}\n")
rows = [dict(protein_id=p, set=k, length=len(seq[p].rstrip('*')), ILVFMAWC=round(sum(c in 'ILVFMAWC' for c in seq[p]) / len(seq[p].rstrip('*')), 3)) for k, v in sets.items() for p in v]
pd.DataFrame(rows).to_csv(f"{W}/fold_in/fold_sets.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
# co-fold inputs, one file per protein, plus MSAs
os.makedirs(f"{W}/msa", exist_ok=True)
fam = con.execute("""SELECT f.protein_id, pr.sequence FROM d.protein_families f JOIN d.proteins pr USING (protein_id)
                     WHERE f.family_id='fam:5ad133d455:DAIDBP010000029.1_20' AND pr.sequence_length BETWEEN 480 AND 700""").df()
can = con.execute("""SELECT DISTINCT a.protein_id, pr.sequence FROM d.annotations a JOIN d.proteins pr USING (protein_id)
                     WHERE a.source='pfam' AND a.name='HMG_CoA_synt_N' AND pr.sequence_length BETWEEN 330 AND 380""").df()
def write_msa(df, query, out):
    tmp = f"{W}/msa/_in.faa"
    with open(tmp, "w") as fh:
        fh.write(f">{query}\n{seq[query].rstrip('*')}\n")
        for r in df.drop_duplicates("sequence").itertuples():
            if r.protein_id != query:
                fh.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
    res = subprocess.run(["mafft", "--auto", "--thread", "4", "--quiet", tmp], capture_output=True, text=True).stdout
    open(out, "w").write(res)
    os.remove(tmp)
    print(out, res.count(">"), "rows")
for q, df in [("OJT94780.1", fam), ("JAJZNR010000018.1_8", fam), ("CAISET010000016.1_25", can)]:
    nm = q
    os.makedirs(f"{W}/fold_in/cofold_{nm}", exist_ok=True)
    with open(f"{W}/fold_in/cofold_{nm}/{nm}.faa", "w") as fh:
        fh.write(f">{nm}\n{seq[q].rstrip('*')}\n")
    write_msa(df, q, f"{W}/fold_in/cofold_{nm}/{nm}.msa.faa")
