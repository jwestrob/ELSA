#!/usr/bin/env python3
"""Architecture screen, step 6: known-long-form control. Members of conserved groups (>= 3 genomes, >= 2 genera) are
searched (MMseqs2, E <= 1e-3, -s 7.5, alignment covering >= 70% of both sequences) against every DPANN carrier of their
label at the gathering threshold. A carrier that aligns full length to a member carries the same architecture.
known form: carriers with the architecture / carriers >= 0.5 x (group genes / labeled genes of the label), or carriers
with the architecture >= 0.5 x group genes. Writes data/known_form.parquet."""
import duckdb, pandas as pd, numpy as np, os, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
S = pd.read_csv("groups_conserved.tsv", sep="\t")
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
A = A[A.rep.isin(S.rep)]
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('../../slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
labs = A[["acc", "kind"]].drop_duplicates(); con.register("labs", labs)
car = con.execute("""WITH c AS (
  SELECT DISTINCT l.acc, a.protein_id FROM labs l JOIN d.annotations a ON a.source='pfam' AND split_part(a.accession,'.',1)=l.acc AND l.kind='pfam'
  UNION SELECT DISTINCT l.acc, a.protein_id FROM labs l JOIN d.annotations a ON a.source='kofam' AND a.accession=l.acc AND l.kind='ko')
  SELECT c.acc, c.protein_id, p.sequence, p.sequence_length len FROM c JOIN d.proteins p USING (protein_id)
  WHERE c.protein_id NOT IN (SELECT protein_id FROM pass)""").df()
print("carriers", len(car), "labels", car.acc.nunique())
os.makedirs(f"{SP}/kf", exist_ok=True)
seen = set()
with open(f"{SP}/kf/car.faa", "w") as f:
    for r in car.itertuples():
        if r.protein_id in seen: continue
        seen.add(r.protein_id); f.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
with open(f"{SP}/kf/q.faa", "w") as f:
    for pid in A.protein_id: f.write(f">{pid}\n{seqs[pid].rstrip('*')}\n")
cmd = ["mmseqs", "easy-search", f"{SP}/kf/q.faa", f"{SP}/kf/car.faa", f"{SP}/kf/hits.m8", f"{SP}/kf/tmp", "-e", "1e-3", "-s", "7.5",
       "-c", "0.7", "--cov-mode", "0", "--threads", "4", "-v", "1", "--max-seqs", "2000",
       "--format-output", "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits"]
subprocess.run(cmd, check=True)
H = pd.read_csv(f"{SP}/kf/hits.m8", sep="\t", header=None, names="query target fident alnlen qstart qend qlen tstart tend tlen evalue bits".split())
H.to_csv("data/known_form_hits.tsv.gz", sep="\t", index=False)
H = H.merge(A[["protein_id", "rep", "acc"]], left_on="query", right_on="protein_id")
H = H.merge(car[["acc", "protein_id"]].rename(columns={"protein_id": "target", "acc": "tacc"}), on="target")
H = H[H.acc == H.tacc]
ncar = car.groupby("acc").protein_id.nunique()
nlab = pd.read_parquet("data/genes2.parquet", columns=["acc"]).acc.value_counts()
k = H.groupby("rep").agg(carriers_fl=("target", "nunique"), carrier_fl_fident_med=("fident", "median")).reset_index()
S = S.merge(k, on="rep", how="left"); S["carriers_fl"] = S.carriers_fl.fillna(0).astype(int)
S["ncar_all"] = S.acc.map(ncar)
S["carrier_arch_rate"] = S.carriers_fl / S.ncar_all
S["labeled_arch_rate"] = S.genes / S.acc.map(nlab)
S["known_form"] = (S.carrier_arch_rate >= 0.5 * S.labeled_arch_rate) | (S.carriers_fl >= 0.5 * S.genes)
S.to_parquet("data/known_form.parquet", index=False)
print("conserved groups", len(S), "known form", int(S.known_form.sum()), "novel vs DPANN carriers", int((~S.known_form).sum()))
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 50); pd.set_option("display.max_rows", 400)
print(S[~S.known_form][["rep", "label", "genes", "genomes", "genera", "top_order", "len_med", "car_med", "L_from_med", "L_to_med", "pat_len", "pat_comp", "pat_dup",
      "top_other", "carriers_fl", "ncar_all", "carrier_long_rate", "hyd_med"]].to_string(index=False))
