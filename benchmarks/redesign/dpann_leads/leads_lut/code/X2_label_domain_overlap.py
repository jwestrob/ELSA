#!/usr/bin/env python
"""Frame-challenge test 2 (lut): does the candidate's homology to its label-carrying partner cover the partner's LABEL
domain? Hypothesis 'domain-architecture mismatch': the partner is multi-domain, the candidate is homologous to a part of it
outside the label-Pfam envelope, so the candidate lacks the labeled domain (no Pfam row, no fold match to the family
reference). Expected under 'divergent family member': the aligned partner region covers most (>= 50%) of the label envelope.
Inputs: data/partner_homology.tsv (X1), partner label-Pfam envelopes (collection Pfam table, GA rows of the family label;
fetched on biotite), sequences as in X1. Same local alignment (BLOSUM62, -10/-0.5).
usage: X2_label_domain_overlap.py lut   writes data/label_domain_overlap.tsv; prints the comparison by fold match."""
import json, subprocess, sys
from pathlib import Path
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

C = sys.argv[1]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
RS = CFG["remote"].replace("RD/", RD + "/")
B, PY = "jwestrob@biotite.berkeley.edu", "/home/jwestrob/miniconda3/envs/elsa/bin/python"
T = pd.read_csv(H / "data/partner_homology.tsv", sep="\t", dtype={"protein_id": str})
pop = pd.read_parquet(H / "data/population.parquet", columns=["gid", "proposed_label_positional_homology", "proposed_label_name"])
T = T.merge(pop.rename(columns={"gid": "gid_", "proposed_label_positional_homology": "label"}).assign(protein_id=lambda d: d.gid_.astype(str)),
            on="protein_id")
q = T[["partner", "label"]].drop_duplicates()
q.to_csv(H / "data/x2_partner_labels.tsv", sep="\t", index=False)
subprocess.run(["rsync", "-a", str(H / "data/x2_partner_labels.tsv"), f"{B}:{RS}/data/"], check=True)
code = (f"import pandas as pd; q=pd.read_csv('{RS}/data/x2_partner_labels.tsv',sep='\\t'); "
        f"pf=pd.read_parquet('{RD}/annot_gpu/{C}/PFAM.parquet',columns=['gid','accession','name','env_from','env_to'],filters=[('gid','in',list(q.partner))]); "
        f"pf['label']=pf.accession.str.split('.').str[0]; pf.to_csv('{RS}/data/x2_partner_pfam.tsv',sep='\\t',index=False); print(len(pf))")
subprocess.run(["ssh", B, f"{PY} -c \"{code}\""], check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/x2_partner_pfam.tsv", str(H / "data") + "/"], check=True)
PF = pd.read_csv(H / "data/x2_partner_pfam.tsv", sep="\t")


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


SEQ = {**read_faa(H / "folds/seqs/candidates.faa"), **read_faa(H / "folds/seqs/x1_partners.faa")}
al = Align.PairwiseAligner(mode="local", open_gap_score=-10, extend_gap_score=-0.5)
al.substitution_matrix = substitution_matrices.load("BLOSUM62")
rows = []
for r in T.itertuples():
    c, p = SEQ.get(r.protein_id, ""), SEQ.get(str(r.partner), "")
    env = PF[(PF.gid == r.partner) & (PF.label == r.label)]
    doms = ";".join(f"{x.name}:{x.env_from}-{x.env_to}" for x in PF[PF.gid == r.partner].sort_values("env_from").itertuples())
    if not c or not p or env.empty:
        rows.append(dict(protein_id=r.protein_id, func=r.func, label=r.label, status="no_envelope" if p else "no_partner", partner_domains=doms)); continue
    a = al.align(c, p)[0]
    span = [(s2, e2) for (s1, e1), (s2, e2) in zip(*a.aligned)]
    p_lo, p_hi = span[0][0] + 1, span[-1][1]
    e = env.assign(span=env.env_to - env.env_from).sort_values("span", ascending=False).iloc[0]   # longest label envelope
    ov = max(0, min(p_hi, e.env_to) - max(p_lo, e.env_from) + 1) / (e.env_to - e.env_from + 1)
    rows.append(dict(protein_id=r.protein_id, func=r.func, label=r.label, label_name=r.proposed_label_name, assay=r.assay, fold_match=r.fold_match,
                     PASS=r.PASS, TM_ref=r.TM_ref, cand_len=len(c), partner_len=len(p), partner_aligned=f"{p_lo}-{p_hi}",
                     label_env=f"{e.env_from}-{e.env_to}", label_env_covered=round(ov, 3), partner_identity=r.partner_identity,
                     partner_domains=doms, status="ok"))
D = pd.DataFrame(rows)
D.to_csv(H / "data/label_domain_overlap.tsv", sep="\t", index=False)
ok = D[D.status == "ok"].copy()
ok["covers_label"] = ok.label_env_covered >= 0.5
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(D.status.value_counts().to_string())
print(ok.groupby(["assay", "fold_match"]).agg(n=("protein_id", "size"), covers_label=("covers_label", "sum"),
      env_cov_med=("label_env_covered", "median"), cand_len=("cand_len", "median"), partner_len=("partner_len", "median")).to_string())
print("all ok:", len(ok), "cover >= 50% of the partner's label envelope:", int(ok.covers_label.sum()))
print(ok.groupby("func").agg(n=("protein_id", "size"), label=("label_name", "first"), covers=("covers_label", "sum"), fold=("fold_match", "sum"),
      env_cov=("label_env_covered", "median"), cand_len=("cand_len", "median"), partner_len=("partner_len", "median"),
      doms=("partner_domains", "first")).sort_values("n", ascending=False).head(22).to_string())
