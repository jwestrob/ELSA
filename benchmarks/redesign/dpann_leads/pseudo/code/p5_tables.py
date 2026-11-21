"""Combine the five family read-outs into pseudo/positions.tsv and pseudo/data/summary_all.tsv, adding the structural
residue maps of the earlier workstreams as the shift arbiter (gene-level counts, no sequences):
  rpib: rpib/data/struct_site_residues.tsv (13 LF models on 2VVP_A / 1NN4_A; Ec66, Ec99 and Mt-numbered D11->Ec9,
        H12->Ec10, R137->Ec133, R141->Ec137)
  pepck (K01596) and rela (K07816): invisible/data/residues_full.tsv (US-align structural map, own reference)
  glucdh3 (K06152) and pmmi (K20930): leads_srvp/data/ext/residues.tsv (US-align on the extended-tier reference)"""
import json
from collections import Counter
import pandas as pd
D = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads"
P = D + "/pseudo"
fams = ["rpib", "pepck", "glucdh3", "rela", "pmmi"]


def fmt(vals):
    c = Counter(vals); n = sum(c.values())
    return f"n={n} " + ",".join(f"{a}:{b}" for a, b in c.most_common(6)) if n else ""


struct = {}
s = pd.read_csv(D + "/rpib/data/struct_site_residues.tsv", sep="\t")
s = s[s.role.str.startswith("labeled")]
for col, pos in [("EcC66", 66), ("EcH99", 99), ("D11", 9), ("H12", 10), ("R137", 133), ("R141", 137)]:
    struct[("rpib", pos)] = fmt([str(v)[0] if isinstance(v, str) and v not in ("-",) else "-" for v in s[col]])
r = pd.read_csv(D + "/invisible/data/residues_full.tsv", sep="\t")
lab_rela = set(pd.read_csv(P + "/data/rela_genes.tsv", sep="\t").query("role=='labeled'").protein_id)
lab_pep = set(pd.read_csv(P + "/data/pepck_genes.tsv", sep="\t").query("role=='labeled'").protein_id)
for fam, func, labset in [("pepck", "K01596", lab_pep), ("rela", "K07816", lab_rela)]:
    x = r[(r.ref_func == func) & (r.pair == "own") & (r.role == "candidate")].drop_duplicates(["protein_id", "uniprot_pos"])
    struct[(fam, "overlap")] = f"{x.protein_id.nunique()} folded, {len(set(x.protein_id) & labset)} in the labeled set"
    for pos, g in x.groupby("uniprot_pos"):
        struct[(fam, int(pos))] = fmt(list(g.model_residue.fillna("-")))
e = pd.read_csv(D + "/leads_srvp/data/ext/residues.tsv", sep="\t")
for fam, func in [("glucdh3", "K06152"), ("pmmi", "K20930")]:
    x = e[(e.ref_func == func) & (e.pair == "own") & (e.role == "candidate")].drop_duplicates(["protein_id", "member_pos"])
    for pos, g in x.groupby("member_pos"):
        struct[(fam, int(pos))] = fmt(list(g.model_residue.fillna("-")))
    y = e[(e.ref_func == func) & (e.pair == "own") & (e.role == "positive")].drop_duplicates(["protein_id", "member_pos"])
    for pos, g in y.groupby("member_pos"):
        struct[(fam, int(pos), "pos")] = fmt(list(g.model_residue.fillna("-")))

rows, summ = [], []
for f in fams:
    p = pd.read_csv(f"{P}/data/{f}_positions.tsv", sep="\t")
    p["structure_readout_labeled_genes"] = [struct.get((f, int(x)), "") for x in p.position]
    p["structure_readout_positive_controls"] = [struct.get((f, int(x), "pos"), "") for x in p.position]
    rows.append(p)
    S = json.load(open(f"{P}/data/{f}_summary.json"))
    par = S.pop("paralog"); S.update({f"paralog_{k}": v for k, v in par.items()})
    ex = S.pop("extra_roles")
    for k, v in ex.items():
        S.update({f"{k}_{a}": b for a, b in v.items()})
    S["structure_overlap"] = struct.get((f, "overlap"), "")
    summ.append(S)
A = pd.concat(rows, ignore_index=True)
cols = ["family", "position", "kind", "canonical_residue", "top_substitute", "share_groups_top_substitute", "n_groups",
        "shift_rate", "divergence_control_canonical_share", "divergence_control_n_groups", "share_groups_canonical_labeled",
        "share_groups_gap", "labeled_residue_counts_groups", "canonical_share_groups", "canonical_n_groups", "verdict_position",
        "lf_annotated_canonical_share", "named_canonical_share", "named_n_groups", "structure_readout_labeled_genes",
        "structure_readout_positive_controls", "n_genes"]
A[[c for c in cols if c in A.columns]].to_csv(P + "/positions.tsv", sep="\t", index=False)
pd.DataFrame(summ).to_csv(P + "/data/summary_all.tsv", sep="\t", index=False)
print(pd.DataFrame(summ)[["family", "verdict", "n_labeled_groups", "verdict_positions", "positions_top_sub_ge70", "positions_top_sub_lt50",
                          "labeled_replaced_rate", "control_replaced_rate", "control_groups", "shift_rate_pooled", "own_invariant_n", "structure_overlap"]].to_string())
print(A[A.structure_readout_labeled_genes != ""][["family", "position", "canonical_residue", "structure_readout_labeled_genes", "structure_readout_positive_controls"]].to_string())
