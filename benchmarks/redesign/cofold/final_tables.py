"""Deliverable tables in this directory (small) from work/:
candidates.tsv  every block type that passes recurrence (>= 5 genomes, >= 2 orders), all columns, with the
                split-gene check and fold summary joined for the 150 folded candidates
                (work/candidates_all.tsv keeps all 8,546 types with >= 3 genomes)
folds.tsv       every fold with its class and metrics (screen + MSA/monomer step)
class_distributions.tsv, per_candidate_folds.tsv copied from work/."""
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"

c = pd.read_csv(W / "candidates_all.tsv", sep="\t")
sg = pd.read_csv(W / "split_gene_check.tsv", sep="\t").drop(columns=["type_key"])
pc = pd.read_csv(W / "per_candidate_folds.tsv", sep="\t")
pc = pc[[x for x in pc.columns if x == "type_key" or x.startswith(("cog", "non", "unc", "delta", "c1_", "c123", "c4_",
                                                                     "promisc", "hit_", "min_cog"))]]
out = c[c.pass_recurrence].merge(sg.rename(columns={"cand_rank": "nonredundant_rank"}), on="nonredundant_rank", how="left")
out = out.merge(pc, on="type_key", how="left")
out["homology_dark"] = (out.homologs_A_annotated == 0) & (out.homologs_B_annotated == 0)
out.to_csv(HERE / "candidates.tsv", sep="\t", index=False)
f = pd.read_csv(W / "folds.tsv", sep="\t")
if (W / "folds_msa" / "metrics.jsonl").exists():
    rows = []
    for line in open(W / "folds_msa" / "metrics.jsonl"):
        m = json.loads(line)
        pcm = m.get("pair_chains_iptm") or [[None, None], [None, None]]
        rows.append(dict(fold_id=m["fold_id"], cls=m["fold_id"].split("_")[0] if m["fold_id"].startswith("mono") else
                         {"msa_cog": "msa_cognate", "msa_shuf": "msa_shuffled", "msa_non": "msa_noncognate"}["_".join(m["fold_id"].split("_")[:2])],
                         cand_rank=int(m["fold_id"].split("_")[2] if not m["fold_id"].startswith("mono") else m["fold_id"].split("_")[1]),
                         iptm=m["iptm"], ptm=m["ptm"], pc_ab=pcm[0][1] if len(pcm) > 1 else None,
                         pc_ba=pcm[1][0] if len(pcm) > 1 else None, iface_pae=m.get("iface_pae"),
                         contacts=m["contacts"], interface_plddt=m.get("interface_plddt"),
                         plddt_A=m["chain_plddt"][0], plddt_B=m["chain_plddt"][1] if len(m["chain_plddt"]) > 1 else None,
                         lengths=";".join(map(str, m["lengths"])), fold_dir="work/folds_msa"))
    f = pd.concat([f, pd.DataFrame(rows)], ignore_index=True)
if (W / "folds_extra" / "metrics.jsonl").exists():
    cls_of = {json.loads(l)["fold_id"]: json.loads(l)["cls"] for l in open(W / "jobs_extra.jsonl")}
    rows = []
    for line in open(W / "folds_extra" / "metrics.jsonl"):
        m = json.loads(line)
        rows.append(dict(fold_id=m["fold_id"], cls=cls_of[m["fold_id"]], iptm=m["iptm"], ptm=m["ptm"],
                         iface_pae=m.get("iface_pae"), contacts=m["contacts"], interface_plddt=m.get("interface_plddt"),
                         plddt_A=m["chain_plddt"][0], plddt_B=m["chain_plddt"][1] if len(m["chain_plddt"]) > 1 else None,
                         lengths=";".join(map(str, m["lengths"])), pair_chains_iptm=json.dumps(m.get("pair_chains_iptm")),
                         fold_dir="work/folds_extra"))
    f = pd.concat([f, pd.DataFrame(rows)], ignore_index=True)
f.to_csv(HERE / "folds.tsv", sep="\t", index=False)
for fn in ("msa_step.tsv", "msa_step_by_candidate.tsv", "zipping_calibration.tsv", "control_homology.tsv",
           "replicate_identity.tsv", "class_distributions.tsv", "per_candidate_folds.tsv", "split_gene_check.tsv", "dossier_top.tsv",
           "dossier_top.json", "fold_summary.json", "candidates_summary.json"):
    if (W / fn).exists():
        shutil.copy(W / fn, HERE / fn)
print(len(out), "candidate rows;", len(f), "fold rows")
