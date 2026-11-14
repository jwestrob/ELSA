"""Top-N dossier: per candidate, replicate coordinates, taxonomy spread, lengths, annotation status as
observed, fold metrics per replicate and control, MSA refold and monomer results when present, split-gene
check. Writes work/dossier_top.json and work/dossier_top.tsv. Usage: python dossier.py RANK [RANK ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
D = W / "data"


def load_metrics(path):
    if not Path(path).exists():
        return {}
    out = {}
    for line in open(path):
        m = json.loads(line)
        out[m["fold_id"]] = m
    return out


def main(ranks):
    pc = pd.read_csv(W / "per_candidate_folds.tsv", sep="\t").set_index("cand_rank").loc[ranks].reset_index()
    sg = pd.read_csv(W / "split_gene_check.tsv", sep="\t").set_index("cand_rank")
    man = pd.read_csv(W / "fold_manifest.tsv", sep="\t")
    genes = pd.read_parquet(D / "genes.parquet").set_index("gid")
    gm = pd.read_parquet(D / "genomes.parquet").set_index("genome")
    cand = pd.read_csv(W / "candidates_all.tsv", sep="\t").set_index("type_key")
    msa = load_metrics(W / "folds_msa" / "metrics.jsonl")
    s1 = load_metrics(W / "folds_set1" / "metrics.jsonl")
    out = []
    for _, r in pc.iterrows():
        rk = int(r.cand_rank)
        c = cand.loc[r.type_key]
        d = dict(cand_rank=rk, type_key=r.type_key, hit_predeclared=bool(r.hit_predeclared),
                 genomes=int(c.n_genomes), genera=int(c.n_genus), families=int(c.n_family), orders=int(c.n_ord),
                 classes=int(c["n_class"]), phyla=int(c.n_phylum), phyla_list=c.phyla, len_med=c.len_med,
                 fam_hyd_med=c.fam_hyd_med, intergenic_med_nt=c.intergenic_med_nt, same_strand_frac=float(c.same_strand_frac),
                 gap0_frac_min=float(c.gap0_frac_min), block_integrity=round(float(c.block_integrity), 3),
                 fam_dark_frac=c.fam_dark_frac, fam_copies_mean=c.fam_copies_mean, modules=c.modules,
                 completeness_med=float(c.completeness_med), edge_frac=round(float(c.edge_frac), 3))
        reps = []
        for rep in (1, 2):
            row = man[(man.cls == "cognate") & (man.cand_rank == rk) & (man.rep == rep)].iloc[0]
            gl = [int(x) for x in row.gids.split(";")]
            gg = genes.loc[gl]
            g0 = gm.loc[gg.genome.iloc[0]]
            m = s1.get(row.fold_id, {})
            rec = dict(rep=rep, genome=gg.genome.iloc[0], tax=g0.tax, completeness=float(g0.completeness),
                       contig=gg.contig.iloc[0], proteins=list(gg.protein_id), gids=gl,
                       coords=[f"{int(a)}-{int(b)}({'+' if s > 0 else '-'})" for a, b, s in zip(gg.start, gg.end, gg.strand)],
                       positions=[int(x) for x in gg.position_index], lengths=m.get("lengths"),
                       iptm=m.get("iptm"), pair_chains_iptm=m.get("pair_chains_iptm"), iface_pae=m.get("iface_pae"),
                       contacts=m.get("contacts"), interface_plddt=m.get("interface_plddt"), chain_plddt=m.get("chain_plddt"))
            for tag in ("non", "unc"):
                fid = f"{tag}_{rk:03d}_r{rep}"
                mm = s1.get(fid, {})
                rec[f"{tag}_iptm"] = mm.get("iptm")
                rec[f"{tag}_iface_pae"] = mm.get("iface_pae")
            for tag in ("msa_cog", "msa_shuf", "msa_non"):
                mm = msa.get(f"{tag}_{rk:03d}_r{rep}", {})
                rec[f"{tag}_iptm"] = mm.get("iptm")
                rec[f"{tag}_iface_pae"] = mm.get("iface_pae")
            for ch in ("A", "B"):
                mm = msa.get(f"mono_{rk:03d}_r{rep}_{ch}", {})
                rec[f"mono_{ch}_plddt"] = mm.get("chain_plddt", [None])[0] if mm else None
            reps.append(rec)
        d["replicates"] = reps
        if rk in sg.index:
            d["split_gene"] = sg.loc[rk].to_dict()
        out.append(d)
    json.dump(out, open(W / "dossier_top.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    flat = []
    for d in out:
        for rec in d["replicates"]:
            flat.append({**{k: v for k, v in d.items() if k not in ("replicates", "split_gene")}, **rec,
                         **{f"split_{k}": v for k, v in d.get("split_gene", {}).items()}})
    pd.DataFrame(flat).to_csv(W / "dossier_top.tsv", sep="\t", index=False)
    print(f"{len(out)} candidates")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
