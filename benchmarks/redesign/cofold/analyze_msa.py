"""Step 3 analysis: genome-paired MSA refolds, shuffled-pairing and non-cognate MSA controls, and the
monomer disorder-zipping check, against predeclared_criteria.json. Writes work/msa_step.tsv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"


def load(path):
    return {json.loads(l)["fold_id"]: json.loads(l) for l in open(path) if l.strip()}


def main():
    s1 = load(W / "folds_set1" / "metrics.jsonl")
    ms = load(W / "folds_msa" / "metrics.jsonl")
    pc = pd.read_csv(W / "per_candidate_folds.tsv", sep="\t").set_index("cand_rank")
    ranks = sorted({int(k.split("_")[2]) for k in ms if k.startswith("msa_cog")})
    rows = []
    for rk in ranks:
        for rep in (1, 2):
            cog = s1[f"cog_{rk:03d}_r{rep}"]
            r = dict(cand_rank=rk, rep=rep, screen_iptm=cog["iptm"], screen_iface_pae=cog["iface_pae"])
            for tag in ("msa_cog", "msa_shuf", "msa_non"):
                m = ms.get(f"{tag}_{rk:03d}_r{rep}")
                r[f"{tag}_iptm"] = m["iptm"] if m else np.nan
                r[f"{tag}_iface_pae"] = m["iface_pae"] if m else np.nan
                r[f"{tag}_contacts"] = m["contacts"] if m else np.nan
                r[f"{tag}_iface_plddt"] = m["interface_plddt"] if m else np.nan
            # disorder zipping: interface residues of the screen complex vs the monomer's own pLDDT
            mask = np.array(cog["iface_mask"], bool)
            LA = cog["lengths"][0]
            for ch, sl in (("A", slice(0, LA)), ("B", slice(LA, None))):
                mono = ms.get(f"mono_{rk:03d}_r{rep}_{ch}")
                if mono is None:
                    continue
                mp = np.array(mono["plddt_per_res"])
                cp = np.array(cog["plddt_per_res"])[sl]
                im = mask[sl]
                r[f"mono_{ch}_plddt"] = round(float(mp.mean()), 1)
                r[f"cplx_{ch}_plddt"] = round(float(cp.mean()), 1)
                r[f"iface_{ch}_n"] = int(im.sum())
                r[f"iface_{ch}_mono_lowconf_frac"] = round(float((mp[im] < 60).mean()), 3) if im.any() else np.nan
            r["zipping_flag"] = bool((r.get("iface_A_mono_lowconf_frac", 0) >= 0.5) or (r.get("iface_B_mono_lowconf_frac", 0) >= 0.5)
                                     or (r.get("cplx_A_plddt", 100) < 60) or (r.get("cplx_B_plddt", 100) < 60))
            rows.append(r)
    d = pd.DataFrame(rows)
    d["msa_delta"] = d.msa_cog_iptm - d.screen_iptm
    d["pairing_effect"] = d.msa_cog_iptm - d.msa_shuf_iptm
    d["msa_cog_minus_msa_non"] = d.msa_cog_iptm - d.msa_non_iptm
    d.to_csv(W / "msa_step.tsv", sep="\t", index=False)
    g = d.groupby("cand_rank").agg(screen_min=("screen_iptm", "min"), msa_min=("msa_cog_iptm", "min"),
                                   msa_max=("msa_cog_iptm", "max"), shuf_min=("msa_shuf_iptm", "min"),
                                   non_max=("msa_non_iptm", "max"), msa_pae_max=("msa_cog_iface_pae", "max"),
                                   zipping=("zipping_flag", "any"),
                                   monoA=("mono_A_plddt", "min"), monoB=("mono_B_plddt", "min"))
    g["msa_supported"] = g.msa_min >= 0.75
    g["msa_artefact"] = g.msa_min < 0.5
    g = g.join(pc[["hit_predeclared", "n_genomes", "n_ord", "n_phylum"]])
    g = g.sort_values(["hit_predeclared", "msa_min"], ascending=[False, False])
    g.to_csv(W / "msa_step_by_candidate.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())
    print("\nMSA effect (all 42 replicate folds): median msa-screen", round(float(d.msa_delta.median()), 3),
          "; median msa-shuffled", round(float(d.pairing_effect.median()), 3),
          "; median msa_cog-msa_non", round(float(d.msa_cog_minus_msa_non.median()), 3))
    print("msa_noncognate iptm >= 0.5:", int((d.msa_non_iptm >= 0.5).sum()), "of", len(d), "; >= 0.75:", int((d.msa_non_iptm >= 0.75).sum()))


if __name__ == "__main__":
    main()
