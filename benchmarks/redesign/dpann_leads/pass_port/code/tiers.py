#!/usr/bin/env python
"""pass_port tier assignment (verification definitions, plus Pfam-clan level).

  T1            the gene's own weak hit (Aksha, E <= 10) to the proposed label's model
  T1c           no hit to the proposed model, but a hit to a label family in the same Pfam clan (sister family)
  contradicted  weak hits only to label families outside the proposed label's clan (or unclanned)
  T2            no weak hit; support (votes_for / supporting_genomes) >= 0.5 and vote share > 0.8
  T2b           no weak hit; support 0.2-0.5 and vote share > 0.8
  rest          everything else
tier_family = the original family-level tier (T1c counted as contradicted), identical to slot_rescue's verification tiers.
Clan of a Pfam = the public Pfam clan table (data/Pfam-A.clans.tsv, Sharur snapshot copy); clans of a KO
label or KO hit = clans of its proxy Pfams (the collection's ko_proxy.tsv).

usage: tiers.py --collection C                 (reads OUT/<C>_slot_rescue_pass.parquet, OUT/weakhit/weakhit_per_protein.parquet)
       tiers.py --dpann-check                  (DPANN: hit lists from slot_rescue check A raw output; tier_family must equal
                                                verify/verification_per_protein.parquet; reports the clan-level moves)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PP = RD + "/dpann_leads/pass_port"


def clan_maps(ko_proxy):
    cl = pd.read_csv(PP + "/data/Pfam-A.clans.tsv", sep="\t", header=None, names=["acc", "clan", "clan_id", "name", "desc"])
    clan = {a: c for a, c in zip(cl.acc, cl.clan) if isinstance(c, str) and c}
    kp = pd.read_csv(ko_proxy, sep="\t")
    kc = {}
    for k, pf in zip(kp.ko, kp.pf):
        if pf in clan:
            kc.setdefault(k, set()).add(clan[pf])
    return lambda lab: ({clan[lab]} if lab in clan else set()) if lab.startswith("PF") else kc.get(lab, set())


def assign(df, clans_of):
    s = df.votes_for / df.supporting_genomes
    pc = [clans_of(l) for l in df.label]
    sister = np.array([bool(c) and any(clans_of(h) & c for h in hl.split(";") if h and h != l)
                       for c, hl, l in zip(pc, df.hit_labels, df.label)])
    anyhit = df.hit_labels != ""
    base = np.where((s >= 0.5) & (df.vote_share > 0.8), "T2", np.where((s >= 0.2) & (df.vote_share > 0.8), "T2b", "rest"))
    fam = np.where(df.proposed_any, "T1", np.where(anyhit, "contradicted", base))
    tier = np.where(df.proposed_any, "T1", np.where(anyhit & sister, "T1c", np.where(anyhit, "contradicted", base)))
    return pd.DataFrame(dict(protein_id=df.protein_id, tier=tier, tier_family=fam, support=np.round(s, 4),
                             proposed_label_clans=[";".join(sorted(c)) for c in pc]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection")
    ap.add_argument("--dpann-check", action="store_true")
    a = ap.parse_args()
    if a.dpann_check:
        SR = RD + "/slot_rescue"
        p = pd.read_parquet(SR + "/pass/dpann_slot_rescue_pass.parquet",
                            columns=["protein_id", "proposed_label_positional_homology", "votes_for", "supporting_genomes", "vote_share"])
        p = p.rename(columns={"proposed_label_positional_homology": "label"})
        M = pd.read_csv(SR + "/verify/A/models/model_map.tsv", sep="\t")
        H = pd.read_csv(SR + "/verify/A/full/aksha/user_hmms_hits_df.tsv", sep="\t", usecols=["sequence_id", "hmm_name", "bitscore", "evalue"])
        H["label"] = H.hmm_name.map(dict(zip(M.model, M.label)))
        H = H.sort_values("bitscore", ascending=False).drop_duplicates(["sequence_id", "label"])
        H = H[H.evalue <= 10.0]
        hl = H.groupby("sequence_id").label.apply(lambda s: ";".join(s))
        p["hit_labels"] = p.protein_id.map(hl).fillna("")
        p["proposed_any"] = [l in set(h.split(";")) for l, h in zip(p.label, p.hit_labels)]
        T = assign(p, clan_maps(SR + "/data/ko_proxy.tsv"))
        v = pd.read_parquet(PP + "/data/verification_per_protein.parquet", columns=["protein_id", "tier"])  # copy of the local verify table
        m = T.merge(v, on="protein_id", suffixes=("", "_verification"))
        res = dict(rows=len(m), family_tier_equal=int((m.tier_family == m.tier_verification).sum()),
                   tiers=T.tier.value_counts().to_dict(), family_tiers=T.tier_family.value_counts().to_dict(),
                   contradicted_moved_to_T1c=int(((T.tier_family == "contradicted") & (T.tier == "T1c")).sum()))
        os.makedirs(PP + "/out/dpann", exist_ok=True)
        T.to_parquet(PP + "/out/dpann/tiers_clan.parquet", index=False)
        json.dump(res, open(PP + "/out/dpann/tiers_check.json", "w"), indent=1)
        print(json.dumps(res, indent=1))
        assert res["family_tier_equal"] == res["rows"], "family-level tiers differ from the verification tiers"
        return
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "collections.json")))[a.collection]
    out = cfg["out"].replace("RD/", RD + "/")
    p = pd.read_parquet(f"{out}/{a.collection}_slot_rescue_pass.parquet",
                        columns=["protein_id", "gid", "proposed_label_positional_homology", "votes_for", "supporting_genomes", "vote_share"])
    p = p.rename(columns={"proposed_label_positional_homology": "label"})
    w = pd.read_parquet(f"{out}/weakhit/weakhit_per_protein.parquet", columns=["gid", "hit_labels", "proposed_any"])
    p = p.merge(w, on="gid", how="left")
    p["hit_labels"] = p.hit_labels.fillna("")
    p["proposed_any"] = p.proposed_any.fillna(False).astype(bool)
    T = assign(p, clan_maps(cfg["ko_proxy"].replace("RD/", RD + "/")))
    T.to_parquet(f"{out}/tiers.parquet", index=False)
    s = dict(tiers=T.tier.value_counts().to_dict(), family_tiers=T.tier_family.value_counts().to_dict())
    json.dump(s, open(f"{out}/tiers_summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
