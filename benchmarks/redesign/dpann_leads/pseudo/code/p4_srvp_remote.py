#!/usr/bin/env python
"""SR-VP families (biotite login node only; SR-VP sequences stay on biotite). Builds member tables, alignments and
configs for the gluconate 2-dehydrogenase gamma-subunit relatives (label Gluconate_2-dh3, PF13618) and the
phosphinomethylmalate-isomerase relatives (label Aconitase_C with partner function K20930), then runs p2_readout.py.
Roles: labeled (all labeled genes of the family in the pass table); canonical (gluconate: whole-proteome Pfam PF13618 at
GA, not labeled; K20930: extended-tier annotated members = KO K20930 at cut_ga + Pfam PF00694 at GA, label carriers
only); named (secondary control: the earlier assay's already-named stratum, labeled genes named by a relaxed own
KOfam/Pfam hit or CDD; duplicated rows, reported apart). Primary divergence control: no CDD table exists for non-labeled
SR-VP genes, so p2 falls back to identity-matched GA canonical members (rule addendum).
Positions: the extended-tier invariant site positions on the extended-tier reference member (site_ref.tsv).
usage: p4_srvp_remote.py   (writes RD/dpann_leads/pseudo/srvp/)"""
import json, os, subprocess
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
OUT = RD + "/dpann_leads/pseudo/srvp"
LS = RD + "/dpann_leads/leads_srvp"
PKG = RD + "/ws_srvp_bins/package/universe"
MAFFT = "/shared/software/bin/mafft"
os.makedirs(OUT, exist_ok=True)
genes = pd.read_parquet(PKG + "/genes.parquet", columns=["gid", "genome", "genome_code", "length"])
G = pd.read_parquet(PKG + "/genomes.parquet")
ocol = "ord" if "ord" in G.columns else "order"
grp = pd.read_parquet(RD + "/long_loci/frozen2/groups/emb_hom_average_t0.75.parquet")
gg = G[["genome", ocol]].rename(columns={ocol: "order"}).merge(grp[["genome", "group"]], on="genome", how="left")
gg.rename(columns={"genome": "bin_id"})[["bin_id", "group", "order"]].to_csv(OUT + "/srvp_genomes.tsv", sep="\t", index=False)
gene_meta = genes.merge(gg, on="genome", how="left").rename(columns={"genome": "bin_id"})
passt = pd.read_parquet(RD + "/dpann_leads/pass_port/out/srvp/srvp_slot_rescue_pass.parquet",
                        columns=["gid", "proposed_label_name", "partner_ko", "partner_function_strong"])
pf = pd.read_parquet(RD + "/annot_gpu/srvp/PFAM.parquet", columns=["gid", "accession", "env_from", "env_to"])
pf["pf"] = pf.accession.str.split(".").str[0]
members = pd.read_csv(LS + "/data/ext_members.tsv", sep="\t")
site_ref = pd.read_csv(LS + "/data/ext/site_ref.tsv", sep="\t") if os.path.exists(LS + "/data/ext/site_ref.tsv") else None
named = pd.read_csv(OUT + "/named_stratum.tsv", sep="\t")  # func, gid, named (copied from leads_srvp/data/population.parquet)


def load_seqs(wanted):
    wanted = set(int(x) for x in wanted); out = {}
    with open(PKG + "/proteins.faa") as f:
        keep = False
        for line in f:
            if line[0] == ">":
                k = int(line[1:].split()[0]); keep = k in wanted
                if keep:
                    out[k] = []
            elif keep:
                out[k].append(line.strip())
    return {k: "".join(v).rstrip("*") for k, v in out.items()}


def build(fam, title, lab_gids, canon_gids, func, ref_member, sites, pdb):
    lab_gids = sorted(set(lab_gids)); canon_gids = sorted(set(canon_gids) - set(lab_gids))
    nm = named[(named.func == func) & named.named].gid.astype(int).tolist()
    S = load_seqs(lab_gids + canon_gids + [ref_member])
    rows = [dict(protein_id=f"ref{ref_member}", role="ref", bin_id="", group=-1, order="")]
    gm = gene_meta.set_index("gid")
    for role, ids in (("labeled", lab_gids), ("canonical", canon_gids), ("named", [g for g in nm if g in lab_gids])):
        for g in ids:
            rows.append(dict(protein_id=str(g), role=role, bin_id=gm.loc[g, "bin_id"], group=gm.loc[g, "group"], order=gm.loc[g, "order"]))
    meta = pd.DataFrame(rows)
    meta.to_csv(f"{OUT}/{fam}_meta.tsv", sep="\t", index=False)
    import random
    random.seed(17)
    m1 = meta[meta.role.isin(["labeled", "canonical"])].drop_duplicates("protein_id")
    core = pd.concat([m1[m1.role == "canonical"].drop_duplicates("group").sample(frac=1, random_state=17).head(40),
                      m1[m1.role == "labeled"].drop_duplicates("group").sample(frac=1, random_state=17).head(20)])
    core = core[core.protein_id != str(ref_member)]
    with open(f"{OUT}/{fam}_core_in.faa", "w") as f:
        f.write(f">ref{ref_member}\n{S[ref_member]}\n")
        for g in core.protein_id:
            f.write(f">{g}\n{S[int(g)]}\n")
    with open(f"{OUT}/{fam}_core.faa", "w") as f:
        subprocess.run([MAFFT, "--localpair", "--maxiterate", "1000", "--thread", "4", "--quiet", f"{OUT}/{fam}_core_in.faa"], stdout=f, check=True)
    rest = [g for g in m1.protein_id if g not in set(core.protein_id)]
    with open(f"{OUT}/{fam}_add.faa", "w") as f:
        for g in rest:
            f.write(f">{g}\n{S[int(g)]}\n")
    with open(f"{OUT}/{fam}_aln.faa", "w") as f:
        subprocess.run([MAFFT, "--add", f"{OUT}/{fam}_add.faa", "--keeplength", "--thread", "4", "--quiet", f"{OUT}/{fam}_core.faa"], stdout=f, check=True)
    cfg = dict(fam=fam, title=title, aln=f"{OUT}/{fam}_aln.faa", meta=f"{OUT}/{fam}_meta.tsv", ref_id=f"ref{ref_member}",
               positions=sites, roles=dict(labeled="labeled", canonical="canonical", cdd_only="cdd_only", extra=["named"]),
               genomes_tsv=OUT + "/srvp_genomes.tsv", pdb=pdb, out=OUT)
    json.dump(cfg, open(f"{OUT}/{fam}_config.json", "w"), indent=1)
    print(fam, meta.role.value_counts().to_dict(), flush=True)
    subprocess.run(["/home/jwestrob/miniconda3/envs/elsa/bin/python", OUT + "/p2_readout.py", f"{OUT}/{fam}_config.json"], check=True)


sites = json.load(open(OUT + "/sites.json"))  # {func: {ref_member, positions [[pos, res, kind]]}}
# gluconate 2-DH gamma relatives
lab = passt[passt.proposed_label_name == "Gluconate_2-dh3"].gid.astype(int)
can = pf[pf.pf == "PF13618"].gid.astype(int)
build("glucdh3", "Gluconate 2-dehydrogenase gamma-subunit relatives (SR-VP), label Gluconate_2-dh3; extended-tier reference member",
      lab, can, "K06152", sites["K06152"]["ref_member"], sites["K06152"]["positions"], LS + "/usalign_ext/refs/K06152_ref.pdb")
# phosphinomethylmalate isomerase relatives
gf = pd.read_parquet(LS + "/data/genes_function.parquet", columns=["gid", "func"])  # partner-derived function (leads_srvp)
k = set(gf[gf.func == "K20930"].gid.astype(int))
lab = passt[(passt.proposed_label_name.isin(["Aconitase_C", "Aconitase"])) & passt.gid.astype(int).isin(k)].gid.astype(int)
can = members[members.func == "K20930"].gid.astype(int)
build("pmmi", "Phosphinomethylmalate isomerase relatives (SR-VP), label Aconitase_C, partner function K20930; extended-tier reference member",
      lab, can, "K20930", sites["K20930"]["ref_member"], sites["K20930"]["positions"], LS + "/usalign_ext/refs/K20930_ref.pdb")
