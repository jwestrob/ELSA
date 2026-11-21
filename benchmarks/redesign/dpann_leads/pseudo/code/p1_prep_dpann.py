"""Prepare member tables and alignments for the three DPANN families (RpiB-fold LF, PEPCK_GTP, RelA_SpoT).
Roles: ref, labeled (positional calls of the family), canonical (family Pfam at GA, outside the labeled set and, for
RpiB, outside the LF slot), cdd_only (CDD best hit naming the family, no family Pfam at GA, not labeled, not an LF carrier).
Alignments: RpiB reuses the rpib workstream L-INS-i MSA and adds cdd_only with mafft --add --keeplength.
PEPCK / RelA: L-INS-i core (reference + up to 40 canonical, one per genome group + up to 20 labeled), then every
other sequence with mafft --add --keeplength. Writes pseudo/data/<fam>_meta.tsv, <fam>_aln.faa."""
import json, subprocess, random, os
import pandas as pd, duckdb
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
D = R + "/dpann_leads"
P = D + "/pseudo/data"
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
c = duckdb.connect(DB, read_only=True)
grp = pd.read_parquet(P + "/dpann_groups_t0.75.parquet")[["genome", "group"]].rename(columns={"genome": "bin_id"})
bins = c.sql("select bin_id, completeness, taxonomy from bins").df()
bins["order"] = bins.taxonomy.str.extract(r"(o__[^;]*)")[0]
bins["phylum"] = bins.taxonomy.str.extract(r"(p__[^;]*)")[0]
passt = pd.read_parquet(R + "/slot_rescue/pass/dpann_slot_rescue_pass.parquet", columns=["protein_id", "genome", "proposed_label_name"])
random.seed(17)


def seqs_for(pids):
    if not len(pids):
        return pd.DataFrame(columns=["protein_id", "bin_id", "sequence"])
    t = pd.DataFrame({"protein_id": list(pids)})
    c.register("t", t)
    return c.sql("select p.protein_id, p.bin_id, p.sequence from proteins p join t using(protein_id)").df()


def wfa(path, recs):
    with open(path, "w") as f:
        for i, s in recs:
            f.write(f">{i}\n{s.rstrip('*')}\n")


def ref_seq(js):
    return json.load(open(js))["sequence"]["value"]


def annotate(m):
    m = m.merge(grp, on="bin_id", how="left").merge(bins[["bin_id", "order", "phylum", "completeness"]], on="bin_id", how="left")
    return m


def family(fam, label, pfam_name, cdd_acc, refacc, refjson, trim_canonical):
    lab = passt[passt.proposed_label_name == label].protein_id.tolist()
    ga = c.sql(f"select protein_id, min(start_aa) s, max(end_aa) e from annotations where source='pfam' and name='{pfam_name}' group by 1").df()
    accs = ",".join(f"'{a}'" for a in cdd_acc)
    cdd = c.sql(f"select distinct protein_id from protein_resemblance where accession in ({accs})").df().protein_id
    cdd_only = sorted(set(cdd) - set(ga.protein_id) - set(lab))
    canon = sorted(set(ga.protein_id) - set(lab))
    S = pd.concat([seqs_for(lab).assign(role="labeled"), seqs_for(canon).assign(role="canonical"),
                   seqs_for(cdd_only).assign(role="cdd_only")])
    if trim_canonical:  # long multidomain proteins: keep the family envelope +-25 aa
        env = ga.set_index("protein_id")
        def trim(r):
            if r.role == "canonical" and r.protein_id in env.index:
                s0, e0 = env.loc[r.protein_id, ["s", "e"]]
                return r.sequence[max(0, s0 - 26): e0 + 25]
            return r.sequence
        S["sequence"] = S.apply(trim, axis=1)
    S = annotate(S)
    rs = ref_seq(refjson)
    meta = pd.concat([pd.DataFrame([dict(protein_id=refacc, role="ref", bin_id="", group=-1)]), S.drop(columns="sequence")])
    # core
    canon_rep = S[S.role == "canonical"].drop_duplicates("group")
    canon_rep = canon_rep.sample(min(40, len(canon_rep)), random_state=17)
    lab_rep = S[S.role == "labeled"].drop_duplicates("group")
    lab_rep = lab_rep.sample(min(20, len(lab_rep)), random_state=17)
    core = pd.concat([canon_rep, lab_rep])
    wfa(f"{P}/{fam}_core_in.faa", [(refacc, rs)] + list(zip(core.protein_id, core.sequence)))
    with open(f"{P}/{fam}_core.faa", "w") as f:
        subprocess.run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", "4", "--quiet", f"{P}/{fam}_core_in.faa"], stdout=f, check=True)
    rest = S[~S.protein_id.isin(core.protein_id)]
    wfa(f"{P}/{fam}_add.faa", list(zip(rest.protein_id, rest.sequence)))
    with open(f"{P}/{fam}_aln.faa", "w") as f:
        subprocess.run(["mafft", "--add", f"{P}/{fam}_add.faa", "--keeplength", "--thread", "4", "--quiet", f"{P}/{fam}_core.faa"], stdout=f, check=True)
    meta.to_csv(f"{P}/{fam}_meta.tsv", sep="\t", index=False)
    print(fam, meta.role.value_counts().to_dict())


def rpib():
    meta0 = pd.read_csv(D + "/rpib/data/msa_meta.tsv", sep="\t")
    pid2bin = c.sql("select protein_id, bin_id from proteins").df()
    rolemap = lambda g: ("labeled" if g.startswith("labeled_") else "lf_annotated" if g == "carrier_same_slot_as_labeled"
                         else "canonical" if g.startswith("carrier_other_slot") else "ref_other")
    meta0["role"] = meta0.group.map(rolemap)
    meta0.loc[meta0.id == "P37351", "role"] = "ref"
    meta0 = meta0.rename(columns={"id": "protein_id", "group": "msa_group"})
    lf = set(pd.read_csv(D + "/rpib/data/carriers_class.tsv", sep="\t").protein_id) | set(meta0.protein_id)
    lab = set(passt[passt.proposed_label_name == "LacAB_rpiB"].protein_id)
    ga = set(c.sql("select distinct protein_id from annotations where source='pfam' and name='LacAB_rpiB'").df().protein_id)
    cdd = set(c.sql("select distinct protein_id from protein_resemblance where accession in ('PRK05571','pfam02502','COG0698','TIGR00689')").df().protein_id)
    cdd_only = sorted(cdd - ga - lab - lf)
    S = seqs_for(cdd_only)
    wfa(P + "/rpib_add.faa", list(zip(S.protein_id, S.sequence)))
    with open(P + "/rpib_aln.faa", "w") as f:
        subprocess.run(["mafft", "--add", P + "/rpib_add.faa", "--keeplength", "--thread", "4", "--quiet", D + "/rpib/data/msa_aln.faa"], stdout=f, check=True)
    # full GA canonical set of DPANN (for the paralog test) comes from the MSA's other-slot group (Micrarchaeota)
    m = pd.concat([meta0[["protein_id", "role"]], pd.DataFrame({"protein_id": cdd_only, "role": "cdd_only"})])
    m = m.merge(pid2bin, on="protein_id", how="left")
    m = annotate(m)
    m.to_csv(P + "/rpib_meta.tsv", sep="\t", index=False)
    print("rpib", m.role.value_counts().to_dict())


if __name__ == "__main__":
    import sys
    rpib()
    if len(sys.argv) > 1 and sys.argv[1] == "rpib_only":
        sys.exit(0)
    family("pepck", "PEPCK_GTP", "PEPCK_GTP", ["PRK04210", "pfam00821", "cd00819"],
           "A5TYT6", D + "/invisible/ref/A5TYT6.json", trim_canonical=False)
    family("rela", "RelA_SpoT", "RelA_SpoT", ["cd05399", "pfam04607", "smart00954", "COG2357"],
           "O31611", D + "/gapscreen/ref/O31611.json", trim_canonical=True)
