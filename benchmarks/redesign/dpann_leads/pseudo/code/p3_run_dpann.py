"""Build configs for the three DPANN families (positions = curated Active-site + Binding-site residues of the reference
record) and run p2_readout. Writes pseudo/data/dpann_genomes.tsv and pseudo/data/<fam>_config.json."""
import json, subprocess, sys
import pandas as pd, duckdb
D = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads"
P = D + "/pseudo/data"
c = duckdb.connect("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb", read_only=True)
b = c.sql("select bin_id, taxonomy from bins").df()
b["order"] = b.taxonomy.str.extract(r"(o__[^;]*)")[0]
g = pd.read_parquet(P + "/dpann_groups_t0.75.parquet")[["genome", "group"]].rename(columns={"genome": "bin_id"})
b.merge(g, on="bin_id", how="left")[["bin_id", "group", "order"]].to_csv(P + "/dpann_genomes.tsv", sep="\t", index=False)


def feats(js):
    d = json.load(open(js)); s = d["sequence"]["value"]; out = {}
    for f in d["features"]:
        if f["type"] not in ("Active site", "Binding site"):
            continue
        lig = (f.get("ligand") or {}).get("name", "")
        kind = "active" if f["type"] == "Active site" else "binding:" + lig[:20]
        for p in range(f["location"]["start"]["value"], f["location"]["end"]["value"] + 1):
            if p not in out or kind == "active":
                out[p] = kind
    return [[p, s[p - 1], k] for p, k in sorted(out.items())]


fams = {
    "rpib": dict(title="RpiB-fold LF family, Micrarchaeales (DPANN); ref P37351 RPIB_ECOLI", ref_id="P37351",
                 js=D + "/rpib/data/ref/P37351.json", pdb=D + "/rpib/data/pdb/1NN4_A.pdb",
                 roles=dict(labeled="labeled", canonical="canonical", cdd_only="cdd_only", extra=["lf_annotated"])),
    "pepck": dict(title="PEPCK_GTP relatives (DPANN); ref A5TYT6 PCKG_MYCTA", ref_id="A5TYT6",
                  js=D + "/invisible/ref/A5TYT6.json", pdb=D + "/invisible/ref/4WPT_A_fam.pdb",
                  roles=dict(labeled="labeled", canonical="canonical", cdd_only="cdd_only", extra=[])),
    "rela": dict(title="RelA_SpoT small alarmone synthetase relatives (DPANN, mostly Woesearchaeales); ref O31611 YJBM_BACSU",
                 ref_id="O31611", js=D + "/gapscreen/ref/O31611.json", pdb=D + "/gapscreen/ref/5DEC_A.pdb",
                 roles=dict(labeled="labeled", canonical="canonical", cdd_only="cdd_only", extra=[])),
}
for fam, f in fams.items():
    if len(sys.argv) > 1 and fam not in sys.argv[1:]:
        continue
    cfg = dict(fam=fam, title=f["title"], aln=f"{P}/{fam}_aln.faa", meta=f"{P}/{fam}_meta.tsv", ref_id=f["ref_id"],
               positions=feats(f["js"]), roles=f["roles"], genomes_tsv=P + "/dpann_genomes.tsv", pdb=f["pdb"], out=P)
    json.dump(cfg, open(f"{P}/{fam}_config.json", "w"), indent=1)
    subprocess.run(["python", D + "/pseudo/code/p2_readout.py", f"{P}/{fam}_config.json"], check=True)
