#!/usr/bin/env python
"""ws_showcase2 context, part 2 (Mac, DuckDB one thread, read-only): observed annotations of member and flanking genes
from the republished Sharur DPANN database (12 Nov 2026 seal; the biotite copy named in ~/.claude/CLAUDE.md predates
the family_annotations and protein_resemblance tables).

Per gene: Pfam / KOfam / KEGG names (annotations), CDD (protein_resemblance, E <= 0.01), raw DefenseFinder / TXSScan
profile hits (annotations sources defensefinder, txsscan: observations, not calls), and curated-caller membership
(system_proteins joined to the caller of each system_source: DefenseFinder systems, TXSScan systems, CRISPRCasTyper,
HydDB classifications). Family level for members: family_annotations rows (HHsearch Pfam / PDB, family_only) of the
member's permissive-e3c50s75 family (protein_families), probability >= 50 %.
Inputs: work/context_genes.tsv (from sc2_context_genes.py). Writes analysis/context_annot.tsv,
analysis/context_summary.tsv, analysis/member_family_hits.tsv."""
import re
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
UNKNOWN = re.compile(r"DUF\d|UPF\d|unknown function|uncharacteri|hypothetical|function unknown|unknown protein", re.I)


def main():
    cg = pd.read_csv(HERE / "work/context_genes.tsv", sep="\t")
    con = duckdb.connect(DB, read_only=True, config={"threads": 1})
    con.register("w", pd.DataFrame(dict(protein_id=cg.protein_id.unique())))
    ann = con.execute("""select a.protein_id, a.source, a.accession, a.name from annotations a join w using (protein_id)
                         where a.source in ('pfam','kofam','kegg','defensefinder','txsscan','hyddb','hyddb_subgroup')""").fetchdf()
    cdd = con.execute("""select r.protein_id, r.accession, r.name, r.evalue from protein_resemblance r join w using (protein_id)
                         where r.evalue <= 0.01""").fetchdf()
    sysp = con.execute("""select s.protein_id, s.system_source, s.system_id, s.profile_name from system_proteins s
                          join w using (protein_id)""").fetchdf()
    dsys = con.execute("select * from defense_systems limit 0").fetchdf().columns.tolist()
    ssys = con.execute("select * from secretion_systems limit 0").fetchdf().columns.tolist()
    print("defense_systems cols", dsys)
    print("secretion_systems cols", ssys)
    print("system_sources", sysp.system_source.value_counts().to_dict())
    mem = cg[cg.member].protein_id.unique()
    con.register("m", pd.DataFrame(dict(protein_id=mem)))
    fam = con.execute("""select pf.protein_id, pf.family_id, fa.source, fa.msa_kind, fa.hit_accession, fa.hit_name,
                                fa.hit_description, fa.probability, fa.aligned_cols, fa.unknown_function
                         from protein_families pf join m using (protein_id)
                         join family_annotations fa on fa.family_id = pf.family_id and fa.clustering_run = pf.clustering_run
                         where fa.probability >= 50""").fetchdf()
    con.close()
    lab = {}
    for p, d in ann.groupby("protein_id"):
        lab[p] = "; ".join(sorted({f"{s}:{n}" for s, n in zip(d.source, d.name)}))[:400]
    cddl = {p: "; ".join(sorted(set(d.name.astype(str))))[:200] for p, d in cdd.groupby("protein_id")}
    sysl = {p: "; ".join(sorted({f"{s}:{pn}" for s, pn in zip(d.system_source, d.profile_name.astype(str))}))
            for p, d in sysp.groupby("protein_id")}
    cg["annotations"] = cg.protein_id.map(lab).fillna("")
    cg["cdd"] = cg.protein_id.map(cddl).fillna("")
    cg["curated_member"] = cg.protein_id.map(sysl).fillna("")
    cg.to_csv(HERE / "analysis/context_annot.tsv", sep="\t", index=False)
    fam.to_csv(HERE / "analysis/member_family_hits.tsv", sep="\t", index=False)
    # summary per candidate: members, flanks (block windows) vs control windows
    rows = []
    for cand, d in cg.groupby("cand"):
        nreal = d[d.kind == "block"].real.nunique()
        mem_d = d[d.member]
        fl = d[(d.kind == "block") & ~d.member]
        ct = d[(d.kind == "control") & (d.offset != 0)]
        def frac(x, col):
            return x.groupby("real")[col].apply(lambda s: (s != "").any()).mean() if len(x) else float("nan")
        def top(x, col, n=6):
            c = {}
            for v in x[col]:
                for t in [t.strip() for t in v.split(";") if t.strip()]:
                    c[t] = c.get(t, 0) + 1
            return "; ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:n])
        rows.append(dict(cand=cand, realisations=nreal,
                         members_annotated_frac=round(float((mem_d.annotations != "").mean()), 3),
                         members_cdd_frac=round(float((mem_d.cdd != "").mean()), 3),
                         members_curated=int((mem_d.curated_member != "").sum()),
                         member_annotations=top(mem_d, "annotations"), member_cdd=top(mem_d, "cdd"),
                         flank_windows_with_curated=round(float(frac(fl, "curated_member")), 3),
                         control_windows_with_curated=round(float(frac(ct, "curated_member")), 3),
                         flank_annotations=top(fl, "annotations", 10), flank_curated=top(fl, "curated_member", 6),
                         adjacent_same_strand_annotated=top(fl[(fl.offset.abs() == 1) & fl.same_strand], "annotations", 6)))
    S = pd.DataFrame(rows)
    S.to_csv(HERE / "analysis/context_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(S.T.to_string())


if __name__ == "__main__":
    main()
