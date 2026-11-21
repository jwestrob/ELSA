#!/usr/bin/env python3
"""Architecture screen, step 5: summarize MMseqs2 groups of screen candidates; conservation across genomes and genera;
known-form control against DPANN carriers of the label using the same evidence source (CDD best-per-region hits and
length). Writes data/groups.parquet and groups.tsv."""
import duckdb, pandas as pd, numpy as np, os
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
A = pd.read_parquet("data/screen_candidates.parquet")
SR = pd.read_parquet("data/screen_regions.parquet")
cl = pd.read_csv("data/cand_cluster.tsv", sep="\t", header=None, names=["rep", "protein_id"])
A = A.merge(cl, on="protein_id")
tax = A.taxonomy.str.split(";", expand=True)
A["phylum"], A["order"], A["family"], A["genus"] = tax[1], tax[3], tax[4], tax[5]
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('../../slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
labs = A[["acc", "kind"]].drop_duplicates()
con.register("labs", labs)
con.execute("""CREATE TEMP TABLE car AS
  SELECT DISTINCT l.acc, a.protein_id FROM labs l JOIN d.annotations a ON a.source='pfam' AND split_part(a.accession,'.',1)=l.acc AND l.kind='pfam'
  UNION SELECT DISTINCT l.acc, a.protein_id FROM labs l JOIN d.annotations a ON a.source='kofam' AND a.accession=l.acc AND l.kind='ko'""")
con.execute("DELETE FROM car WHERE protein_id IN (SELECT protein_id FROM pass)")
carcdd = con.execute("""SELECT c.acc, r."name" fam, count(DISTINCT c.protein_id) n FROM car c JOIN d.protein_resemblance r USING (protein_id) GROUP BY 1,2""").df()
carlen = con.execute("""SELECT c.acc, count(*) ncar, median(p.sequence_length) med FROM car c JOIN d.proteins p USING (protein_id) GROUP BY 1""").df()
carlong = con.execute("""SELECT c.acc, p.sequence_length len FROM car c JOIN d.proteins p USING (protein_id)""").df()
carlong = carlong.merge(carlen, on="acc"); carlong["long"] = (carlong.len >= 1.4 * carlong.med) & (carlong.len - carlong.med >= 100)
car_long_rate = carlong.groupby("acc").long.mean()
ccd = {(r.acc, r.fam): r.n for r in carcdd.itertuples()}
ncar = dict(zip(carlen.acc, carlen.ncar))
# labeled genes per label (denominator for the labeled rate)
nlab = pd.read_parquet("data/genes2.parquet", columns=["acc"]).acc.value_counts()
# clade sizes
bins = con.execute("SELECT bin_id, taxonomy FROM d.bins").df()
bt = bins.taxonomy.str.split(";", expand=True); bins["order"], bins["family"], bins["genus"] = bt[3], bt[4], bt[5]
osz = bins.order.value_counts(); fsz = bins.family.value_counts()
rows = []
for rep, g in A.groupby("rep"):
    lab = g.lname.value_counts().index[0]; acc = g.acc.value_counts().index[0]
    r = SR[SR.protein_id.isin(g.protein_id)]
    famc = Counter(); side = Counter()
    for pid, rr in r.groupby("protein_id"):
        gg = g.set_index("protein_id").loc[pid]
        for f in set(rr.fam): famc[f] += 1
        for _, x in rr.iterrows(): side[(x.fam, "N" if x.end_aa <= gg.L_from else ("C" if x.start_aa >= gg.L_to else "in"))] += 1
    top = famc.most_common(3)
    topfam = top[0][0] if top else ""
    topfrac = top[0][1] / len(g) if top else 0.0
    carrier_arch = ccd.get((acc, topfam), 0) if topfam else np.nan
    rows.append(dict(rep=rep, label=lab, acc=acc, genes=len(g), genomes=g.bin_id.nunique(), genera=g.genus.nunique(), families=g.family.nunique(),
        orders=";".join(f"{k}:{v}" for k, v in g.order.value_counts().head(3).items()), phyla=";".join(g.phylum.value_counts().index[:3]),
        top_order=g.order.value_counts().index[0], top_order_genomes=int(osz.get(g.order.value_counts().index[0], 0)),
        len_med=float(g.len.median()), len_min=int(g.len.min()), len_max=int(g.len.max()), car_med=float(g.car_med_len.iloc[0]), ncar=int(ncar.get(acc, 0)),
        L_from_med=float(g.L_from.median()), L_to_med=float(g.L_to.median()),
        pat_len=int(g.pat_len.sum()), pat_comp=int(g.pat_comp.sum()), pat_dup=int(g.pat_dup.sum()), t1=int((g.tier == "T1").sum()),
        support_med=float(g.support.median()), hyd_med=float(g.hyd.median()), emb_only=float((g.alignment_lists_same_label != "same").mean()),
        top_other=";".join(f"{f}:{n}" for f, n in top), top_other_frac=topfrac,
        top_side=";".join(f"{k[1]}:{v}" for k, v in side.items() if k[0] == topfam),
        carriers_with_top_other=carrier_arch, carrier_rate_top_other=(carrier_arch / ncar.get(acc, 1)) if topfam else np.nan,
        carrier_long_rate=float(car_long_rate.get(acc, np.nan)), labeled_of_label=int(nlab.get(acc, 0)),
        members=",".join(g.protein_id[:6])))
S = pd.DataFrame(rows).sort_values(["genomes", "genes"], ascending=False)
S["arch_rate_in_group"] = S.top_other_frac
S.to_parquet("data/groups.parquet", index=False)
A.to_parquet("data/screen_candidates_grouped.parquet", index=False)
cons = S[(S.genomes >= 3) & (S.genera >= 2)]
print("groups", len(S), "| groups with >= 3 genomes and >= 2 genera", len(cons), "genes", int(cons.genes.sum()))
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 400)
cons.to_csv("groups_conserved.tsv", sep="\t", index=False)
print(cons[["rep", "label", "genes", "genomes", "genera", "top_order", "len_med", "car_med", "pat_len", "pat_comp", "pat_dup", "t1", "top_other", "top_side", "carrier_rate_top_other", "carrier_long_rate"]].to_string(index=False))
