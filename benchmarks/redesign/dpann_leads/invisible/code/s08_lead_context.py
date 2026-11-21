#!/usr/bin/env python
"""Step 8: family-level context for the passing families (unannotated stratum): neighbor recurrence within +-5 genes
(Pfam names), clade distribution with clade genome counts, function rates in the clade by any source, Leloir / module step
presence (lead genomes vs clade). Writes data/lead_family_context.json and data/neighbor_recurrence.tsv."""
import json, re
from collections import Counter
from pathlib import Path
import duckdb
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
L = pd.read_csv(H / "leads.tsv", sep="\t")
PA = pd.read_parquet(H / "data/population_amended.parquet")
GF = pd.read_parquet(H / "../gapscreen/data/genes_function.parquet", columns=["protein_id", "func", "tier", "genome", "order", "embedding_only"])
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=4")
bins = c.execute("SELECT bin_id, completeness, taxonomy FROM d.bins").fetchdf()
bins["order"] = bins.taxonomy.str.extract(r"o__([^;]*)")[0]
out, nbrows = {}, []
FAM = {"K00965": dict(kos=["K00965"], pf=["PF01087", "PF02744"], cog=["COG1085"],
                      steps={"galK K00849": ["K00849"], "galE K01784": ["K01784"], "galM K01785": ["K01785"],
                             "glgC K00975": ["K00975"], "glycogen synthase K00703/K16148/K20812": ["K00703", "K16148", "K20812"],
                             "UGP K00963": ["K00963"], "USP K12447": ["K12447"]}),
       "K01448": dict(kos=["K01448", "K01447", "K11066"], pf=["PF01520"], cog=["COG0860"], steps={}),
       "K00560": dict(kos=["K00560", "K13998"], pf=["PF00303"], cog=["COG0207"], steps={"thyX K03465": ["K03465"]}),
       "K02683": dict(kos=["K02683"], pf=["PF01896"], cog=["COG1467"], steps={"priL K18882": ["K18882"]})}
for f, spec in FAM.items():
    lead = L[(L.function.str.startswith(f)) & ~L.named_by_amendment]
    if lead.empty:
        continue
    ids = list(lead.lead)
    # neighbor recurrence (+-5) for passing genes and for all labeled genes of the function in the novel population
    def nbcount(pids):
        c.register("q", pd.DataFrame({"protein_id": pids}))
        nb = c.execute("""SELECT q.protein_id lead, p2.protein_id nb FROM q JOIN d.proteins p1 USING(protein_id)
           JOIN d.proteins p2 ON p2.contig_id = p1.contig_id AND abs(p2.gene_index - p1.gene_index) BETWEEN 1 AND 5""").fetchdf()
        c.register("nbq", nb)
        a = c.execute("""SELECT nbq.lead, a.name FROM nbq JOIN d.annotations a ON a.protein_id = nbq.nb WHERE a.source = 'pfam'""").fetchdf()
        cnt = Counter()
        for _, g in a.groupby("lead"):
            cnt.update(set(g.name))
        return cnt, len(pids)
    cl, nl = nbcount(ids)
    allg = list(GF[GF.func == f].protein_id)
    ca, na = nbcount(allg)
    for k, v in cl.most_common(12):
        nbrows.append(dict(func=f, pfam=k, passing=f"{v}/{nl}", all_labeled=f"{ca.get(k, 0)}/{na}"))
    # clade distribution and function rates
    orders = lead.clade.value_counts().to_dict()
    rec = dict(passing_genes=len(lead), passing_by_order=orders, labeled_all=len(allg),
               labeled_by_order=GF[GF.func == f].groupby("order").size().sort_values(ascending=False).head(8).to_dict(),
               neighbors_top=[f"{k} {v}/{nl} (all labeled {ca.get(k, 0)}/{na})" for k, v in cl.most_common(8)])
    kos = "','".join(spec["kos"]); pfs = "','".join(spec["pf"]); cogs = "','".join(spec["cog"])
    have = c.execute(f"""SELECT DISTINCT p.bin_id FROM d.proteins p WHERE EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = p.protein_id
        AND (a.accession IN ('{kos}') OR a.accession IN ('{pfs}'))) OR EXISTS (SELECT 1 FROM d.protein_resemblance r
        WHERE r.protein_id = p.protein_id AND r.accession IN ('{cogs}'))""").fetchdf()
    hv = set(have.bin_id)
    lead_genomes = set(lead.genome)
    for o in orders:
        ob = bins[bins.order == o]
        rec[f"function_rate_{o}"] = f"{len(set(ob.bin_id) & hv)}/{len(ob)} genomes carry the function by KO / Pfam GA / KEGG-linked COG"
    for s_, kk in spec["steps"].items():
        q = "','".join(kk)
        hb = set(c.execute(f"SELECT DISTINCT p.bin_id FROM d.proteins p JOIN d.annotations a USING(protein_id) WHERE a.accession IN ('{q}')").fetchdf().bin_id)
        o = max(orders, key=orders.get)
        ob = set(bins[bins.order == o].bin_id)
        rec[f"step {s_}"] = f"lead genomes {len(lead_genomes & hb)}/{len(lead_genomes)}; {o} {len(ob & hb)}/{len(ob)}"
    out[f] = rec
json.dump(out, open(H / "data/lead_family_context.json", "w"), indent=1)
pd.DataFrame(nbrows).to_csv(H / "data/neighbor_recurrence.tsv", sep="\t", index=False)
print(json.dumps(out, indent=1))
