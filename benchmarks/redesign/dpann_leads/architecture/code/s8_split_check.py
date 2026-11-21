#!/usr/bin/env python3
"""Architecture screen, step 8: gene-call artifact control for groups on the shortlist (data/shortlist.tsv: rep).
For each member the extra region is the part outside the label region on the longer side. Questions:
 (1) In genomes of the same orders that hold a short form of the label (GA carrier or labeled gene < 1.2x carrier
     median), is a gene within 2 positions homologous to the extra region (MMseqs2 E <= 1e-3, >= 40% of the extra region)?
     If yes, the long form corresponds to two separate genes elsewhere (a fusion, or a merged gene call).
 (2) Does a member's own genome hold a separate homolog of the extra region within 2 positions?
 (3) Member identity spread (independent assemblies) and distance of the gene to the contig ends.
Writes data/split_check.parquet and data/split_hits.tsv."""
import duckdb, pandas as pd, numpy as np, os, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
SL = pd.read_csv("data/shortlist.tsv", sep="\t")
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
A = A[A.rep.isin(SL.rep)].copy()
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
A["nlen"] = A.L_from - 1; A["clen"] = A.len - A.L_to
A["side"] = np.where(A.clen >= A.nlen, "C", "N")
A["x_from"] = np.where(A.side == "C", A.L_to + 1, 1); A["x_to"] = np.where(A.side == "C", A.len, A.L_from - 1)
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('../../slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
os.makedirs(f"{SP}/split", exist_ok=True)
qf = open(f"{SP}/split/q.faa", "w")
for r in A.itertuples():
    x = seqs[r.protein_id].rstrip("*")[int(r.x_from) - 1:int(r.x_to)]
    if len(x) >= 30: qf.write(f">{r.protein_id}|{r.rep}\n{x}\n")
qf.close()
# short forms in the same orders
rows = []
for rep, g in A.groupby("rep"):
    acc, kind = g.acc.iloc[0], g.kind.iloc[0]
    orders = list(g.taxonomy.str.split(";").str[3].unique())
    src = "pfam" if kind == "pfam" else "kofam"
    accq = "split_part(a.accession,'.',1)=?" if kind == "pfam" else "a.accession=?"
    sf = con.execute(f"""SELECT DISTINCT a.protein_id, p.bin_id, p.contig_id, p.gene_index, p.sequence_length len FROM d.annotations a JOIN d.proteins p USING (protein_id)
        JOIN d.bins b USING (bin_id) WHERE a.source='{src}' AND {accq} AND split_part(b.taxonomy,';',4) IN (SELECT unnest(?))""", [acc, orders]).df()
    lab = con.execute("""SELECT s.protein_id, p.bin_id, p.contig_id, p.gene_index, p.sequence_length len FROM pass s JOIN d.proteins p USING (protein_id) JOIN d.bins b USING (bin_id)
        WHERE s.proposed_label_positional_homology=? AND split_part(b.taxonomy,';',4) IN (SELECT unnest(?))""", [acc, orders]).df()
    sf = pd.concat([sf, lab]).drop_duplicates("protein_id")
    sf = sf[(sf.len < 1.2 * g.car_med_len.iloc[0]) & ~sf.protein_id.isin(A.protein_id)]
    sf["rep"] = rep; sf["kind_"] = "short"
    mem = g[["protein_id", "bin_id", "contig_id", "gene_index", "len"]].copy(); mem["rep"] = rep; mem["kind_"] = "member"
    rows += [sf, mem]
F = pd.concat(rows, ignore_index=True)
con.register("F", F[["protein_id", "contig_id", "gene_index", "rep", "kind_"]])
N = con.execute("""SELECT F.protein_id focal, F.rep, F.kind_, p.protein_id nb, p.gene_index - F.gene_index pos, p.sequence FROM F JOIN d.proteins p
   ON p.contig_id=F.contig_id AND p.gene_index BETWEEN F.gene_index-2 AND F.gene_index+2 AND p.gene_index<>F.gene_index""").df()
with open(f"{SP}/split/t.faa", "w") as f:
    for r in N.drop_duplicates("nb").itertuples(): f.write(f">{r.nb}\n{r.sequence.rstrip('*')}\n")
subprocess.run(["mmseqs", "easy-search", f"{SP}/split/q.faa", f"{SP}/split/t.faa", f"{SP}/split/hits.m8", f"{SP}/split/tmp", "-e", "1e-3", "-s", "7.5",
                "--threads", "4", "-v", "1", "--max-seqs", "5000", "--format-output", "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits"], check=True)
H = pd.read_csv(f"{SP}/split/hits.m8", sep="\t", header=None, names="query target fident alnlen qstart qend qlen tstart tend tlen evalue bits".split())
H["qcov"] = (H.qend - H.qstart + 1) / H.qlen
H = H[H.qcov >= 0.4]
H["rep"] = H["query"].str.split("|").str[1]
H.to_csv("data/split_hits.tsv", sep="\t", index=False)
hitset = H.groupby("rep").target.apply(set).to_dict()
N["hit"] = [nb in hitset.get(rep, set()) for nb, rep in zip(N.nb, N.rep)]
per = N.groupby(["rep", "kind_", "focal"]).hit.any().reset_index()
out = []
for rep, g in A.groupby("rep"):
    p = per[per.rep == rep]
    fs = F[(F.rep == rep)]
    sh = fs[fs.kind_ == "short"]; sh_split = set(p[(p.kind_ == "short") & p.hit].focal)
    me = p[(p.kind_ == "member")]
    pairs = H[H.rep == rep]
    out.append(dict(rep=rep, side=g.side.value_counts().index[0], x_len_med=float((g.x_to - g.x_from + 1).median()),
        short_genes=len(sh), short_genomes=sh.bin_id.nunique(), short_genomes_with_adjacent_extra=sh[sh.protein_id.isin(sh_split)].bin_id.nunique(),
        members=len(g), members_with_separate_adjacent_extra=int(me.hit.sum()), split_partner_fident_med=float(pairs.fident.median()) if len(pairs) else np.nan,
        near_contig_end=int(((g.start < 300) | (g.contig_len - g.end_coord < 300)).sum())))
O = pd.DataFrame(out)
# member identity spread from the all-vs-all
ava = pd.read_csv("data/cand_ava.m8.gz", sep="\t", header=None, names="query target fident alnlen qstart qend qlen tstart tend tlen evalue bits".split())
mm = A[["protein_id", "rep"]]
ava = ava.merge(mm.rename(columns={"protein_id": "query"}), on="query").merge(mm.rename(columns={"protein_id": "target", "rep": "rep2"}), on="target")
ava = ava[(ava.rep == ava.rep2) & (ava["query"] != ava["target"])]
idq = ava.groupby("rep").fident.quantile([0.1, 0.5, 0.9]).unstack(); idq.columns = ["fident_q10", "fident_q50", "fident_q90"]
O = O.merge(idq, left_on="rep", right_index=True, how="left")
O.to_parquet("data/split_check.parquet", index=False)
pd.set_option("display.width", 250)
print(O.to_string(index=False))
