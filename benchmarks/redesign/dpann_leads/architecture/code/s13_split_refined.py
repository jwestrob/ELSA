#!/usr/bin/env python3
"""Step 13: refined gene-call artifact control for the lead groups. Changes from step 8:
  - the label domain is the union of the proposed-label weak hits and the CDD region that overlaps them most (step 8 used
    the weak hit alone, which for partial-coverage hits placed part of the label domain in the 'extra' region);
  - targets that are themselves members or GA carriers of the label are excluded (paralogs of the label domain).
  - window +-2 genes, any strand; MMseqs2 E <= 1e-3, >= 40% of the extra region aligned.
usage: s13_split_refined.py REP [REP ...]   -> data/split_refined.tsv
A partner gene counts only when shorter than 75% of the long form."""
import sys, duckdb, pandas as pd, numpy as np, os, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
reps = sys.argv[1:]
A = pd.read_parquet("data/screen_candidates_grouped.parquet"); A = A[A.rep.isin(reps)].copy()
R = pd.read_parquet("data/regions.parquet"); R = R[R.protein_id.isin(A.protein_id) & (R.source == "cdd")]
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
# label domain = union(weak label hit, best-overlapping CDD region)
LF, LT = {}, {}
for r in A.itertuples():
    lf, lt = r.L_from, r.L_to
    c = R[R.protein_id == r.protein_id]
    if len(c):
        ov = (np.minimum(c.end_aa, lt) - np.maximum(c.start_aa, lf)).clip(lower=0)
        if ov.max() > 0:
            b = c.iloc[int(np.argmax(ov.values))]; lf, lt = min(lf, b.start_aa), max(lt, b.end_aa)
    LF[r.protein_id], LT[r.protein_id] = lf, lt
A["D_from"] = A.protein_id.map(LF); A["D_to"] = A.protein_id.map(LT)
A["side"] = np.where(A.len - A.D_to >= A.D_from - 1, "C", "N")
A["x_from"] = np.where(A.side == "C", A.D_to + 1, 1).astype(int); A["x_to"] = np.where(A.side == "C", A.len, A.D_from - 1).astype(int)
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('../../slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
os.makedirs(f"{SP}/split2", exist_ok=True)
with open(f"{SP}/split2/q.faa", "w") as f:
    for r in A.itertuples():
        x = seqs[r.protein_id].rstrip("*")[r.x_from - 1:r.x_to]
        if len(x) >= 30: f.write(f">{r.protein_id}|{r.rep}\n{x}\n")
rows, excl = [], set(A.protein_id)
for rep, g in A.groupby("rep"):
    acc, kind = g.acc.iloc[0], g.kind.iloc[0]
    orders = list(g.taxonomy.str.split(";").str[3].unique())
    src = "pfam" if kind == "pfam" else "kofam"; accq = "split_part(a.accession,'.',1)=?" if kind == "pfam" else "a.accession=?"
    car = con.execute(f"""SELECT DISTINCT a.protein_id, p.bin_id, p.contig_id, p.gene_index, p.sequence_length len FROM d.annotations a JOIN d.proteins p USING (protein_id)
        WHERE a.source='{src}' AND {accq}""", [acc]).df()
    excl |= set(car.protein_id)
    bins = con.execute("SELECT bin_id, split_part(taxonomy,';',4) o FROM d.bins").df()
    car = car[car.bin_id.isin(bins[bins.o.isin(orders)].bin_id)]
    lab = con.execute("""SELECT s.protein_id, p.bin_id, p.contig_id, p.gene_index, p.sequence_length len FROM pass s JOIN d.proteins p USING (protein_id) JOIN d.bins b USING (bin_id)
        WHERE s.proposed_label_positional_homology=? AND split_part(b.taxonomy,';',4) IN (SELECT unnest(?))""", [acc, orders]).df()
    sf = pd.concat([car, lab]).drop_duplicates("protein_id")
    sf = sf[(sf.len < 1.2 * g.car_med_len.iloc[0]) & ~sf.protein_id.isin(A.protein_id)]; sf["rep"] = rep; sf["kind_"] = "short"
    mem = g[["protein_id", "bin_id", "contig_id", "gene_index", "len"]].copy(); mem["rep"] = rep; mem["kind_"] = "member"
    rows += [sf, mem]
F = pd.concat(rows, ignore_index=True)
con.register("F", F[["protein_id", "contig_id", "gene_index", "rep", "kind_"]])
N = con.execute("""SELECT F.protein_id focal, F.rep, F.kind_, p.protein_id nb, p.sequence FROM F JOIN d.proteins p
   ON p.contig_id=F.contig_id AND p.gene_index BETWEEN F.gene_index-2 AND F.gene_index+2 AND p.gene_index<>F.gene_index""").df()
N = N[~N.nb.isin(excl)]
with open(f"{SP}/split2/t.faa", "w") as f:
    for r in N.drop_duplicates("nb").itertuples(): f.write(f">{r.nb}\n{r.sequence.rstrip('*')}\n")
subprocess.run(["mmseqs", "easy-search", f"{SP}/split2/q.faa", f"{SP}/split2/t.faa", f"{SP}/split2/hits.m8", f"{SP}/split2/tmp", "-e", "1e-3", "-s", "7.5",
                "--threads", "4", "-v", "1", "--format-output", "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits"], check=True)
H = pd.read_csv(f"{SP}/split2/hits.m8", sep="\t", header=None, names="query target fident alnlen qstart qend qlen tstart tend tlen evalue bits".split())
H["qcov"] = (H.qend - H.qstart + 1) / H.qlen; H = H[H.qcov >= 0.4]; H["rep"] = H["query"].str.split("|").str[1]
# a split partner is a separate gene about the size of the extension: shorter than 75% of the long form (longer hits are
# other long-form copies next to a short paralog, which is a duplication, not a split)
mlen = dict(zip(A.protein_id, A.len)); H["mlen"] = H["query"].str.split("|").str[0].map(mlen)
H["long_copy"] = H.tlen >= 0.75 * H.mlen
print("hits to long-form copies (excluded):", int(H.long_copy.sum()), "| extension-sized partners:", int((~H.long_copy).sum()))
H = H[~H.long_copy]
hs = H.groupby("rep").target.apply(set).to_dict()
N["hit"] = [nb in hs.get(rep, set()) for nb, rep in zip(N.nb, N.rep)]
per = N.groupby(["rep", "kind_", "focal"]).hit.any().reset_index()
out = []
for rep, g in A.groupby("rep"):
    fs = F[F.rep == rep]; sh = fs[fs.kind_ == "short"]
    p = per[per.rep == rep]
    sp = set(p[(p.kind_ == "short") & p.hit].focal)
    out.append(dict(rep=rep, side=g.side.value_counts().index[0], D_from_med=g.D_from.median(), D_to_med=g.D_to.median(), x_from_med=g.x_from.median(), x_to_med=g.x_to.median(),
                    short_genomes=sh.bin_id.nunique(), short_genomes_split=sh[sh.protein_id.isin(sp)].bin_id.nunique(),
                    members=len(g), members_with_separate_copy=int(p[(p.kind_ == "member")].hit.sum())))
O = pd.DataFrame(out); O.to_csv("data/split_refined.tsv", sep="\t", index=False)
A[["protein_id", "rep", "len", "D_from", "D_to", "side", "x_from", "x_to"]].to_csv("data/lead_domains.tsv", sep="\t", index=False)
print(O.to_string(index=False))
