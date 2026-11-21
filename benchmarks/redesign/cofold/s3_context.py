"""Screen 3: genomic context of passing blocks (+-3 genes around every instance) with same-contig controls.

For every instance of each block type in work/s3/context_types.txt (census_instances rows), the genes at offsets
-3..-1 before the block's first gene and +1..+3 after its last gene in transcription order of the block (offsets
are in block orientation: -1 = before chain A; offset_tx is the same offset in transcription order, negative = 5'), with: strand relative to the block, length, token,
curated annotation (Pfam names, KOfam ids, DefenseFinder / TXSScan / CRISPRCasTyper membership), CDD best names,
tier resemblance of the gene's own searched family (characterised / unknown-function names, P >= 90 % over >= 30
columns). Observations only.
'Recognised neighbour' = curated annotation, or a CDD model / characterised tier hit (as observed on that gene).
Control windows: per instance up to 3 windows of the same number of genes on the same contig, >= 6 genes from the
block, whose genes are unannotated (as the block's are); the same neighbour statistics.
Writes work/s3/context_neighbours.tsv, context_summary.tsv.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W3 = HERE / "work" / "s3"
D3 = W3 / "data"
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"


def main():
    types = [x.strip() for x in open(W3 / "context_types.txt") if x.strip()]
    inst = pd.read_parquet(W3 / "census" / "census_instances.parquet")
    inst = inst[inst.type_key.isin(types)].reset_index(drop=True)
    g = pd.read_parquet(HERE / "work" / "data" / "genes.parquet", columns=["gid", "protein_id", "contig_code", "strand", "position_index"])
    fl = pd.read_parquet(D3 / "prot_flags_slim.parquet")
    g = g.merge(fl, on="protein_id", how="left")
    pf = pd.read_parquet(D3 / "prot_fams.parquet")[["protein_id", "fam_e3"]]  # local copy slimmed to these columns
    g = g.merge(pf, on="protein_id", how="left")
    assert (g.gid.values == np.arange(len(g))).all()
    cc = g.contig_code.to_numpy()
    strand = g.strand.to_numpy()
    rng = np.random.default_rng(17)
    rows = []
    for _, r in inst.iterrows():
        gl = list(r.gids)
        fwd = gl[-1] > gl[0] if len(gl) > 1 else True
        bstrand = int(strand[gl[0]])
        lo, hi = min(gl), max(gl)
        # neighbours in block orientation
        def neigh(lo_, hi_, fwd_):
            out = []
            for off in (1, 2, 3):
                up = lo_ - off if fwd_ else hi_ + off
                dn = hi_ + off if fwd_ else lo_ - off
                for o, x in ((-off, up), (off, dn)):
                    if 0 <= x < len(g) and cc[x] == cc[lo_]:
                        out.append((o, x))
            return out
        for o, x in neigh(lo, hi, fwd):
            same = (fwd == (bstrand > 0))
            rows.append(dict(type_key=r.type_key, instance=f"{r.genome}:{r.contig_code}:{r.pos_start}", kind="block",
                             offset=o, offset_tx=o if same else -o, gid=x, rel_strand=int(strand[x]) * bstrand))
        # controls on the same contig
        n = hi - lo + 1
        c0 = int(g.gid[(cc == cc[lo])].min())
        c1 = int(g.gid[(cc == cc[lo])].max())
        starts = [s for s in range(c0, c1 - n + 2) if (s + n - 1 < lo - 6 or s > hi + 6)
                  and not (g.cur_annot.values[s:s + n].any() or g.in_sys.values[s:s + n].any())]
        for s in rng.permutation(starts)[:3] if len(starts) else []:
            s = int(s)
            for o, x in neigh(s, s + n - 1, True):
                rows.append(dict(type_key=r.type_key, instance=f"{r.genome}:{r.contig_code}:{r.pos_start}", kind=f"control_{s}",
                                 offset=o, offset_tx=o if strand[s] > 0 else -o, gid=x, rel_strand=int(strand[x]) * int(strand[s])))
    nb = pd.DataFrame(rows)
    nb = nb.merge(g[["gid", "protein_id", "cur_annot", "src_pfam", "src_ko", "src_df", "src_txss", "in_sys", "cctyper", "cdd_any",
                     "fam_e3"]], on="gid", how="left")
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET threads=1")
    con.register("p", nb[["protein_id"]].drop_duplicates())
    ann = con.execute("""SELECT a.protein_id, a.source, a.accession, a.name FROM annotations a JOIN p USING (protein_id)
                         WHERE a.source IN ('pfam','kofam','cdd','defensefinder_system','txsscan_system')""").df()
    sysm = con.execute("""SELECT s.protein_id, s.system_source, s.profile_name FROM system_proteins s JOIN p USING (protein_id)""").df()
    by = ann.groupby(["protein_id", "source"]).name.agg(lambda s: "|".join(sorted(set(map(str, s)))[:4])).unstack()
    for src in ("pfam", "kofam", "cdd", "defensefinder_system", "txsscan_system"):
        nb[src] = by[src].reindex(nb.protein_id).to_numpy() if src in by else None
    sm = sysm.groupby("protein_id").apply(lambda d: "|".join(sorted({f"{a}:{b}" for a, b in zip(d.system_source, d.profile_name)})[:3]))
    nb["system_member"] = sm.reindex(nb.protein_id).to_numpy() if len(sm) else None
    th = pd.read_parquet(D3 / "tier_hits.parquet")
    rule = th[th.source.isin(["hhsearch_pfam", "hhsearch_pdb", "hhblits_pdb"]) & (th.probability >= 90) & (th.aligned_cols >= 30)]
    ch = rule[~rule.unknown_function].sort_values("probability", ascending=False).drop_duplicates("family_id").set_index("family_id")
    uk = rule[rule.unknown_function].sort_values("probability", ascending=False).drop_duplicates("family_id").set_index("family_id")
    nb["tier_char"] = (ch.hit_name.reindex(nb.fam_e3)).to_numpy()
    nb["tier_unknown"] = (uk.hit_name.reindex(nb.fam_e3)).to_numpy()
    cm = pd.read_parquet(D3 / "cdd_models.parquet").set_index("accession")
    cdd = pd.read_parquet(D3 / "cdd_prot.parquet")
    cdd = cdd[cdd.protein_id.isin(nb.protein_id)]
    cdd["unknown"] = cm.unknown_function.reindex(cdd.accession).fillna(False).to_numpy()
    cdd_char = cdd[~cdd.unknown].groupby("protein_id").accession.agg(lambda s: "|".join(sorted(set(s))[:3]))
    nb["cdd_char"] = cdd_char.reindex(nb.protein_id).to_numpy()
    nb["recognised"] = nb.cur_annot.fillna(False) | nb.in_sys.fillna(False) | nb.cdd_char.notna() | nb.tier_char.notna()
    nb.to_csv(W3 / "context_neighbours.tsv", sep="\t", index=False)
    rows = []
    for (tk, kind0), d in nb.assign(kind0=np.where(nb.kind == "block", "block", "control")).groupby(["type_key", "kind0"]):
        ni = d.instance.nunique() if kind0 == "block" else d[["instance", "kind"]].drop_duplicates().shape[0]
        adj = d[d.offset.abs() == 1]
        same = adj[adj.rel_strand == 1]
        windows = d.groupby(["instance", "kind"])
        rows.append(dict(type_key=tk, kind=kind0, windows=ni,
                         adj_recognised_same_strand=int(same.groupby(["instance", "kind"]).recognised.any().sum()),
                         adj_curated_same_strand=int(same.groupby(["instance", "kind"]).cur_annot.any().sum()),
                         any_recognised_pm3=int(windows.recognised.any().sum()),
                         any_curated_pm3=int(windows.cur_annot.any().sum()),
                         any_system_member_pm3=int(windows.in_sys.any().sum())))
    sm = pd.DataFrame(rows)
    sm.to_csv(W3 / "context_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(sm.to_string())


if __name__ == "__main__":
    main()
