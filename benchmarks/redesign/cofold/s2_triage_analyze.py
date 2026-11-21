"""Screen 2, step 3b: classify every family from the triage searches (rules in
predeclared_criteria_screen2.json, 'triage_before_folding'), then every block.

Inputs (copied from biotite to work/s2/triage/res/): out/<token>/{uniref,pfam_enr,pfam_raw,pdb_enr,pdb_raw}.hhr,
pfam39_hmmsearch_{e01,ga}.domtbl, cdd_hits.tsv (+ the orchestrator's cdd_check hits for 20 chains),
db/cdd/family_superfamily_links.
Outputs: work/s2/triage_family.tsv (one row per family: best hit per source, class), work/s2/triage_hits.tsv
(every hit at probability >= 50 % / E <= 1, per source), work/s2/triage_blocks.tsv (pool blocks + Part A).
"""
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
W2 = W / "s2"
R = W2 / "triage" / "res"
UNKNOWN = re.compile(r"DUF\d|UPF\d|unknown function|uncharacteri|hypothetical|function unknown|unknown protein",
                     re.I)
COORD_MAP = {"rank13tri": ["164787", "351854", "221251"], "rank13": ["221251", "351854"],
             "rank27": ["100239", "199393"], "rank09": ["114055", "318532"], "rank01": ["81718", "393054"]}


def parse_hhr(path):
    """Hits from the alignment section: name/description, probability, E-value, aligned columns, query range."""
    if not path.exists():
        return None
    hits, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("No ") and line[3:].strip().isdigit():
            cur = dict(rank=int(line[3:]), q0=10 ** 9, q1=0)
            hits.append(cur)
        elif cur is not None and line.startswith(">") and "name" not in cur:
            cur["name"] = line[1:].strip()
        elif cur is not None and line.startswith("Probab="):
            kv = dict(x.split("=") for x in line.split())
            cur.update(prob=float(kv["Probab"]), evalue=float(kv["E-value"]), score=float(kv["Score"]),
                       cols=int(kv["Aligned_cols"]), ident=kv.get("Identities", ""))
        elif cur is not None and line.startswith("Q ") and len(line.split()) >= 5:
            t = line.split()
            if t[1] in ("ss_pred", "ss_conf", "ss_dssp", "Consensus"):
                continue
            try:
                a, b = int(t[2]), int(t[4])
            except ValueError:
                continue
            cur["q0"], cur["q1"] = min(cur["q0"], a), max(cur["q1"], b)
    return [h for h in hits if "prob" in h]


def main():
    fam = pd.read_csv(W2 / "families.tsv", sep="\t")
    toks = [str(t) for t in fam.token]
    hitrows = []
    ttl = {}
    if (W2 / "pdb_titles.tsv").exists():
        tt = pd.read_csv(W2 / "pdb_titles.tsv", sep="\t", header=None, names=["id", "title"], dtype=str).fillna("")
        ttl = dict(zip(tt.id, tt.title))
    # ---- HH-suite ------------------------------------------------------------------------------
    for t in toks:
        for src in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw", "uniref"):
            hh = parse_hhr(R / "out" / t / f"{src}.hhr")
            if hh is None:
                hitrows.append(dict(token=t, source=src, missing=True))
                continue
            for h in hh:
                if h["prob"] >= 50 or (src == "uniref" and h["rank"] <= 20):
                    nm = h["name"]
                    if src.startswith("pdb_"):
                        nm = f"{nm.split()[0]} {ttl.get(nm[:4].lower(), '')}".strip()
                    hitrows.append(dict(token=t, source=src, target=nm[:200], prob=h["prob"], evalue=h["evalue"],
                                        cols=h["cols"], q0=h["q0"], q1=h["q1"], n_instances=None))
    # ---- hmmsearch Pfam 39 -------------------------------------------------------------------------
    for tag in ("ga", "e01"):
        p = R / f"pfam39_hmmsearch_{tag}.domtbl"
        rows = []
        if not p.exists():
            print("missing", p.name, "- hmmsearch leg not yet available")
            continue
        for line in open(p):
            if line.startswith("#"):
                continue
            x = line.split()
            desc = " ".join(x[22:])
            rows.append(dict(token=x[0].split("|")[0], member=x[0].split("|")[1], pfam=x[3], acc=x[4], desc=desc,
                             evalue=float(x[6]), i_evalue=float(x[12]), score=float(x[7])))
        hm = pd.DataFrame(rows)
        if len(hm):
            agg = hm.groupby(["token", "acc", "pfam", "desc"]).agg(n_members=("member", "nunique"),
                                                                   best_e=("i_evalue", "min")).reset_index()
            for _, r in agg.iterrows():
                hitrows.append(dict(token=r.token, source=f"hmmsearch_{tag}", target=f"{r.acc} {r.pfam}; {r.desc}",
                                    evalue=r.best_e, n_instances=int(r.n_members)))
    # ---- CDD -----------------------------------------------------------------------------------
    links = pd.read_csv(W2 / "db" / "family_superfamily_links", sep="\t", header=None,
                        names=["acc", "pssm", "sf_acc", "sf_pssm"])
    sf = dict(zip(links.pssm.astype(str), links.sf_acc))
    cols = ["q", "s", "qlen", "qs", "qe", "evalue", "bits", "title"]
    cd = pd.read_csv(R / "cdd_hits.tsv", sep="\t", header=None, names=cols, quoting=3)
    cd["token"] = cd.q.str.split("|").str[0]
    cd["member"] = cd.q.str.split("|").str[1]
    co = pd.read_csv(W / "cdd_check_hits.tsv", sep="\t", header=None, names=cols, quoting=3)
    rr = []
    for _, r in co.iterrows():
        m = re.match(r"(rank\d+(?:tri)?)_r(\d)_([ABC])", r.q)
        t = COORD_MAP[m.group(1)]["ABC".index(m.group(3))]
        rr.append(dict(r, token=t, member=f"coord_{m.group(1)}_r{m.group(2)}"))
    cd = pd.concat([cd, pd.DataFrame(rr)], ignore_index=True)
    cd["pssm"] = cd.s.str.split("|").str[-1]
    cd["model"] = cd.title.str.split(",").str[0]
    cd["superfamily"] = cd.pssm.map(sf).fillna(cd.model)
    cd = cd.sort_values("evalue").drop_duplicates(["token", "member", "pssm"])
    n_inst = cd.groupby("token").member.nunique()
    for key in ("model", "superfamily"):
        a = cd[cd.evalue <= 0.01].groupby(["token", key]).agg(n=("member", "nunique"), best_e=("evalue", "min"),
                                                               title=("title", "first")).reset_index()
        for _, r in a.iterrows():
            hitrows.append(dict(token=r.token, source=f"cdd_{key}_e01", target=f"{r[key]} | {r.title[:150]}",
                                evalue=r.best_e, n_instances=int(r.n)))
        b = cd[(cd.evalue > 0.01) & (cd.evalue <= 1)].groupby(["token", key]).agg(
            n=("member", "nunique"), best_e=("evalue", "min"), title=("title", "first")).reset_index()
        for _, r in b[b.n >= 2].iterrows():
            hitrows.append(dict(token=r.token, source=f"cdd_{key}_weak", target=f"{r[key]} | {r.title[:150]}",
                                evalue=r.best_e, n_instances=int(r.n)))
    hits = pd.DataFrame(hitrows)
    for c in ("target", "prob", "evalue", "cols", "q0", "q1", "n_instances", "missing"):
        if c not in hits:
            hits[c] = None
    hits = hits[hits.missing.isna()].drop(columns=["missing"])
    hits["unknown_function"] = hits.target.fillna("").map(lambda s: bool(UNKNOWN.search(s)))
    hits.to_csv(W2 / "triage_hits.tsv", sep="\t", index=False)

    # ---- classify ----------------------------------------------------------------------------
    out = []
    for _, f in fam.iterrows():
        t = str(f.token)
        h = hits[(hits.token == t)]
        nm = int(f.in_fasta)
        strong, weak = [], []
        for _, x in h.iterrows():
            s = x.source
            if s in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"):
                if x.prob >= 90 and x.cols >= 30:
                    strong.append(x)
                elif x.prob >= 50:
                    weak.append(x)
            elif s == "hmmsearch_ga":
                need = 2 if nm <= 20 else max(2, int(round(0.10 * nm)))
                (strong if x.n_instances >= need else weak).append(x)
            elif s == "hmmsearch_e01":
                weak.append(x)
            elif s.startswith("cdd_") and s.endswith("_e01"):
                (strong if x.n_instances >= 2 else weak).append(x)
            elif s.startswith("cdd_") and s.endswith("_weak"):
                weak.append(x)
        chr_ = [x for x in strong if not x.unknown_function]
        unk = [x for x in strong if x.unknown_function]
        cls = "characterised" if chr_ else "unknown_function" if unk else "weak" if weak else "none"
        done = all((R / "out" / t / f"{x}.hhr").exists() for x in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"))
        if not done and cls != "characterised":
            cls = "pending"

        def best(src_prefix, by="prob"):
            x = h[h.source.str.startswith(src_prefix)]
            if not len(x):
                return ""
            x = x.sort_values(by, ascending=(by == "evalue")).iloc[0]
            if by == "prob":
                return f"{x.target[:90]} | P={x.prob:.1f} E={x.evalue:.2g} cols={int(x.cols)} q{int(x.q0)}-{int(x.q1)}"
            return f"{x.target[:90]} | E={x.evalue:.2g} n={x.n_instances}"
        u = h[h.source == "uniref"].sort_values("prob", ascending=False)
        u_named = u[~u.target.fillna("").str.contains(r"[Uu]ncharacteri|hypothetical|unknown", regex=True)]
        out.append(dict(token=t, sources=f.sources, query_gid=f.query_gid, query_len=f.query_len, members=f.members,
                        genomes=f.genomes, in_fasta=nm, cls=cls, n_char_hits=len(chr_), n_unknown_fn_hits=len(unk),
                        n_weak=len(weak),
                        reasons=" || ".join(sorted({f"{x.source}:{str(x.target)[:70]}" for x in chr_ + unk}))[:600],
                        weak_reasons=" || ".join(sorted({f"{x.source}:{str(x.target)[:60]}" for x in weak}))[:600],
                        best_pfam=best("pfam_"), best_pdb=best("pdb_"),
                        best_cdd=best("cdd_model_e01", "evalue") or best("cdd_model_weak", "evalue"),
                        best_hmm_ga=best("hmmsearch_ga", "evalue"), best_hmm_e01=best("hmmsearch_e01", "evalue"),
                        uniref_hits_p95=int((u.prob >= 95).sum()), uniref_top=(u.target.iloc[0][:120] if len(u) else ""),
                        uniref_top_prob=(float(u.prob.iloc[0]) if len(u) else None),
                        uniref_named_top=(f"{u_named.target.iloc[0][:120]} P={u_named.prob.iloc[0]:.1f}" if len(u_named) else ""),
                        cdd_instances=int(n_inst.get(t, 0))))
    tf = pd.DataFrame(out)
    tf.to_csv(W2 / "triage_family.tsv", sep="\t", index=False)
    print(tf.cls.value_counts().to_dict())

    # ---- blocks ------------------------------------------------------------------------------
    pool = pd.read_csv(W2 / "pool.tsv", sep="\t")
    cls = dict(zip(tf.token, tf.cls))
    order = {"characterised": 4, "pending": 3, "unknown_function": 2, "weak": 1, "none": 0}
    br = []
    for _, p in pool.iterrows():
        ts = p.type_key.split("-")
        c = [cls[t] for t in ts]
        br.append(dict(set="s2", id=int(p.s2_rank), pool=p.pool, type_key=p.type_key, k=int(p.k),
                       n_genomes=int(p.n_genomes), n_ord=int(p.n_ord), n_phylum=int(p.n_phylum), chain_classes=";".join(c),
                       block_class=max(c, key=order.get), fold=max(c, key=order.get) not in ("characterised", "pending")))
    for rk, tk in {27: "100239-199393", 13: "164787-351854-221251", 67: "229034-423021", 38: "23083-68571",
                   106: "27009-27010", 71: "171614-322104", 6: "322564-326340", 14: "48264-404027", 9: "114055-318532",
                   1: "81718-393054"}.items():
        c = [cls[t] for t in tk.split("-")]
        br.append(dict(set="partA", id=rk, pool="screen1", type_key=tk, k=len(c), chain_classes=";".join(c),
                       block_class=max(c, key=order.get), fold=None))
    bt = pd.DataFrame(br)
    bt.to_csv(W2 / "triage_blocks.tsv", sep="\t", index=False)
    s2 = bt[bt.set == "s2"]
    print("screen-2 blocks by class:", s2.block_class.value_counts().to_dict(), "| to fold:", int(s2.fold.sum()))
    print(s2.groupby(["pool", "block_class"]).size().to_string())
    open(W2 / "cleared_ranks.txt", "w").write("\n".join(str(i) for i in s2[s2.fold].id) + "\n")


if __name__ == "__main__":
    main()
