"""Part A triage table: per chain of the screen-1 dossier blocks and the four uncalled pairs, the profile
evidence (work/s2/triage_family.tsv, from s2_triage_analyze.py) and the structure evidence (Foldseek of the
predicted chains in both replicates, work/s2/partA_foldseek_chains.tsv), and a combined observation:
  'no detectable similarity'  profile class none or weak, and no Foldseek hit at tier moderate or strong in
                              either replicate
  'resembles <...>'           otherwise, naming the best evidence (observation, UNVERIFIED as any system call)
Structure is counted when the tier is strong in >= 1 replicate or moderate-or-better in both.
Writes partA_triage.tsv (deliverable directory)."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
W2 = HERE / "work" / "s2"
BLOCKS = {27: "100239-199393", 13: "221251-351854", 67: "229034-423021", 38: "23083-68571", 106: "27009-27010",
          71: "171614-322104", 6: "322564-326340", 14: "48264-404027", 9: "114055-318532", 1: "81718-393054"}
TRI = ["164787", "351854", "221251"]
UNC = {"unc_002_r1": ["305819", "303977"], "unc_069_r2": ["269054", "203874"], "unc_071_r2": ["125177", "322055"],
       "unc_087_r1": ["172558", "172557"]}
RANKOK = {"strong": 2, "moderate": 1, "none": 0}


HITS = pd.read_csv(W2 / "triage_hits.tsv", sep="\t")


def main():
    tf = pd.read_csv(W2 / "triage_family.tsv", sep="\t", dtype={"token": str}).set_index("token")
    fs = pd.read_csv(W2 / "partA_foldseek_chains.tsv", sep="\t")
    rows = []

    def add(label, block, chain, token, fold_ids):
        f = tf.loc[token]
        s = fs[fs["query"].isin([f"{x}_{chain}" for x in fold_ids])]
        best = None
        tiers = []
        for _, r in s.iterrows():
            for db in ("pdb", "alphafold_swissprot"):
                t = r[f"{db}_tier"]
                tiers.append((r["query"], db, t))
                cand = (RANKOK[t], -float(r[f"{db}_evalue"]), f"{db}:{str(r[f'{db}_theader'])[:80]} "
                        f"(E={float(r[f'{db}_evalue']):.1e}, TM={float(r[f'{db}_alntmscore']):.2f})")
                if best is None or cand > best:
                    best = cand
        per_rep = {}
        for q, db, t in tiers:
            per_rep[q] = max(per_rep.get(q, 0), RANKOK[t])
        vals = list(per_rep.values())
        struct = bool(vals) and (max(vals) == 2 or (len(vals) >= 2 and min(vals) >= 1))
        prof = f.cls
        if prof in ("none", "weak") and not struct:
            obs = "no detectable similarity"
        else:
            ev = []
            if prof in ("characterised", "unknown_function"):
                ev.append(f"profile[{prof}]: {str(f.reasons)[:160]}")
            if struct:
                ev.append(f"structure: {best[2]}")
            obs = "resembles: " + " | ".join(ev)
        hh = HITS[HITS.token == int(token)]
        def top(prefix):
            x = hh[hh.source.str.startswith(prefix)].sort_values("prob", ascending=False)
            if not len(x):
                return ""
            x = x.iloc[0]
            cov = (x.q1 - x.q0 + 1) / f.query_len
            return f"{str(x.target)[:70]} P={x.prob:.1f} E={x.evalue:.1g} qcov={cov:.2f} ({x.source.split('_')[1]})"
        rows.append(dict(set=label, block=block, chain=chain, token=token, query_len=int(f.query_len),
                         top_pfam=top("pfam_"), top_pdb=top("pdb_"),
                         members=int(f.members), genomes=int(f.genomes), profile_class=prof,
                         best_pfam_hh=f.best_pfam, best_pdb_hh=f.best_pdb, best_cdd=f.best_cdd,
                         best_pfam_ga=f.best_hmm_ga, uniref_top=f.uniref_named_top,
                         foldseek_tiers=";".join(f"{q.rsplit('_', 2)[-2]}:{db[:3]}={t}" for q, db, t in tiers),
                         foldseek_best=best[2] if best else "", observation=obs))

    for rk, tk in BLOCKS.items():
        for chain, tok in zip("AB", tk.split("-")):
            add("screen1", rk, chain, tok, [f"cog_{rk:03d}_r1", f"cog_{rk:03d}_r2"])
    for chain, tok in zip("ABC", TRI):
        add("screen1_triple", 13, chain, tok, ["tri_013_r1", "tri_013_r2"])
    for fid, tt in UNC.items():
        for chain, tok in zip("AB", tt):
            add("uncalled", fid, chain, tok, [fid])
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "partA_triage.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(out[["set", "block", "chain", "token", "query_len", "profile_class", "observation"]].to_string(index=False))


if __name__ == "__main__":
    main()
