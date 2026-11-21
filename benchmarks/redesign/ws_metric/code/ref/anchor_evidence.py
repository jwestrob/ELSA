#!/usr/bin/env python
"""Which pair-level evidence function best separates homologs from non-homologs?

Population A: a uniform hash sample of each model's original ANN60 seeds, labelled
              by pairwise sequence visibility (Smith-Waterman E <= 1e-3).
Population B: sequence-invisible pairs inside blocks that have a TM-score from the
              key-subset job, labelled by TM >= 0.5.

Scores compared on identical pairs:
  raw            cosine of L2-normalized mean-pooled vectors (what ELSA uses)
  global_center  cosine after subtracting the reference mean vector
  genome_center  cosine after subtracting each protein's own genome mean vector
  csls_q         raw minus the mean of the query's 10 best seed cosines
  z_q            raw standardized within the query's seed list
  rank_q         negative rank within the query's seed list
  synteny_head   cosine after a linear map trained with a contrastive loss on anchor
                 pairs from order-significant blocks (>= 6 anchors) of training query
                 genomes; evaluated on held-out query genomes only
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
MODELS = ["esm2_8m", "esm2_650m", "esmc6b"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def group_means(mat, codes, n_groups, chunk=100_000):
    """Mean vector per group; rows with any non-finite value are skipped."""
    d = mat.shape[1]
    sums = np.zeros((n_groups, d), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    for s in range(0, len(mat), chunk):
        x = np.asarray(mat[s:s + chunk], dtype=np.float32)
        c = codes[s:s + chunk]
        ok = np.isfinite(x).all(axis=1)
        x, c = x[ok], c[ok]
        if not len(c):
            continue
        order = np.argsort(c, kind="stable")
        x, c = x[order], c[order]
        starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
        sums[c[starts]] += np.add.reduceat(x, starts, axis=0)
        counts[c[starts]] += np.diff(np.r_[starts, len(c)])
    total = sums.sum(axis=0) / max(counts.sum(), 1)
    means = sums / np.maximum(counts, 1)[:, None]
    return means.astype(np.float32), total.astype(np.float32), counts


def gather(mat, rows):
    uniq, inv = np.unique(rows, return_inverse=True)
    return np.asarray(mat[uniq], dtype=np.float32), inv


def cos_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.einsum("ij,ij->i", a, b)


def metrics(label, scores):
    out = {}
    label = np.asarray(label, dtype=bool)
    if label.all() or (~label).all() or len(label) < 50:
        return out
    for name, s in scores.items():
        ok = np.isfinite(s)
        out[name] = dict(auroc=float(roc_auc_score(label[ok], s[ok])),
                         ap=float(average_precision_score(label[ok], s[ok])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    W, K, OUT = a.run / "work", a.run / "work_key", a.run / "results/anchor_evidence"
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(17)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{a.run}/tmp'")
    for t in ("qx", "tx"):
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{W}/{t}.parquet')")
    con.execute(f"""
        CREATE TABLE seq AS
        SELECT qkey, tkey, evalue FROM read_csv('{W}/aln_seq.tsv', delim='\t', header=false,
            names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'],
            types={{'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}})""")
    has_tm = (K / "tm_key.parquet").exists()
    if has_tm:
        con.execute(f"CREATE TABLE tm AS SELECT * FROM read_parquet('{K}/tm_key.parquet')")
        con.execute(f"CREATE TABLE BPK AS SELECT * FROM read_parquet('{K}/block_pairs.parquet')")
    prep = json.load(open(W / "prep_stats.json"))
    qmeta = pd.read_parquet(ROOT / "bgc_queries/metadata.parquet", columns=["row_index", "sample_id"])
    rmeta = pd.read_parquet(ROOT / "results/esm2_t6/shard_000/export/metadata.parquet",
                            columns=["row_index", "sample_id"])
    q_codes, q_names = pd.factorize(qmeta.sample_id, sort=True)
    r_codes, r_names = pd.factorize(rmeta.sample_id, sort=True)
    test_genomes = set(q_names[::3])           # every third query genome, held out from head training
    log("held-out query genomes:", sorted(test_genomes))
    result = {"held_out_query_genomes": sorted(test_genomes)}

    for model in MODELS:
        t0 = time.time()
        base = SCALE / "results" / model / "analysis/full_workflow"
        proto = json.load(open(base / "protocol.json"))
        qv = np.load(proto["vectors"]["query"]["path"], mmap_mode="r")
        rv = np.load(proto["vectors"]["reference"]["path"], mmap_mode="r")
        r_means, mu, _ = group_means(rv, r_codes, len(r_names))
        q_means, _, _ = group_means(qv, q_codes, len(q_names))
        log(model, "genome means done", f"{time.time()-t0:.0f}s", "dim", rv.shape[1])

        # ---- per-query seed statistics and populations ---------------------------
        thr = prep["models"][model]["hash_threshold_seed"]
        con.execute(f"""
            CREATE OR REPLACE TABLE S AS
            SELECT s.query_row_index AS qrow, s.target_row_index AS trow, s.similarity AS sim,
                   avg(s.similarity) OVER w AS mean_q, stddev_pop(s.similarity) OVER w AS sd_q,
                   row_number() OVER (PARTITION BY s.query_row_index ORDER BY s.similarity DESC) AS rank_q
            FROM read_parquet('{base}/seeds.parquet') s
            WINDOW w AS (PARTITION BY s.query_row_index)""")
        con.execute("""CREATE OR REPLACE TABLE QS AS
                       SELECT qrow, avg(sim) FILTER (WHERE rank_q <= 10) AS r_q, any_value(mean_q) AS mean_q,
                              any_value(sd_q) AS sd_q FROM S GROUP BY ALL""")
        popA = con.execute(f"""
            SELECT S.qrow, S.trow, S.sim, S.rank_q, QS.r_q, QS.mean_q, QS.sd_q,
                   COALESCE(seq.evalue <= 1e-3, FALSE) AS label, (seq.qkey IS NOT NULL) AS scored
            FROM S JOIN QS USING (qrow) JOIN qx ON qx.row_index = S.qrow JOIN tx ON tx.trow = S.trow
            LEFT JOIN seq ON seq.qkey = qx.qkey AND seq.tkey = tx.tkey
            WHERE hash(CAST(qx.qkey AS BIGINT) * 5000000 + CAST(tx.tkey AS BIGINT)) % 1000 < {thr}""").df()
        res = dict(popA_n=len(popA), popA_scored=int(popA.scored.sum()), popA_label_rate=float(popA.label.mean()))
        popA = popA[popA.scored].reset_index(drop=True)
        popB = None
        if has_tm:
            popB = con.execute(f"""
                SELECT DISTINCT qx.row_index AS qrow, tx.trow, tm.tm_max, QS.r_q, QS.mean_q, QS.sd_q,
                       B.accept05, B.n_anchors
                FROM BPK B JOIN tm USING (qkey, tkey) JOIN qx ON qx.qkey = B.qkey JOIN tx ON tx.tkey = B.tkey
                LEFT JOIN QS ON QS.qrow = qx.row_index
                WHERE B.model = '{model}' AND NOT B.vis""").df()
            popB["label"] = popB.tm_max >= 0.5
        # positives for the head: anchors of blocks with >= 6 anchors
        pos = con.execute(f"""
            WITH b AS (
                SELECT from_json(anchor_query_gene_ids, '["VARCHAR"]') AS q,
                       CASE WHEN orientation = -1
                            THEN list_reverse(from_json(anchor_target_gene_ids, '["VARCHAR"]'))
                            ELSE from_json(anchor_target_gene_ids, '["VARCHAR"]') END AS t
                FROM read_parquet('{base}/all_blocks.parquet') WHERE n_anchors >= 6),
            e AS (SELECT unnest(q) AS qg, unnest(t) AS tg FROM b)
            SELECT DISTINCT qx.row_index AS qrow, tx.trow
            FROM e JOIN qx ON qx.gene_id = e.qg
            JOIN tx ON tx.tkey = CAST(split_part(e.tg, ':', 2) AS BIGINT)""").df()

        def score_population(df, with_seed_stats=True):
            Q, qi = gather(qv, df.qrow.to_numpy())
            T, ti = gather(rv, df.trow.to_numpy())
            q, t = Q[qi], T[ti]
            qg = q_codes[df.qrow.to_numpy()]
            tg = r_codes[df.trow.to_numpy()]
            raw = np.einsum("ij,ij->i", q, t)
            sc = dict(raw=raw, global_center=cos_rows(q - mu, t - mu),
                      genome_center=cos_rows(q - q_means[qg], t - r_means[tg]))
            if with_seed_stats:
                sc["csls_q"] = raw - df.r_q.to_numpy()
                sc["z_q"] = (raw - df.mean_q.to_numpy()) / (df.sd_q.to_numpy() + 1e-6)
            return sc, q, t, qg, tg

        is_test_q = np.array([n in test_genomes for n in q_names])
        # ---- train the synteny head on training-genome positives -------------------
        pos_train = pos[~is_test_q[q_codes[pos.qrow.to_numpy()]]].reset_index(drop=True)
        Qp, qpi = gather(qv, pos_train.qrow.to_numpy())
        Tp, tpi = gather(rv, pos_train.trow.to_numpy())
        d, d_out = rv.shape[1], 256
        mu_t = torch.from_numpy(mu)
        Qp_t, Tp_t = torch.from_numpy(Qp) - mu_t, torch.from_numpy(Tp) - mu_t
        head = torch.nn.Linear(d, d_out, bias=False)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        qpi_t, tpi_t = torch.from_numpy(qpi), torch.from_numpy(tpi)
        n_pos, bs, temp = len(pos_train), 2048, 0.05
        steps = 0
        for epoch in range(4):
            perm = torch.randperm(n_pos)
            for s in range(0, n_pos - bs + 1, bs):
                idx = perm[s:s + bs]
                zq = torch.nn.functional.normalize(head(Qp_t[qpi_t[idx]]), dim=1)
                zt = torch.nn.functional.normalize(head(Tp_t[tpi_t[idx]]), dim=1)
                logits = zq @ zt.T / temp
                target = torch.arange(len(idx))
                loss = (torch.nn.functional.cross_entropy(logits, target) +
                        torch.nn.functional.cross_entropy(logits.T, target)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 1
            log(model, "head epoch", epoch, "loss", round(float(loss), 4), "positives", n_pos)
        Wm = head.weight.detach().numpy().T.astype(np.float32)
        res.update(head_train_pairs=int(n_pos), head_steps=steps, head_final_loss=float(loss))

        def head_score(q, t):
            return cos_rows((q - mu) @ Wm, (t - mu) @ Wm)

        # ---- population A -----------------------------------------------------------
        scA, q, t, qg, tg = score_population(popA)
        scA["rank_q"] = -popA.rank_q.to_numpy().astype(np.float64)
        scA["synteny_head"] = head_score(q, t)
        lab = popA.label.to_numpy()
        test = is_test_q[qg]
        res["popA_all"] = metrics(lab, {k: v for k, v in scA.items() if k != "synteny_head"})
        res["popA_heldout_genomes"] = metrics(lab[test], {k: v[test] for k, v in scA.items()})
        res["popA_heldout_n"] = int(test.sum())
        # precision of the top 20% of seeds by each score, held-out genomes
        top = {}
        for k, v in scA.items():
            vv, ll = v[test], lab[test]
            cut = np.quantile(vv, 0.8)
            top[k] = float(ll[vv >= cut].mean())
        res["popA_heldout_precision_top20pct"] = top
        del q, t
        # ---- population B -----------------------------------------------------------
        if popB is not None and len(popB) > 100:
            scB, q, t, qg, tg = score_population(popB)
            scB["synteny_head"] = head_score(q, t)
            lab = popB.label.to_numpy()
            test = is_test_q[qg]
            res["popB_n"] = len(popB)
            res["popB_label_rate"] = float(lab.mean())
            res["popB_all"] = metrics(lab, {k: v for k, v in scB.items() if k != "synteny_head"})
            res["popB_heldout_genomes"] = metrics(lab[test], {k: v[test] for k, v in scB.items()})
            res["popB_heldout_n"] = int(test.sum())
            del q, t
        res["seconds"] = round(time.time() - t0, 1)
        result[model] = res
        log(model, json.dumps({k: res[k] for k in res if k.startswith("popA_heldout") or k.startswith("popB_heldout")})[:1500])
        with open(OUT / "anchor_evidence.json", "w") as fh:
            json.dump(result, fh, indent=1)
    log("done")


if __name__ == "__main__":
    main()
