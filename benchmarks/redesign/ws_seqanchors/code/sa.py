#!/usr/bin/env python
"""Sequence, embedding and hybrid anchor arms through one chainer, one null, one calibration.

Modes
  emb  --model M      regenerate context60 anchors under the three query-order permutations
                      (frozen extension kernel, permuted query neighbours, as permuted_panels.py),
                      replay the published run, then chain and score E-ctx60/{cos,rank} and
                      E-seed60/{cos,rank} for model M
  arms --arms A,B     chain and score named arms (sequence and hybrid arms need the search)

Every arm: observed chain plus three permuted chains (query gene order permuted by
ordered_metadata, seeds 17/18/19, before every order-dependent step), per-size calibration
of the mean anchor weight at 5% and 1% permuted/observed, accepted pairs, source207 F1 with
random placements, target-genome strata, covered query genes, wall time.

Arm names
  S-{all,top60,top60cov50}/{unit,bitrank}
  E-{ctx60,seed60}/{cos,rank}@MODEL
  H-{ctx60,seed60}@MODEL              S-top60 (bit-score percentile) union embedding (cosine percentile)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
W = RD / "ws_seqanchors"
CODE = W / "code"
NREF = 4190383
SEEDS = (17, 18, 19)
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
SAVED_PERMUTED_TOTALS = {"esm2_8m": [495161, 495380, 495342], "esmc6b": [535642, 532999, 534641]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_metadata(metadata, seed):
    """Identical to permuted_panels.ordered_metadata / run_spacedust_seed_extension (September)."""
    result = metadata.copy()
    for (sample, contig), frame in metadata.groupby(["sample_id", "contig_id"], observed=True, sort=True):
        rng = np.random.default_rng(int.from_bytes(
            hashlib.sha256(f"{seed}:{sample}:{contig}".encode()).digest()[:8], "big"))
        result.loc[frame.index, "position_index"] = rng.permutation(len(frame))[
            frame.position_index.to_numpy(dtype=np.int64)]
    return result


def cut_for(obs_m, perm_m, n_perm, target):
    """Identical to permuted_panels.cut_for."""
    if len(obs_m) == 0:
        return None
    obs_sorted, perm_sorted = np.sort(obs_m), np.sort(perm_m)
    for t in np.unique(np.quantile(obs_m, np.linspace(0, 1, 401))):
        o = len(obs_sorted) - np.searchsorted(obs_sorted, t, side="left")
        p = (len(perm_sorted) - np.searchsorted(perm_sorted, t, side="left")) / n_perm
        if o > 0 and p / o <= target:
            return float(t)
    return None


class Ctx:
    def __init__(self, workers):
        self.workers = workers
        os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
        import numba
        numba.set_num_threads(workers)
        sys.path.insert(0, str(CODE / "scale_code"))
        import run_spacedust_scaling as rss
        from evaluate_spacedust_queries import evaluate
        self.rss, self.evaluate = rss, evaluate
        self.chain, self.group_type = rss.load_published(CODE / "production")
        t = time.time()
        self.qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
        self.rmeta = rss.complete_metadata(
            pd.read_parquet(ROOT / "results/esm2_t6/shard_000/export/metadata.parquet"), rss.REFERENCE_ROWS)
        assert not (set(self.qmeta.sample_id) & set(self.rmeta.sample_id))
        self.regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
        g = pd.read_parquet(RD / "universe/genomes.parquet")
        self.stratum = dict(zip(g.genome, g.stratum))
        self.phylum = dict(zip(g.genome, g.phylum))
        self.perm_meta = {}
        self.cache = {}
        log("context loaded", f"{time.time() - t:.0f}s")

    def meta(self, tag):
        if tag == "observed":
            return self.qmeta
        seed = int(tag.replace("seed", ""))
        if seed not in self.perm_meta:
            self.perm_meta[seed] = ordered_metadata(self.qmeta, seed)
        return self.perm_meta[seed]

    # ---- anchor sources ------------------------------------------------------------------
    def hits(self):
        if "hits" not in self.cache:
            h = pd.read_parquet(W / "work/hits.parquet", columns=["qrow", "trow", "bits", "rank_all", "rank_cov50"])
            self.cache["hits"] = h
            self.cache["bits_sorted"] = np.load(W / "work/bits_top60_sorted.npy")
        return self.cache["hits"]

    def seq(self, filt):
        key = ("seq", filt)
        if key not in self.cache:
            h = self.hits()
            mask = {"all": np.ones(len(h), bool), "top60": (h.rank_all <= 60).to_numpy(),
                    "top60cov50": (h.rank_cov50 <= 60).fillna(False).to_numpy(bool)}[filt]
            d = h.loc[mask, ["qrow", "trow", "bits"]]
            self.cache[key] = pd.DataFrame(dict(query_row_index=d.qrow.to_numpy(np.int64),
                                                target_row_index=d.trow.to_numpy(np.int64),
                                                bits=d.bits.to_numpy(np.float64)))
        return self.cache[key]

    def bitrank(self, bits):
        s = self.cache["bits_sorted"]
        return np.searchsorted(s, bits, side="right") / len(s)

    def base(self, model):
        return SCALE / "results" / model / "analysis/full_workflow"

    def seeds(self, model):
        key = ("seeds", model)
        if key not in self.cache:
            s = pd.read_parquet(self.base(model) / "seeds.parquet", columns=["query_row_index", "target_row_index", "similarity"])
            self.cache[key] = s
            self.cache[("cos_sorted", model)] = np.sort(s.similarity.to_numpy(np.float64))
        return self.cache[key]

    def cosrank(self, model, cos):
        self.seeds(model)
        s = self.cache[("cos_sorted", model)]
        return np.searchsorted(s, np.asarray(cos, np.float64), side="right") / len(s)

    def ctx(self, model, tag):
        if tag == "observed":
            p = self.base(model) / "anchors.parquet"
        else:
            p = W / "work/ctx_perm" / f"{model}_{tag}.parquet"
        return pd.read_parquet(p, columns=["query_row_index", "target_row_index", "similarity"])

    def exact(self, space, model):
        """Exact top-60 hits in a given space (zsearch.py) and the percentile weight of their similarity."""
        key = ("exact", space, model)
        if key not in self.cache:
            d = pd.read_parquet(W / "work/zseeds" / f"{space}__{model}.parquet",
                                columns=["query_row_index", "target_row_index", "similarity"])
            s = np.sort(d.similarity.to_numpy(np.float64))
            w = np.searchsorted(s, d.similarity.to_numpy(np.float64), side="right") / len(s)
            self.cache[key] = pd.DataFrame(dict(query_row_index=d.query_row_index.to_numpy(np.int64),
                                                target_row_index=d.target_row_index.to_numpy(np.int64),
                                                similarity=w))
        return self.cache[key]

    def anchors(self, arm, tag):
        """Return numeric anchors with the arm weight in `similarity`."""
        kind, rest = arm.split("-", 1)
        space = {"Z": "zca10", "L": "zlin256", "R": "raw", "HZ": "zca10", "HL": "zlin256"}.get(kind)
        if kind in ("Z", "L", "R"):
            model = rest.split("@")[1]
            return self.exact(space, model).copy()
        if kind in ("HZ", "HL"):
            model = rest.split("@")[1]
            s = self.seq("top60")
            both = pd.concat([
                pd.DataFrame(dict(query_row_index=s.query_row_index.to_numpy(), target_row_index=s.target_row_index.to_numpy(),
                                  similarity=self.bitrank(s.bits.to_numpy()))),
                self.exact(space, model)], ignore_index=True)
            both = both.sort_values(["query_row_index", "target_row_index", "similarity"],
                                    ascending=[True, True, False], kind="stable")
            return both.drop_duplicates(["query_row_index", "target_row_index"]).reset_index(drop=True)
        if kind == "S":
            filt, weight = rest.split("/")
            d = self.seq(filt)
            w = np.ones(len(d)) if weight == "unit" else self.bitrank(d.bits.to_numpy())
            return pd.DataFrame(dict(query_row_index=d.query_row_index.to_numpy(),
                                     target_row_index=d.target_row_index.to_numpy(), similarity=w))
        if kind == "E":
            spec, model = rest.split("@")
            source, weight = spec.split("/")
            d = self.ctx(model, tag) if source == "ctx60" else self.seeds(model)
            if weight == "cos":  # keep the published dtype for an exact replay
                return d[["query_row_index", "target_row_index", "similarity"]].reset_index(drop=True)
            return pd.DataFrame(dict(query_row_index=d.query_row_index.to_numpy(np.int64),
                                     target_row_index=d.target_row_index.to_numpy(np.int64),
                                     similarity=self.cosrank(model, d.similarity.to_numpy())))
        if kind == "H":
            source, model = rest.split("@")
            s = self.seq("top60")
            e = self.ctx(model, tag) if source == "ctx60" else self.seeds(model)
            both = pd.concat([
                pd.DataFrame(dict(query_row_index=s.query_row_index.to_numpy(), target_row_index=s.target_row_index.to_numpy(),
                                  similarity=self.bitrank(s.bits.to_numpy()))),
                pd.DataFrame(dict(query_row_index=e.query_row_index.to_numpy(np.int64),
                                  target_row_index=e.target_row_index.to_numpy(np.int64),
                                  similarity=self.cosrank(model, e.similarity.to_numpy())))], ignore_index=True)
            both = both.sort_values(["query_row_index", "target_row_index", "similarity"],
                                    ascending=[True, True, False], kind="stable")
            return both.drop_duplicates(["query_row_index", "target_row_index"]).reset_index(drop=True)
        raise ValueError(arm)

    # ---- chaining --------------------------------------------------------------------------
    def chain_one(self, numeric, tag):
        t = time.time()
        blocks = self.rss.chain_numeric(numeric, self.meta(tag), self.rmeta, self.chain, self.group_type)
        return blocks, time.time() - t


def arm_dir(arm):
    return W / "work/arms" / arm.replace("/", "_").replace("@", "__")


def covered_rows(ctx, calls):
    """Boolean over query rows: inside the query span of any given block."""
    q = ctx.qmeta
    cov = np.zeros(len(q), bool)
    if len(calls) == 0:
        return cov
    for (g, c), b in calls.groupby(["query_genome", "query_contig"], sort=False):
        sub = q[(q.sample_id == g) & (q.contig_id == c)]
        n = len(sub)
        diff = np.zeros(n + 1, np.int64)
        np.add.at(diff, b.query_start.to_numpy(np.int64), 1)
        np.add.at(diff, b.query_end.to_numpy(np.int64) + 1, -1)
        inside = np.cumsum(diff[:-1]) > 0
        rows = sub.row_index.to_numpy()[np.argsort(sub.position_index.to_numpy())]
        cov[rows[inside]] = True
    return cov


def block_pairs(ctx, blocks_path, numeric):
    """Anchor pairs of every observed block, as row indices (trap 2: reverse target lists for orientation -1)."""
    import duckdb
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={ctx.workers}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    qg = ctx.qmeta[["row_index", "gene_id"]]
    rg = ctx.rmeta[["row_index", "gene_id"]]
    con.register("qg", qg)
    con.register("rg", rg)
    pairs = con.execute(f"""
        WITH b AS (SELECT block_id, from_json(anchor_query_gene_ids, '["VARCHAR"]') AS q,
                          CASE WHEN orientation = -1
                               THEN list_reverse(from_json(anchor_target_gene_ids, '["VARCHAR"]'))
                               ELSE from_json(anchor_target_gene_ids, '["VARCHAR"]') END AS t
                   FROM read_parquet('{blocks_path}')),
        e AS (SELECT block_id, unnest(q) AS qgid, unnest(t) AS tgid FROM b)
        SELECT e.block_id, qg.row_index AS qrow, rg.row_index AS trow
        FROM e JOIN qg ON qg.gene_id = e.qgid JOIN rg ON rg.gene_id = e.tgid""").df()
    con.register("pairs", pairs)
    num = numeric[["query_row_index", "target_row_index"]]
    con.register("num", num)
    missing = con.execute("""SELECT count(*) FROM pairs p ANTI JOIN num n
                             ON n.query_row_index = p.qrow AND n.target_row_index = p.trow""").fetchone()[0]
    assert missing == 0, f"{missing} block pairs absent from the arm's anchors"
    return pairs


def score207(ctx, calls, mask, label):
    b = calls[mask]
    spans = b[SPAN].drop_duplicates()
    observed, ctrl, report = ctx.evaluate(ctx.regions, ctx.qmeta, spans, threads=ctx.workers, controls=200, seed=17)
    per = (observed.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).to_numpy()
    rng = np.random.default_rng(17)
    boot = rng.choice(per, size=(10000, len(per)), replace=True).mean(axis=1)
    return dict(selection=label, n_blocks=int(mask.sum()), n_query_intervals=len(spans),
                f1_observed=report["observed"]["mean_best_single_block_f1"],
                f1_random=report["controls"]["mean_best_single_block_f1"],
                f1_excess=float(per.mean()),
                f1_excess_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                localized_observed=report["observed"]["localized_fraction"],
                localized_random=report["controls"]["localized_fraction"],
                per_region_f1=observed.set_index("region_id").f1.round(6).to_dict())


def run_arm(ctx, arm):
    t_arm = time.time()
    out = arm_dir(arm)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(arm=arm, workers=ctx.workers, chain_seconds={}, anchors_in={})
    perms = []
    obs_numeric = None
    for tag in ("observed",) + tuple(f"seed{s}" for s in SEEDS):
        t = time.time()
        numeric = ctx.anchors(arm, tag)
        build = time.time() - t
        blocks, secs = ctx.chain_one(numeric, tag)
        res["chain_seconds"][tag] = round(secs, 1)
        res["anchors_in"][tag] = int(len(numeric))
        log(arm, tag, "anchors", len(numeric), "blocks", len(blocks), f"build {build:.0f}s chain {secs:.0f}s")
        if tag == "observed":
            blocks.to_parquet(out / "observed_blocks.parquet", index=False)
            obs_blocks, obs_numeric = blocks, numeric
        else:
            perms.append(blocks[["n_anchors", "chain_score"]].assign(seed=int(tag[4:])))
        del blocks
        if tag != "observed":
            del numeric
    perm = pd.concat(perms, ignore_index=True)
    perm.to_parquet(out / "permuted_blocks.parquet", index=False)
    res["observed_blocks"] = int(len(obs_blocks))
    res["observed_anchor_pairs"] = int(obs_blocks.n_anchors.sum())
    res["permuted_totals"] = [int((perm.seed == s).sum()) for s in SEEDS]
    res["permuted_over_observed"] = float(np.mean(res["permuted_totals"]) / max(len(obs_blocks), 1))

    # ---- calibration -----------------------------------------------------------------------
    obs = obs_blocks.drop(columns=["anchor_query_ids", "anchor_target_ids", "anchor_query_gene_ids",
                                   "anchor_target_gene_ids"]).copy()
    obs["m"] = obs.chain_score / obs.n_anchors
    perm["m"] = perm.chain_score / perm.n_anchors
    accept = {0.05: np.zeros(len(obs), bool), 0.01: np.zeros(len(obs), bool)}
    res["sizes"] = {}
    for size in (2, 3, 4, 5, 6):
        so = obs[obs.n_anchors == size] if size < 6 else obs[obs.n_anchors >= 6]
        sp = perm[perm.n_anchors == size] if size < 6 else perm[perm.n_anchors >= 6]
        om, pm = so.m.to_numpy(), sp.m.to_numpy()
        entry = dict(observed=len(so), permuted_mean=len(sp) / len(SEEDS),
                     ratio=(len(sp) / len(SEEDS)) / max(len(so), 1))
        for target in (0.05, 0.01):
            t = cut_for(om, pm, len(SEEDS), target)
            key = f"fdr{int(target * 100):02d}"
            if t is None:
                entry[key] = dict(threshold=None, accepted=0, permuted_above=0.0)
                continue
            acc = so[so.m >= t]
            accept[target][acc.index.to_numpy()] = True
            entry[key] = dict(threshold=t, accepted=len(acc), permuted_above=float((pm >= t).sum()) / len(SEEDS))
        res["sizes"][str(size) if size < 6 else "6+"] = entry
    obs["accept05"], obs["accept01"] = accept[0.05], accept[0.01]
    for k, msk in (("05", accept[0.05]), ("01", accept[0.01])):
        res[f"accepted{k}_blocks"] = int(msk.sum())
        res[f"accepted{k}_anchor_pairs"] = int(obs.n_anchors[msk].sum())
        res[f"accepted{k}_permuted_expected"] = float(sum(res["sizes"][s][f"fdr{k}"]["permuted_above"] for s in res["sizes"]))
    obs["stratum"] = obs.target_genome.map(ctx.stratum)
    obs["target_phylum"] = obs.target_genome.map(ctx.phylum)
    assert obs.stratum.notna().all()
    obs.to_parquet(out / "block_calls.parquet", index=False)

    # ---- pairs -----------------------------------------------------------------------------
    pairs = block_pairs(ctx, out / "observed_blocks.parquet", obs_numeric)
    assert len(pairs) == res["observed_anchor_pairs"], (len(pairs), res["observed_anchor_pairs"])
    pairs = pairs.merge(obs[["block_id", "n_anchors", "m", "accept05", "accept01", "stratum"]], on="block_id")
    w = obs_numeric.rename(columns={"query_row_index": "qrow", "target_row_index": "trow", "similarity": "weight"})
    pairs = pairs.merge(w, on=["qrow", "trow"], how="left")
    hits_path = W / "work/hits.parquet"
    if hits_path.exists():
        h = ctx.hits()[["qrow", "trow", "rank_all"]]
        pairs = pairs.merge(h, on=["qrow", "trow"], how="left")
        pairs["in_search"] = pairs.rank_all.notna()
        for k in ("05", "01"):
            a = pairs[pairs[f"accept{k}"]]
            r = pairs[~pairs[f"accept{k}"]]
            res[f"accepted{k}_pair_in_search"] = float(a.in_search.mean()) if len(a) else None
            res[f"rejected{k}_pair_in_search"] = float(r.in_search.mean()) if len(r) else None
        res["all_pair_in_search"] = float(pairs.in_search.mean())
    pairs.to_parquet(out / "pairs.parquet", index=False)
    res["accepted05_distinct_pairs"] = int(pairs[pairs.accept05][["qrow", "trow"]].drop_duplicates().shape[0])

    # ---- coverage and strata -----------------------------------------------------------------
    for k in ("05", "01"):
        acc = obs[obs[f"accept{k}"]]
        cov = covered_rows(ctx, acc)
        np.save(out / f"covered_query_rows_{k}.npy", cov)
        res[f"accepted{k}_covered_query_genes"] = int(cov.sum())
        res[f"accepted{k}_anchor_query_genes"] = int(pairs[pairs[f"accept{k}"]].qrow.nunique())
    cov_all = covered_rows(ctx, obs)
    res["all_covered_query_genes"] = int(cov_all.sum())
    acc = obs[obs.accept05]
    res["accepted05_by_stratum"] = {s: dict(blocks=int(len(g)), anchor_pairs=int(g.n_anchors.sum()),
                                            blocks_ge10=int((g.n_anchors >= 10).sum()))
                                    for s, g in acc.groupby("stratum")}
    res["all_by_stratum"] = {s: int(len(g)) for s, g in obs.groupby("stratum")}
    cp = acc[acc.stratum == "3_other_phylum"]
    res["accepted05_cross_phylum_blocks"] = int(len(cp))
    res["accepted05_cross_phylum_blocks_ge10"] = int((cp.n_anchors >= 10).sum())
    res["accepted05_cross_phylum_pairs"] = int(cp.n_anchors.sum())

    # ---- source207 ---------------------------------------------------------------------------
    t = time.time()
    res["source207"] = [score207(ctx, obs, np.ones(len(obs), bool), "all"),
                        score207(ctx, obs, obs.accept05.to_numpy(), "calibrated_5pct"),
                        score207(ctx, obs, obs.accept01.to_numpy(), "calibrated_1pct")]
    res["source207_seconds"] = round(time.time() - t, 1)
    res["arm_seconds"] = round(time.time() - t_arm, 1)
    (W / "results/arms").mkdir(parents=True, exist_ok=True)
    with open(W / "results/arms" / (out.name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    s5 = res["source207"][1]
    log("DONE", arm, "obs", res["observed_blocks"], "perm/obs", f"{res['permuted_over_observed']:.3f}",
        "acc05", res["accepted05_blocks"], "pairs", res["accepted05_anchor_pairs"],
        "F1", f"{s5['f1_observed']:.4f}", "excess", f"{s5['f1_excess']:+.4f}", "xphy", res["accepted05_cross_phylum_blocks"])
    return res


def mode_emb(ctx, model):
    """Regenerate context60 under permutation (and replay the observed run), then the embedding arms."""
    from spacedust_prepared_extension_executor import ARRAY_NAMES, extend_prepared_queries
    base = ctx.base(model)
    proto = json.load(open(base / "protocol.json"))
    paths = {name: base / "arrays" / f"{name}.npy" for name in ARRAY_NAMES if not name.endswith("_vectors")}
    paths["query_vectors"] = Path(proto["vectors"]["query"]["path"])
    paths["reference_vectors"] = Path(proto["vectors"]["reference"]["path"])
    for p in paths.values():
        assert p.exists(), p
    rows = np.arange(len(ctx.qmeta), dtype=np.int64)
    cache = W / "tmp" / f"nb_{model}"
    cache.mkdir(parents=True, exist_ok=True)
    pdir = W / "work/ctx_perm"
    pdir.mkdir(parents=True, exist_ok=True)
    rep = dict(model=model, extension_seconds={})
    # observed replay of the extension: must equal the saved anchors exactly in identity
    t = time.time()
    ext = extend_prepared_queries(paths, rows, workers=ctx.workers)
    rep["extension_seconds"]["observed"] = round(time.time() - t, 1)
    got = ext.pop("anchors")[["query_row_index", "target_row_index", "similarity"]]
    saved = ctx.ctx(model, "observed")
    a = got.sort_values(["query_row_index", "target_row_index"]).reset_index(drop=True)
    b = saved.sort_values(["query_row_index", "target_row_index"]).reset_index(drop=True)
    assert len(a) == len(b) and (a.query_row_index.values == b.query_row_index.values).all() \
        and (a.target_row_index.values == b.target_row_index.values).all(), "context60 replay differs"
    rep["observed_replay"] = dict(anchors=len(a), identical_pairs=True,
                                  max_similarity_difference=float(np.abs(a.similarity.values - b.similarity.values).max()))
    log(model, "context60 replay identical", len(a))
    for seed in SEEDS:
        tag = f"seed{seed}"
        out = pdir / f"{model}_{tag}.parquet"
        nb = ctx.rss.neighbors(ctx.meta(tag))
        np.save(cache / f"query_neighbors_{tag}.npy", nb)
        p = dict(paths)
        p["query_neighbors"] = cache / f"query_neighbors_{tag}.npy"
        t = time.time()
        ext = extend_prepared_queries(p, rows, workers=ctx.workers)
        rep["extension_seconds"][tag] = round(time.time() - t, 1)
        anchors = ext.pop("anchors")[["query_row_index", "target_row_index", "similarity"]]
        tmp = out.with_suffix(".tmp.parquet")
        anchors.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        log(model, tag, "context60 anchors", len(anchors))
    (W / "results/ctx_perm").mkdir(parents=True, exist_ok=True)
    arms = [f"E-ctx60/cos@{model}", f"E-ctx60/rank@{model}", f"E-seed60/cos@{model}", f"E-seed60/rank@{model}"]
    for arm in arms:
        r = run_arm(ctx, arm)
        if arm.startswith("E-ctx60/cos"):
            saved_blocks = pd.read_parquet(base / "all_blocks.parquet", columns=["n_anchors"])
            rep["blocks_replay"] = dict(blocks=r["observed_blocks"], saved_blocks=len(saved_blocks),
                                        anchors=r["observed_anchor_pairs"], saved_anchors=int(saved_blocks.n_anchors.sum()))
            panel = json.load(open(ROOT / "seq_invisible_20261111/results/permuted_panels" / f"{model}.json"))
            rep["permuted_totals"] = r["permuted_totals"]
            rep["panel_permuted_totals"] = panel["permuted_totals"]
            rep["permuted_totals_match_panel"] = r["permuted_totals"] == panel["permuted_totals"]
            if model in SAVED_PERMUTED_TOTALS:
                rep["permuted_totals_match_saved"] = r["permuted_totals"] == SAVED_PERMUTED_TOTALS[model]
            log(model, "replay", rep["blocks_replay"], "perm", r["permuted_totals"], "panel", panel["permuted_totals"])
            with open(W / "results/ctx_perm" / f"{model}.json", "w") as fh:
                json.dump(rep, fh, indent=1)
            assert rep["blocks_replay"]["blocks"] == rep["blocks_replay"]["saved_blocks"]
            assert rep["blocks_replay"]["anchors"] == rep["blocks_replay"]["saved_anchors"]
            if not rep["permuted_totals_match_panel"]:
                log("WARNING", model, "permuted totals differ from tonight's panels", r["permuted_totals"],
                    panel["permuted_totals"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["emb", "arms"])
    ap.add_argument("--model")
    ap.add_argument("--arms")
    ap.add_argument("--workers", type=int, required=True)
    a = ap.parse_args()
    ctx = Ctx(a.workers)
    if a.mode == "emb":
        mode_emb(ctx, a.model)
    else:
        for arm in a.arms.split(","):
            run_arm(ctx, arm)


if __name__ == "__main__":
    main()
