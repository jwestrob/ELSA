#!/usr/bin/env python
"""Test one Sharur embedding store for the padded-batch all-position-mean defect.

Pre-July 2026 Sharur (commits fc03c29 and 4592775) pooled `last_hidden_state.mean(dim=1)` over a
padded batch (BOS, residues, EOS and padding), batch size 16 on MPS, 32 on CUDA, 4 on CPU, rows
written in processing order. From 8f89a02 (17 Jul 2026) the mean is residue-masked.

For random 32-row windows (aligned to 32, hence to 4, 8, 16 and 32) the stored vectors are compared
(cosine) with:
  padded_all_b{4,8,16,32}  re-run as padded batches of b, all-position mean (old writers)
  padded_residue_b32       padded batch of 32, residue-masked mean (new writer)
  single_all_token_mean    one protein per pass, mean over BOS + residues + EOS
  single_residue_mean      one protein per pass, residue mean
Sequences: the store's stage03 FASTA (the embedder's input; trailing '*' stripped as both old
writers did), located through the manifest's genome order; DuckDB (READ_ONLY) fallback.
"""
import argparse
import json
import os
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import torch

MODEL = "facebook/esm2_t6_8M_UR50D"


def read_faa(path):
    out, cur, parts = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur and parts:
                    out[cur] = "".join(parts).rstrip("*")
                cur, parts = line[1:].split()[0], []
            else:
                parts.append(line)
    if cur and parts:
        out[cur] = "".join(parts).rstrip("*")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage03", nargs="*", default=[], help="stage03 dirs holding genomes/<genome>/*.faa")
    ap.add_argument("--db", help="DuckDB with proteins(protein_id, sequence), opened READ_ONLY")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, add_pooling_layer=False).eval()
    man = json.loads(Path(a.manifest).read_text())
    counts = man["statistics"]["genome_protein_counts"]
    with h5py.File(a.h5, "r") as f:
        ids = np.array([x.decode() for x in f["protein_ids"][:]])
        n = len(ids)
        rng = np.random.default_rng(17)
        starts = np.sort(rng.choice(n // 32 - 1, size=a.windows, replace=False) * 32)
        rows = [int(r) for s in starts for r in range(s, s + 32)]
        V = {r: f["embeddings"][r] for r in rows}
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    info = dict(store=a.name, h5=a.h5, rows=int(n), unique_ids=int(len(set(ids))), h5_attrs=attrs,
                manifest_created=man.get("created_at"), manifest_total=man.get("total_proteins"),
                manifest_genomes=len(counts), manifest_count_sum=int(sum(counts.values())))
    # genome of each row from the manifest's insertion order (= processing order) when it sums to n
    seq, src = {}, {}
    if sum(counts.values()) == n and a.stage03:
        bounds = np.cumsum([0] + list(counts.values()))
        names = list(counts.keys())
        need = {}
        for r in rows:
            g = names[np.searchsorted(bounds, r, side="right") - 1]
            need.setdefault(g, []).append(r)
        for g, rr in need.items():
            fasta = {}
            for d in a.stage03:
                gd = Path(d) / "genomes" / g
                if gd.is_dir():
                    for p in sorted(gd.glob("*.faa")):
                        fasta.update(read_faa(p))
                    break
            for r in rr:
                if ids[r] in fasta:
                    seq[r], src[r] = fasta[ids[r]], "stage03_fasta"
    missing = [r for r in rows if r not in seq]
    if missing and a.db:
        con = duckdb.connect(config={"threads": a.threads})
        con.execute(f"ATTACH '{a.db}' AS d (READ_ONLY)")
        con.register("q", pd.DataFrame({"protein_id": ids[missing], "row": missing}))
        for r, s in con.execute("SELECT q.row, p.sequence FROM q JOIN d.proteins p USING (protein_id)").fetchall():
            if s is not None:
                seq[int(r)], src[int(r)] = s.rstrip("*"), "duckdb"
    info["sequence_source_counts"] = pd.Series(src).value_counts().to_dict()
    ok_windows = [s for s in starts if all((s + i) in seq for i in range(32))]
    info["windows_sampled"] = [int(s) for s in starts]
    info["windows_complete"] = [int(s) for s in ok_windows]
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    res = []
    with torch.inference_mode():
        def batch(rr, residue):
            enc = tok([seq[r] for r in rr], return_tensors="pt", padding=True, truncation=True, max_length=1024,
                      return_special_tokens_mask=True)
            sm = enc.pop("special_tokens_mask").bool()
            h = model(**enc).last_hidden_state
            if not residue:
                return h.mean(dim=1).numpy()
            w = (enc["attention_mask"].bool() & ~sm).unsqueeze(-1).to(h.dtype)
            return ((h * w).sum(1) / w.sum(1)).numpy()
        for s in ok_windows:
            for bs in (4, 8, 16, 32):
                for b0 in range(s, s + 32, bs):
                    rr = list(range(b0, b0 + bs))
                    for r, e in zip(rr, batch(rr, False)):
                        res.append(dict(row=r, variant=f"padded_all_b{bs}", cos=cos(e, V[r]), length=len(seq[r])))
            rr = list(range(s, s + 32))
            for r, e in zip(rr, batch(rr, True)):
                res.append(dict(row=r, variant="padded_residue_b32", cos=cos(e, V[r]), length=len(seq[r])))
            for r in rr:
                enc = tok([seq[r]], return_tensors="pt", truncation=True, max_length=1024, return_special_tokens_mask=True)
                sm = enc.pop("special_tokens_mask")[0].bool()
                h = model(**enc).last_hidden_state[0]
                res.append(dict(row=r, variant="single_all_token_mean", cos=cos(h.mean(0).numpy(), V[r]), length=len(seq[r])))
                res.append(dict(row=r, variant="single_residue_mean", cos=cos(h[~sm].mean(0).numpy(), V[r]), length=len(seq[r])))
    df = pd.DataFrame(res, columns=["row", "variant", "cos", "length"])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"{a.name}_pooling.tsv", sep="\t", index=False)
    order = ["padded_all_b4", "padded_all_b8", "padded_all_b16", "padded_all_b32", "padded_residue_b32",
             "single_all_token_mean", "single_residue_mean"]
    summ = {}
    for v in order:
        g = df[df.variant == v]
        if len(g):
            summ[v] = dict(n=int(len(g)), cos_median=float(g.cos.median()), cos_min=float(g.cos.min()),
                           frac_ge_0999=float((g.cos >= 0.999).mean()))
    best_old = max((summ[v]["frac_ge_0999"], v) for v in order[:4] if v in summ) if summ else (0, None)
    clean = summ.get("single_residue_mean", {}).get("frac_ge_0999", 0)
    verdict = ("defect" if best_old[0] >= 0.90 and clean < 0.50 else "clean" if clean >= 0.95 else "unresolved")
    info.update(summary=summ, best_old_variant=best_old[1], best_old_frac=best_old[0], verdict=verdict)
    (a.out / f"{a.name}_pooling.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
