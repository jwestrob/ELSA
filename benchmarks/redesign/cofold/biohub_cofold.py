#!/usr/bin/env python3
"""Hetero-complex folding with ESMFold2 through the Biohub batch API (HTTP only, no local compute).

Adapted from Sharur scripts/fold_biohub.py (Jacob's client): same endpoint (`fold_all_atom`),
same auth (ESM_API_KEY / BIOHUB_API_KEY), trailing '*' stripped. Differences: two or more protein
entities with different sequences per item, `include_pae` + `include_pair_chains_iptm`, optional
per-chain MSAs, batch-level polling through /batch/list, resumable state, and interface metrics
computed on arrival (raw responses are not kept; PDB + inter-chain PAE blocks + metrics are).

Usage:
  python biohub_cofold.py submit  --jobs jobs.jsonl --out OUT_DIR [--limit N] [--model M]
  python biohub_cofold.py collect --out OUT_DIR            # poll until every item is terminal
  jobs.jsonl: one JSON per line {"fold_id": str, "chains": [{"id": "A", "seq": str, "msa": [str,...]?}, ...]}

Metrics per fold (metrics.jsonl in OUT_DIR):
  ptm, iptm (interface_ptm), pair_chains_iptm (C x C), per-chain mean pLDDT,
  contacts = inter-chain residue pairs with CB-CB <= 8 A (CA for Gly),
  interface residues = residues in such a pair, interface_plddt = mean pLDDT over them,
  iface_pae = mean of PAE(i,j) and PAE(j,i) over contact pairs (A), pae_inter_mean = mean over all
  inter-chain residue pairs (A), pae_inter_min = min over inter-chain pairs (A).
"""
import argparse
import gzip
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

BASE = "https://biohub.ai/api/v1/batch"
MODEL = "esmfold2-2026-05"
AA3 = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU", G="GLY", H="HIS", I="ILE", L="LEU",
           K="LYS", M="MET", F="PHE", P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")
TRANSIENT = (500, 502, 503, 504)


def headers():
    key = os.environ.get("BIOHUB_API_KEY") or os.environ.get("ESM_API_KEY")
    if not key:
        sys.exit("set BIOHUB_API_KEY or ESM_API_KEY")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "elsa-cofold/0.1"}


def post(path, body, H, attempts=10):
    for attempt in range(attempts):
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H, method="POST")
        wait = min(300, 15 * 2 ** attempt)
        try:
            return json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After", "30")))
                continue
            if e.code in TRANSIENT and attempt < attempts - 1:
                print(f"HTTP {e.code} on {path}; retry in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code} on {path}: {e.read().decode()[:500]}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt < attempts - 1:
                print(f"{type(e).__name__} on {path}; retry in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"{path}: gave up")


def fetch_json(url, attempts=6):
    for attempt in range(attempts):
        try:
            raw = urllib.request.urlopen(url, timeout=300).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001 - network errors of every kind are retried
            if attempt == attempts - 1:
                raise
            time.sleep(10 * 2 ** attempt)


def clean(seq):
    s = seq.strip().rstrip("*").upper()
    if "*" in s or re.search(r"[^ACDEFGHIKLMNPQRSTVWYX]", s):
        raise ValueError("internal stop or non-standard residue")
    return s


def item_for(job, model):
    ents = []
    for ch in job["chains"]:
        e = {"type": "protein", "id": ch["id"], "sequence": clean(ch["seq"])}
        if ch.get("msa"):
            e["msa"] = {"sequences": ch["msa"]}
        ents.append(e)
    ntok = sum(len(e["sequence"]) for e in ents)
    if ntok > 4096:
        raise ValueError(f"{job['fold_id']}: {ntok} tokens > 4096")
    return {"model": model, "all_atom_input": {"sequences": ents}, "include_pae": True,
            "include_pair_chains_iptm": True}


# ---------------------------------------------------------------- structure + metrics
def parse_complex(j):
    """Return per-token arrays (chain, resno, aa, plddt, rep_xyz) and PDB text from j['complex']."""
    c = j["complex"]
    lookup = c["metadata"]["chain_lookup"]
    chains = [lookup[str(x)] for x in c["chain_id"]]
    seq = c["sequence"]
    pl = np.asarray(c["plddt"], dtype=float)
    lines, n, resno, rep = [], 1, {}, []
    rn_list = []
    for t, (a0, a1) in enumerate(c["token_to_atoms"]):
        ch = chains[t]
        resno[ch] = resno.get(ch, 0) + 1
        tok = seq[t]
        rn = tok if len(tok) == 3 else AA3.get(tok, "UNK")
        rn_list.append(rn)
        names = c["atom_names"][a0:a1]
        pos = c["atom_positions"][a0:a1]
        want = "CA" if rn == "GLY" else "CB"
        xyz = None
        for nm, p in zip(names, pos):
            if nm == want:
                xyz = p
        if xyz is None:
            for nm, p in zip(names, pos):
                if nm == "CA":
                    xyz = p
        rep.append(xyz if xyz is not None else [np.nan] * 3)
        for k in range(a0, a1):
            x, y, z = c["atom_positions"][k]
            an, el = c["atom_names"][k], c["atom_elements"][k]
            nm = f" {an:<3}" if len(an) < 4 else an[:4]
            lines.append(f"ATOM  {n:5d} {nm} {rn:>3} {ch}{resno[ch]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00"
                         f"{pl[t] * 100:6.2f}          {el:>2}")
            n += 1
    res_in_chain = []
    cnt = {}
    for ch in chains:
        cnt[ch] = cnt.get(ch, 0) + 1
        res_in_chain.append(cnt[ch])
    return (np.array(chains), np.array(res_in_chain), np.array(rn_list), pl, np.asarray(rep, dtype=float),
            "\n".join(lines + ["END"]) + "\n")


def metrics(j, job):
    chains, resi, rn, pl, xyz, pdb = parse_complex(j)
    ids = [ch["id"] for ch in job["chains"]]
    lens = [int((chains == c).sum()) for c in ids]
    for c, ch in zip(ids, job["chains"]):
        assert int((chains == c).sum()) == len(clean(ch["seq"])), "chain length mismatch"
    pae = np.asarray(j["pae"], dtype=float) if j.get("pae") is not None else None
    out = dict(fold_id=job["fold_id"], model=j.get("model"), ptm=j.get("ptm"), iptm=j.get("interface_ptm"),
               pair_chains_iptm=j.get("pair_chains_iptm"), lengths=lens,
               chain_plddt=[round(float(pl[chains == c].mean()) * 100, 2) for c in ids])
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    inter = chains[:, None] != chains[None, :]
    cont = inter & (d <= 8.0)
    iu = np.triu_indices(len(chains), 1)
    out["contacts"] = int(cont[iu].sum())
    iface = cont.any(1)
    out["iface_residues"] = {c: int((iface & (chains == c)).sum()) for c in ids}
    out["interface_plddt"] = round(float(pl[iface].mean()) * 100, 2) if iface.any() else None
    pae_blocks = {}
    if pae is not None:
        assert pae.shape == (len(chains), len(chains)), f"PAE shape {pae.shape}"
        sym = (pae + pae.T) / 2
        out["iface_pae"] = round(float(sym[cont].mean()), 3) if cont.any() else None
        out["pae_inter_mean"] = round(float(pae[inter].mean()), 3) if inter.any() else None
        out["pae_inter_min"] = round(float(pae[inter].min()), 3) if inter.any() else None
        out["pae_mean"] = round(float(pae.mean()), 3)
        for a in ids:
            for b in ids:
                if a < b:
                    pae_blocks[f"{a}{b}"] = pae[np.ix_(chains == a, chains == b)].astype(np.float16)
                    pae_blocks[f"{b}{a}"] = pae[np.ix_(chains == b, chains == a)].astype(np.float16)
        # per chain pair: contacts and interface PAE
        pp = {}
        for ai, a in enumerate(ids):
            for b in ids[ai + 1:]:
                m = np.ix_(chains == a, chains == b)
                cab = cont[m]
                # added for screen 3: interface pLDDT of the a-b interface and its residues (0-based within chain)
                ra, rb = np.flatnonzero(cab.any(1)), np.flatnonzero(cab.any(0))
                pla, plb = pl[chains == a], pl[chains == b]
                pp[f"{a}{b}"] = dict(contacts=int(cab.sum()),
                                     iface_pae=round(float(sym[m][cab].mean()), 3) if cab.any() else None,
                                     iface_plddt=(round(float(np.r_[pla[ra], plb[rb]].mean()) * 100, 2)
                                                  if cab.any() else None),
                                     res_i=ra.tolist(), res_j=rb.tolist())
        out["pairs"] = pp
    out["plddt_per_res"] = [round(float(x) * 100, 1) for x in pl]
    out["iface_mask"] = iface.astype(int).tolist()
    return out, pdb, pae_blocks


# ---------------------------------------------------------------- state handling
def load_jobs(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def state_path(out):
    return os.path.join(out, "state.json")


def load_state(out):
    p = state_path(out)
    return json.load(open(p)) if os.path.exists(p) else {"batches": [], "tasks": {}}


def save_state(out, st):
    tmp = state_path(out) + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, state_path(out))


def cmd_submit(a):
    H = headers()
    os.makedirs(a.out, exist_ok=True)
    st = load_state(a.out)
    jobs = load_jobs(a.jobs)
    done_ids = set(st["tasks"])
    jobs = [j for j in jobs if j["fold_id"] not in done_ids]
    if a.limit:
        jobs = jobs[:a.limit]
    if not jobs:
        print("nothing to submit")
        return
    json.dump({j["fold_id"]: j for j in jobs}, open(os.path.join(a.out, f"jobs_{len(st['batches'])}.json"), "w"))
    for k in range(0, len(jobs), 2000):
        chunk = jobs[k:k + 2000]
        payload = [item_for(j, a.model) for j in chunk]
        r = post("/submit", {"endpoint": "fold_all_atom", "payload": payload}, H)
        batch = r["data"]["batch_id"]
        st["batches"].append(batch)
        st.setdefault("batch_times", {})[batch] = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                       epoch=time.time(), items=len(chunk))
        for i, j in enumerate(chunk):
            st["tasks"][j["fold_id"]] = {"task_id": f"{batch}-{i:05d}", "batch": batch, "status": "queued",
                                         "jobs_file": f"jobs_{len(st['batches']) - 1 if False else len(st['batches']) - 1}.json"}
        save_state(a.out, st)
        print(f"submitted batch {batch}: {len(chunk)} items, model {a.model}", flush=True)


def cmd_collect(a):
    H = headers()
    st = load_state(a.out)
    jobs = {}
    for fn in sorted(os.listdir(a.out)):
        if fn.startswith("jobs_") and fn.endswith(".json"):
            jobs.update(json.load(open(os.path.join(a.out, fn))))
    os.makedirs(os.path.join(a.out, "pdb"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "pae"), exist_ok=True)
    mpath = os.path.join(a.out, "metrics.jsonl")
    have = set()
    if os.path.exists(mpath):
        have = {json.loads(l)["fold_id"] for l in open(mpath) if l.strip()}
    by_task = {v["task_id"]: k for k, v in st["tasks"].items()}
    while True:
        pending = 0
        for batch in st["batches"]:
            cursor = None
            while True:
                body = {"batch_id": batch, "include_responses": True, "limit": 1000}
                if cursor:
                    body["cursor"] = cursor
                d = post("/list", body, H)["data"]
                for it in d.get("items", []):
                    fid = by_task.get(it["task_id"])
                    if fid is None:
                        continue
                    stt = it["status"]
                    st["tasks"][fid]["status"] = stt
                    if stt in ("queued", "in_progress"):
                        pending += 1
                    elif stt in ("failed", "cancelled"):
                        st["tasks"][fid]["error"] = str(it.get("error"))[:500]
                    elif stt == "done" and fid not in have:
                        resp = it.get("response")
                        if isinstance(resp, dict) and "complex" in resp:
                            j = resp
                        else:
                            s2 = post("/status", {"task_id": it["task_id"]}, H)["data"]
                            url = s2.get("response") if isinstance(s2.get("response"), str) else None
                            j = fetch_json(url) if url else s2.get("response")
                        try:
                            m, pdb, blocks = metrics(j, jobs[fid])
                        except Exception as e:  # noqa: BLE001
                            st["tasks"][fid]["status"] = "parse_error"
                            st["tasks"][fid]["error"] = repr(e)[:500]
                            print(f"PARSE ERROR {fid}: {e!r}", flush=True)
                            continue
                        open(os.path.join(a.out, "pdb", f"{fid}.pdb"), "w").write(pdb)
                        np.savez_compressed(os.path.join(a.out, "pae", f"{fid}.npz"), **blocks)
                        with open(mpath, "a") as f:
                            f.write(json.dumps(m) + "\n")
                        have.add(fid)
                        print(f"done {fid} iptm={m['iptm']} ifacePAE={m.get('iface_pae')} contacts={m['contacts']}",
                              flush=True)
                cursor = d.get("cursor")
                if not cursor or not d.get("items"):
                    break
        save_state(a.out, st)
        counts = {}
        for v in st["tasks"].values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        print(time.strftime("%H:%M:%S"), "status", counts, "collected", len(have), flush=True)
        if pending == 0 and all(v["status"] not in ("queued", "in_progress") for v in st["tasks"].values()):
            break
        time.sleep(a.poll)


def cmd_raw(a):
    """Submit one item and dump the raw response structure (payload verification)."""
    H = headers()
    job = load_jobs(a.jobs)[0]
    item = item_for(job, a.model)
    print("payload:", json.dumps(item)[:600])
    r = post("/submit", {"endpoint": "fold_all_atom", "payload": [item]}, H)
    batch = r["data"]["batch_id"]
    print("batch", batch, r)
    tid = f"{batch}-00000"
    while True:
        time.sleep(10)
        d = post("/status", {"task_id": tid}, H)["data"]
        print(d.get("status"), flush=True)
        if d.get("status") in ("done", "failed", "cancelled"):
            break
    print("status keys:", list(d.keys()))
    resp = d.get("response")
    j = fetch_json(resp) if isinstance(resp, str) else resp
    os.makedirs(a.out, exist_ok=True)
    json.dump(j, open(os.path.join(a.out, "raw_test.json"), "w"))

    def shape(x, depth=0):
        if isinstance(x, dict):
            return {k: shape(v, depth + 1) for k, v in x.items()} if depth < 3 else "dict"
        if isinstance(x, list):
            return f"list[{len(x)}] of {shape(x[0], depth + 1) if x else '-'}"
        return type(x).__name__
    print(json.dumps(shape(j), indent=1)[:4000])
    m, pdb, blocks = metrics(j, job)
    print({k: v for k, v in m.items() if k not in ("plddt_per_res", "iface_mask")})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--jobs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--model", default=MODEL)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--poll", type=int, default=60)
    r = sub.add_parser("raw")
    r.add_argument("--jobs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    {"submit": cmd_submit, "collect": cmd_collect, "raw": cmd_raw}[a.cmd](a)


if __name__ == "__main__":
    main()
