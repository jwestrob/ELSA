#!/usr/bin/env python
"""Port check: this workstream's copied learner (words.run_words) on the DPANN inputs must reproduce the stored DPANN
vocabularies exactly: stage 1 group arm at tau 10 ($RD/vocab/runs/main/tau10/obs) and the synteny-retention arm at tau 8
($RD/vocab/runs2a/neff_syn/tau8/obs, S from $RD/vocab/ctx2a/synS_obs.npy). Read-only on DPANN files."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import words as W  # noqa: E402

RD = W.RD
PKG = RD / "ws_dpann/package"


def main():
    t0 = time.time()
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code"])
    cc, gc = g.contig_code.to_numpy(np.int64), g.genome_code.to_numpy(np.int64)
    t = pd.read_parquet(RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet")
    tok = np.full(len(g), -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    grp = W.load_groups(RD / "lineagefree/groups/emb_hom_average_t0.75.parquet", int(gc.max()) + 1)
    out = {}
    for name, mode, tau, d, S in (("stage1_group_tau10", "group", 10.0, RD / "vocab/runs/main/tau10/obs", None),
                                  ("synteny_tau8", "syn", 8.0, RD / "vocab/runs2a/neff_syn/tau8/obs",
                                   np.load(RD / "vocab/ctx2a/synS_obs.npy"))):
        t1 = time.time()
        r = W.run_words(tok, cc, gc, grp, mode, tau, S=S)
        sw = pd.read_parquet(d / "words.parquet").sort_values("sym").reset_index(drop=True)
        si = dict(np.load(d / "inst.npz"))
        nw = r["words"].sort_values("sym").reset_index(drop=True)
        eq = dict(types=len(sw) == len(nw),
                  words=len(sw) == len(nw) and all(np.array_equal(sw[c].to_numpy(), nw[c].to_numpy()) for c in ("sym", "a", "b", "n", "occ")),
                  inst=all(np.array_equal(si[c], r["inst"][c]) for c in ("sym", "s", "e", "a", "b")))
        if (d / "b0.npy").exists():
            eq["b0"] = bool(np.array_equal(np.load(d / "b0.npy"), r["b0"]))
        out[name] = dict(equal=eq, all_equal=all(eq.values()), stored_types=len(sw), new_types=len(nw),
                         seconds=round(time.time() - t1, 1))
        W.log(name, out[name])
    out["reproduced"] = all(v["all_equal"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = round(time.time() - t0, 1)
    (W.TS / "results/portcheck.json").write_text(json.dumps(out, indent=1) + "\n")
    W.log("portcheck reproduced:", out["reproduced"])


if __name__ == "__main__":
    main()
