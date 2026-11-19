#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf_zre
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/zre_%j.log
# long_loci frozen run, J5c: after the tokens exist, the SECONDARY build with fz.py unchanged (adapter smoke; Z arm with the
# token-cohesion check tree), the kNN-cohesion description lines, and the identity check of the knn outputs against the
# provisional build (loci/lut_Z_provisional). Run time capped from outside by the login-node watchdog.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -s "$FZ/emb/kappa_var_tokens.npy"; test -f "$FZ/logs/done_z_prep"
step z_smoke "$PYE" -u fz.py smoke
step z_arm "$PYE" -u fz.py arm --arm Z --threads "$T"
step z_describe "$PYE" -u fzrun.py describe --threads "$T"
"$PYE" - <<'PY'
import json, hashlib
from pathlib import Path
import pandas as pd
FZ = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen")
out = {}
for nm in ("L1", "L4b_t0.02", "L4bp_t0.02"):
    a = pd.read_parquet(FZ / "loci/lut_Z" / f"{nm}.parquet")
    b = pd.read_parquet(FZ / "loci/lut_Z_provisional" / f"{nm}.parquet")
    out[nm] = dict(rows_unchanged_run=int(len(a)), rows_provisional=int(len(b)), identical=bool(a.equals(b)))
r1 = json.load(open(FZ / "results/zbuild_lut_Z.json")); r0 = json.load(open(FZ / "results/zbuild_lut_Z_provisional.json"))
out["tau_equal"] = r1["tau_star"] == r0["tau_star"]
out["ladder_equal"] = r1["ladder"]["-1"]["counts"] == r0["ladder"]["-1"]["counts"]
out["all_identical"] = all(v["identical"] for k, v in out.items() if isinstance(v, dict)) and out["tau_equal"] and out["ladder_equal"]
(FZ / "results/z_identity.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
PY
stamp "ALL DONE"
