#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J f17_hha
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine/logs/hha_%j.log
# f17_machine: pairwise profile-profile alignment (hhalign) of every role profile (UniRef30-enriched a3m) against every
# reference profile (CedA/A1/A2/B, TedA/B/C, VirB2/3/4/6, HerA, A. pernix CedA/CedB); unrelated pairs act as the null.
set -uo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine
cd $W/hh
python3 - <<'PY'
import glob, os, subprocess, re
from concurrent.futures import ThreadPoolExecutor
roles = sorted(f for f in glob.glob('*.uni.a3m') if not f.startswith('ref_'))
refs = sorted(glob.glob('ref_*.uni.a3m'))
def run(pair):
    r, f = pair
    out = subprocess.run(['/shared/software/bin/hhalign', '-i', r, '-t', f, '-o', '/dev/stdout', '-v', '0'], capture_output=True, text=True).stdout
    for l in out.splitlines():
        m = re.match(r'\s*1\s+\S+.*?\s+(\d+\.\d+)\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+(\d+)\s+(\d+-\d+)\s+(\d+-\d+)\s*\((\d+)\)', l)
        if m: return (r[:-8], f[:-8]) + m.groups()
    return (r[:-8], f[:-8], '0', 'NA', 'NA', '0', 'NA', 'NA', 'NA')
with ThreadPoolExecutor(int(os.environ['SLURM_CPUS_ON_NODE'])) as ex:
    res = list(ex.map(run, [(r, f) for r in roles for f in refs]))
with open('hhalign_matrix.tsv', 'w') as fh:
    fh.write('role\tref\tprob\tevalue\tscore\tcols\tq_range\tt_range\tt_len\n')
    for x in res: fh.write('\t'.join(x) + '\n')
PY
echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID" > HHA_STATUS
