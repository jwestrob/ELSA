#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J f17_hhmp
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine/logs/hhmp_%j.log
# f17_machine: the UniRef30 hhblits of the multi-pass MP4610 groups did not finish within ~10 min per group, so these use the
# family MSA only (hhsearch vs Pfam 39 + PDB100); reference profiles CedA, TedA, VirB6 by hhblits (2 iterations, defaults);
# then the role x reference hhalign matrix (MP groups from their family MSAs, -M 50).
set -uo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine
U=/shared/db/uniclust/30_2020_06/UniRef30_2020_06
PF=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/sharur_db/pfam39_hh/pfam39
PDB=/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03
B=/shared/software/bin
cd $W/hh
for g in ref_CedA ref_TedA ref_VirB6; do
  [ -f $g.uni.a3m ] || $B/hhblits -i $W/refs/$g.faa -d $U -n 2 -e 0.001 -oa3m $g.uni.a3m -o /dev/null -cpu $SLURM_CPUS_ON_NODE -v 0
  $B/hhsearch -i $g.uni.a3m -d $PF -d $PDB -o $g.uni.hhr -cpu $SLURM_CPUS_ON_NODE -v 0 -Z 100 -B 100
done
for g in MP4610_s1 MP4610_s2 MP4610_s3 MP4610_s4 MP4610_s5; do
  $B/hhsearch -i $W/roles/$g.aln -M 50 -d $PF -d $PDB -o $g.fam.hhr -cpu $SLURM_CPUS_ON_NODE -v 0 -Z 100 -B 100
  $B/hhmake -i $W/roles/$g.aln -M 50 -o $g.fam.hhm -v 0
done
python3 - <<'PY'
import glob, os, subprocess, re
from concurrent.futures import ThreadPoolExecutor
roles = sorted(f for f in glob.glob('*.uni.a3m') if not f.startswith('ref_') and not f.startswith('MP4610')) + sorted(glob.glob('MP4610_s*.fam.hhm'))
refs = sorted(glob.glob('ref_*.uni.a3m'))
def run(pair):
    r, f = pair
    out = subprocess.run(['/shared/software/bin/hhalign', '-i', r, '-t', f, '-o', '/dev/stdout', '-v', '0'], capture_output=True, text=True).stdout
    name = r.split('.')[0]
    for l in out.splitlines():
        m = re.match(r'\s*1\s+\S+.*?\s+(\d+\.\d+)\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+(\d+)\s+(\d+-\d+)\s+(\d+-\d+)\s*\((\d+)\)', l)
        if m: return (name, f.split('.')[0]) + m.groups()
    return (name, f.split('.')[0], '0', 'NA', 'NA', '0', 'NA', 'NA', 'NA')
with ThreadPoolExecutor(int(os.environ['SLURM_CPUS_ON_NODE'])) as ex:
    res = list(ex.map(run, [(r, f) for r in roles for f in refs]))
with open('hhalign_matrix.tsv', 'w') as fh:
    fh.write('role\tref\tprob\tevalue\tscore\tcols\tq_range\tt_range\tt_len\n')
    for x in res: fh.write('\t'.join(x) + '\n')
PY
echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID" > HHA_STATUS
