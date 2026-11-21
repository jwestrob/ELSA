#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J annot_testset3
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/testset3/annot_testset3_%j.log

# Whole-proteome Pfam 38.1 and KOfam annotation of collection testset3 (842 genomes, 2557405 proteins) through
# Aksha 0.2.3 in GPU mode (--gpu-manifest; manifests from the installed runtime's create_pressed_manifest), --cut_ga
# for both databases (KOfam: ko_list score_type rule), one H200, threads = $SLURM_CPUS_ON_NODE (28, the gpu_h200_4 cap).
# Writes aksha_pfam/ and aksha_kofam/ (raw Aksha output), testset3_pfam.parquet, testset3_kofam.parquet, summaries,
# then STATUS (last), then rewrites /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/STATUS_ALL from every collection's STATUS.
set -euo pipefail
COLL=testset3
FAA=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/testset3/package/proteins_gid.faa
NPROT=2557405
A=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu
O=$A/$COLL
AKSHA=/home/jwestrob/.local/bin/aksha
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
for v in $(compgen -e | grep -E '^(ASTRA_|_ASTRA_|PLAN7_)' || true); do unset "$v"; done
export PYTHONDONTWRITEBYTECODE=1

write_status_all() {
    {
        echo "STATUS_ALL written $(date -u +%FT%TZ) by job ${SLURM_JOB_ID} ($COLL)"
        for c in testset3 srvp_bins lut; do
            if [ -s "$A/$c/STATUS" ]; then echo "$c $(cat "$A/$c/STATUS")";
            else echo "$c NO_STATUS (not submitted or not finished)"; fi
        done
    } > "$A/STATUS_ALL.tmp" && mv "$A/STATUS_ALL.tmp" "$A/STATUS_ALL"
}
fail() {
    trap - ERR TERM
    echo "FAILED $(date -u +%FT%TZ) $1" > "$O/STATUS.tmp" && mv "$O/STATUS.tmp" "$O/STATUS"
    write_status_all
    echo "FAILED: $1"
    exit 1
}
trap 'rc=$?; fail "exit ${rc} at line ${LINENO}: ${BASH_COMMAND}"' ERR
trap 'fail "terminated by signal (scancel or node failure)"' TERM

test ! -e "$O/STATUS"
echo "host $(hostname)  start $(date -u +%FT%TZ)  job $SLURM_JOB_ID  CPUs $SLURM_CPUS_ON_NODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv
$AKSHA --version
sha256sum "$FAA" "$A/manifests/PFAM.manifest.json" "$A/manifests/KOFAM.manifest.json" "$A/meta/pfam_meta.tsv" "$A/meta/kofam_meta.tsv"
n=$(grep -c '^>' "$FAA")
echo "FASTA records $n (expected $NPROT)"
test "$n" -eq "$NPROT"

for DB in PFAM KOFAM; do
    db=${DB,,}
    out=$O/aksha_$db
    test ! -e "$out"
    echo "== $DB GPU search start $(date -u +%FT%TZ)"
    /usr/bin/time -v "$AKSHA" search --installed_hmms "$DB" --cut_ga --prot_in "$FAA" --outdir "$out" \
        --threads "$SLURM_CPUS_ON_NODE" --gpu-manifest "$DB=$A/manifests/$DB.manifest.json"
    echo "== $DB GPU search end $(date -u +%FT%TZ)"
    test -s "$out/${DB}_hits_df.tsv"
    $PY -I "$A/code/to_parquet.py" "$DB" "$out/${DB}_hits_df.tsv" "$A/meta/${db}_meta.tsv" "$NPROT" \
        "$O/${COLL}_${db}.parquet" "$O/${COLL}_${db}_summary.json"
done

line=$($PY -I - "$O" "$COLL" "$NPROT" <<'EOF'
import json, sys
import pyarrow.parquet as pq
o, coll, nprot = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = {db: json.load(open(f"{o}/{coll}_{db}_summary.json")) for db in ("pfam", "kofam")}
g = set()
for db in ("pfam", "kofam"):
    g.update(pq.read_table(f"{o}/{coll}_{db}.parquet", columns=["gid"]).column("gid").to_numpy().tolist())
print(f"PFAM_rows={s['pfam']['rows']} KOFAM_rows={s['kofam']['rows']} "
      f"PFAM_proteins_with_hit={s['pfam']['proteins_with_hit']} KOFAM_proteins_with_hit={s['kofam']['proteins_with_hit']} "
      f"any_proteins_with_hit={len(g)} proteins={nprot}")
EOF
)
echo "DONE $(date -u +%FT%TZ) $line" > "$O/STATUS.tmp" && mv "$O/STATUS.tmp" "$O/STATUS"
cat "$O/STATUS"
write_status_all
trap - ERR TERM
echo "ANNOT_DONE $(date -u +%FT%TZ)"
