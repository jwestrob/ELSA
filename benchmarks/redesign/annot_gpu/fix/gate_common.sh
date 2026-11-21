# Shared definitions for the gpufix gate jobs (sourced).
J=/groups/banfield/projects/environmental/sr/srvp2020/Jacob
RD=$J/ELSA/redesign_20261111
W=$RD/aksha_check/gpufix
G=$W/gates
NEW=/groups/banfield/users/jwestrob/envs/aksha-0.2.4.dev0
SAMPLE=$RD/annot_gpu/validation/sample4000.faa
REF_PFAM=$RD/annot_gpu/validation/cpu_pfam/PFAM_hits_df.tsv      # production 0.2.3 CPU, job 119815
REF_KOFAM=$RD/annot_gpu/validation/cpu_kofam/KOFAM_hits_df.tsv
PFAM_MANIFEST=$RD/annot_gpu/manifests/PFAM.manifest.json
KOFAM_MANIFEST=$RD/annot_gpu/manifests/KOFAM.manifest.json
COMPARE=$RD/annot_gpu/code/compare.py
export AKSHA_GPU_ATTESTATION_DIR=$G/attcache   # isolated record cache for every gate
for v in $(compgen -e | grep -E '^(ASTRA_|_ASTRA_|PLAN7_|AKSHA_GPU_SELF_ATTEST)' || true); do unset "$v"; done
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$G"

# The build under test, invoked like the production launcher.
aksha_dev() { env -u PYTHONPATH -u PYTHONHOME "$NEW/bin/python" -I "$NEW/bin/aksha" "$@"; }

# run_search LABEL DB OUTDIR [extra aksha args...]; records exit status, time and WARNING count.
run_search() {
    local label=$1 db=$2 out=$3; shift 3
    test ! -e "$out"
    echo "== $label start $(date -u +%FT%TZ)"
    local rc=0
    /usr/bin/time -f "$label wall_s=%e maxrss_kb=%M" "$NEW/bin/python" -I "$NEW/bin/aksha" search \
        --installed_hmms "$db" --cut_ga --prot_in "$SAMPLE" --outdir "$out" \
        --threads "$SLURM_CPUS_ON_NODE" "$@" > "$out.stdout" 2> "$out.stderr" || rc=$?
    tail -n 2 "$out.stderr"
    echo "$label exit=$rc warning_lines=$(grep -c '^WARNING' "$out.stdout" || true) trust_lines=$(grep -c 'GPU bias math' "$out.stdout" || true)"
    grep -E 'GPU bias math|GPU search for|WARNING' "$out.stdout" | cut -c1-400 || true
    echo "== $label end $(date -u +%FT%TZ)"
}

# compare_to LABEL REFERENCE CANDIDATE
compare_to() {
    "$NEW/bin/python" -I "$COMPARE" "$1" "$2" "$3" "$G/$1.compare.json" \
        | grep -E '"(identical_hit_sets|bytes_identical|raw_order_identical|cpu_rows|gpu_rows|full_rows_only_cpu|full_rows_only_gpu|pairs_only_cpu|pairs_only_gpu)"' \
        | tr -d ' \n'; echo "  <- $1"
}
