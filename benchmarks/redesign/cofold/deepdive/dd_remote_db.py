#!/usr/bin/env python3
"""Deep dive, step 2 (biotite login node, serial, DuckDB threads=2): annotations, curated memberships,
pilin-like control pool and sequences.

Reads (read-only): the canonical Sharur DPANN database copy ($RD/ws_dpann/input/sharur.duckdb, seal 805c93a6),
the package gene universe and proteins_gid.faa. Writes to $RD/cofold/deepdive/:
  ctx/annotations.tsv        every annotation row (all sources) for the protein IDs in ctx_protein_ids.txt
  ctx/curated_members.tsv    system_proteins rows (curated callers: txsscan_system, defensefinder_system, cctyper)
                             joined to secretion_systems / defense_systems type, for the same protein IDs
  ctx/t4p_members.tsv        every txsscan_system member (profile name, system id/type, length, gid)
  ctx/classIII_pfam.tsv      every protein with a Pfam row for PF04021 Class_IIIsignal, PF07963 N_methyl or
                             PF07790 Pilin_N (raw domain rows; used only to select control proteins)
  seqs/dd_seqs.faa           sequences of seq_gids.txt plus the two pools above (header = gid)
"""
import sys

import duckdb
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
DD = f"{RD}/cofold/deepdive"
SH = f"{RD}/ws_dpann/input/sharur.duckdb"


def main():
    con = duckdb.connect(SH, read_only=True)
    con.execute("SET threads=2")
    pids = pd.DataFrame({"protein_id": [l.strip() for l in open(f"{DD}/work/ctx_protein_ids.txt") if l.strip()]})
    con.register("pids", pids)
    ann = con.execute("SELECT a.protein_id, a.source, a.accession, a.name, a.description, a.evalue, a.score, "
                      "a.start_aa, a.end_aa FROM annotations a JOIN pids USING (protein_id)").df()
    ann.to_csv(f"{DD}/ctx/annotations.tsv", sep="\t", index=False)
    print("annotation rows", len(ann), ann.source.value_counts().to_dict(), flush=True)
    cur = con.execute("""
        SELECT sp.protein_id, sp.system_id, sp.system_source, sp.position, sp.profile_name,
               coalesce(ss.system_type, ds.system_type) AS system_type
        FROM system_proteins sp JOIN pids USING (protein_id)
        LEFT JOIN secretion_systems ss ON ss.system_id = sp.system_id AND sp.system_source = 'txsscan_system'
        LEFT JOIN defense_systems ds ON ds.system_id = sp.system_id AND sp.system_source = 'defensefinder_system'
    """).df()
    cur.to_csv(f"{DD}/ctx/curated_members.tsv", sep="\t", index=False)
    print("curated member rows", len(cur), cur.system_source.value_counts().to_dict(), flush=True)
    t4p = con.execute("""
        SELECT sp.protein_id, sp.system_id, sp.profile_name, ss.system_type, p.sequence_length
        FROM system_proteins sp JOIN secretion_systems ss ON ss.system_id = sp.system_id
        JOIN proteins p ON p.protein_id = sp.protein_id
        WHERE sp.system_source = 'txsscan_system'""").df()
    c3 = con.execute("""
        SELECT a.protein_id, a.accession, a.name, a.evalue, a.start_aa, a.end_aa, p.sequence_length
        FROM annotations a JOIN proteins p USING (protein_id)
        WHERE a.source = 'pfam' AND (a.accession LIKE 'PF04021%' OR a.accession LIKE 'PF07963%'
                                     OR a.accession LIKE 'PF07790%')""").df()
    genes = pd.read_parquet(f"{RD}/ws_dpann/package/universe/genes.parquet", columns=["gid", "protein_id"])
    pid2gid = genes.set_index("protein_id").gid
    t4p["gid"] = t4p.protein_id.map(pid2gid)
    c3["gid"] = c3.protein_id.map(pid2gid)
    t4p.to_csv(f"{DD}/ctx/t4p_members.tsv", sep="\t", index=False)
    c3.to_csv(f"{DD}/ctx/classIII_pfam.tsv", sep="\t", index=False)
    print("t4p members", len(t4p), "unmapped", int(t4p.gid.isna().sum()), "; class III pfam rows", len(c3),
          "unmapped", int(c3.gid.isna().sum()), flush=True)
    want = {int(l) for l in open(f"{DD}/work/seq_gids.txt") if l.strip()}
    want |= set(t4p.gid.dropna().astype(int)) | set(c3.gid.dropna().astype(int))
    n = 0
    with open(f"{RD}/ws_dpann/work/proteins_gid.faa") as f, open(f"{DD}/seqs/dd_seqs.faa", "w") as o:
        keep = False
        for line in f:
            if line[0] == ">":
                keep = int(line[1:].split()[0]) in want
                n += keep
            if keep:
                o.write(line)
    print("sequences written", n, "of", len(want), flush=True)


if __name__ == "__main__":
    sys.exit(main())
