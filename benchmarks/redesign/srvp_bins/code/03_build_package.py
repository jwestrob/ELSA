#!/usr/bin/env python3
"""SR-VP genome-bin community test package (workstream srvp_bins), DPANN package layout.

usage: 03_build_package.py SELECTION_TSV --pkg DIR --work DIR --threads N STAGE [STAGE ...]
stages: genes universe fasta ribosomal split seal

Inputs (read only):
  SELECTION_TSV   selection/selection_derep_primary.tsv (02_select.py): one bin per joint 95% ANI cluster.
  genome FASTA    input_path (joint_dereplication_v1/genomes/<master_id>.fna -> original bin file); SHA-256 checked
                  against the manifest's file_sha256.
  gene calls      representatives: GTDB-Tk 2.7.2 identify-step Prodigal 2.6.3 single-mode calls (job 18127),
                  marker_genes/<master_id>/<master_id>_protein.faa + prodigal_translation_table.tsv.
                  other picks: Prodigal 2.6.3 run here with GTDB-Tk's procedure (-p single -m, tables 11 and 4, table 4
                  when its coding density exceeds table 11's by > 5 points and is > 70 %); the same procedure is rerun
                  on CONTROL_N representatives and compared with their GTDB-Tk calls (receipt 'genes').
Numbering follows the DPANN package: genome_code = rank of genome (master_id), contig_code = rank of (genome, contig),
gid = rank over (genome_code, contig_code, start, end, protein_id), position_index = rank on the contig.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

GTDB_MG = ("/groups/banfield/projects/environmental/sr/luis/srvp-all/srvp-fina/05_taxonomy_and_community/"
           "gtdb_representatives_r232_v1/runs/r232_18127/gtdbtk/identify/intermediate_results/marker_genes")
R232 = "/groups/banfield/projects/environmental/sr/luis/srvp-all/srvp-fina/05_taxonomy_and_community/r232_integrated_results_v1/genome_master_r232.tsv"
CLUST = "/groups/banfield/projects/environmental/sr/luis/srvp-all/srvp-fina/03_assemblies_and_genomes/joint_dereplication_results_v1/clusters.tsv"
RIBO_HMM = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/testset/work/ribo.hmm"
PRODIGAL = "/shared/software/bin/prodigal"
CONTROL_N = 6


def sha(p, chunk=16 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def receipt(pkg, stage, payload):
    d = Path(pkg) / "receipts"; d.mkdir(parents=True, exist_ok=True)
    payload = dict(stage=stage, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   code_sha256=sha(__file__), **payload)
    tmp = d / f".{stage}.json.tmp"; tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    tmp.rename(d / f"{stage}.json")
    print(json.dumps(payload, default=str)[:4000], flush=True)


def write_parquet(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd"); tmp.rename(path)


def read_fasta_contigs(path):
    names, lens, seqs = [], [], []
    cur, buf = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    s = "".join(buf); names.append(cur); lens.append(len(s)); seqs.append(s)
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        s = "".join(buf); names.append(cur); lens.append(len(s)); seqs.append(s)
    return names, lens, seqs


def parse_prodigal_faa(path):
    """Prodigal .faa: '>{contig}_{n} # start # end # strand # ID=..;partial=..;start_type=..;...'"""
    rows, seq, hdr = [], [], None
    def flush():
        if hdr is None:
            return
        parts = hdr.split(" # ")
        pid = parts[0]; attrs = dict(kv.split("=", 1) for kv in parts[4].strip().split(";") if "=" in kv)
        rows.append((pid, pid.rsplit("_", 1)[0], int(parts[1]), int(parts[2]), int(parts[3]),
                     attrs.get("partial"), attrs.get("start_type"), "".join(seq).rstrip("*")))
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                flush(); hdr = line[1:].rstrip("\n"); seq = []
            else:
                seq.append(line.strip())
    flush()
    return rows


def run_prodigal(fna, workdir, total_len):
    """GTDB-Tk procedure: -p single (meta below 100 kb) -m, tables 11 and 4, choose by coding density."""
    out = {}
    mode = "single" if total_len >= 100000 else "meta"
    for t in (11, 4):
        faa = Path(workdir) / f"t{t}.faa"; gff = Path(workdir) / f"t{t}.gff"
        subprocess.run([PRODIGAL, "-m", "-p", mode, "-g", str(t), "-f", "gff", "-q", "-i", fna, "-a", str(faa),
                        "-o", str(gff)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows = parse_prodigal_faa(faa)
        coding = sum(abs(r[3] - r[2]) + 1 for r in rows)
        out[t] = (rows, 100.0 * coding / total_len)
    best = 4 if (out[4][1] - out[11][1] > 5 and out[4][1] > 70) else 11
    return out[best][0], best, out[11][1], out[4][1], mode


def genome_job(args):
    master_id, fna, expect_sha, is_rep, control, workroot = args
    t0 = time.time()
    file_sha = sha(fna)
    names, lens, seqs = read_fasta_contigs(fna)
    total = sum(lens)
    res = dict(master_id=master_id, file_sha256=file_sha, sha_ok=file_sha == expect_sha, n_contigs=len(names),
               genome_bp=total, dup_contig_names=len(names) - len(set(names)))
    contigs = pd.DataFrame(dict(genome=master_id, contig=names, length_nt=lens, file_order=range(len(names))))
    rows, table, cd11, cd4, mode, src = None, None, None, None, None, None
    if is_rep:
        d = Path(GTDB_MG) / master_id
        rows = parse_prodigal_faa(d / f"{master_id}_protein.faa")
        tt = dict(l.rstrip("\n").split("\t") for l in open(d / "prodigal_translation_table.tsv") if "\t" in l)
        table = int(tt["best_translation_table"]); cd11 = float(tt["coding_density_11"]); cd4 = float(tt["coding_density_4"])
        src = "gtdbtk_2.7.2_prodigal_2.6.3"
    if (not is_rep) or control:
        w = Path(tempfile.mkdtemp(dir=workroot, prefix=master_id + "_"))
        prow, ptable, pcd11, pcd4, mode = run_prodigal(fna, w, total)
        shutil.rmtree(w, ignore_errors=True)
        if control:
            a = {(r[1], r[2], r[3], r[4]) for r in rows}; b = {(r[1], r[2], r[3], r[4]) for r in prow}
            res.update(control_gtdb_genes=len(a), control_rerun_genes=len(b), control_shared=len(a & b),
                       control_table_gtdb=table, control_table_rerun=ptable,
                       control_seq_identical=sum(1 for x, y in zip(sorted(rows), sorted(prow)) if x[7] == y[7]),
                       control_cd11=(cd11, round(pcd11, 2)), control_cd4=(cd4, round(pcd4, 2)))
        else:
            rows, table, cd11, cd4, src = prow, ptable, pcd11, pcd4, f"prodigal_2.6.3_gtdbtk_procedure_{mode}"
    genes = pd.DataFrame(rows, columns=["prodigal_id", "contig", "start", "end", "strand", "partial", "start_type", "seq"])
    genes.insert(0, "genome", master_id)
    genes["protein_id"] = master_id + "|" + genes.prodigal_id
    res.update(n_genes=len(genes), translation_table=table, coding_density_11=cd11, coding_density_4=cd4,
               gene_source=src, contigs_unknown=int((~genes.contig.isin(set(names))).sum()),
               end_beyond=int((genes["end"] > genes.contig.map(dict(zip(names, lens)))).sum()),
               seconds=round(time.time() - t0, 2))
    return res, genes, contigs


def st_genes(a, sel):
    work = Path(a.work); (work / "prodigal_tmp").mkdir(parents=True, exist_ok=True)
    reps = sel[sel.is_rep].master_id.tolist()
    rng = np.random.default_rng(17)
    control = set(rng.choice(sorted(reps), size=min(CONTROL_N, len(reps)), replace=False))
    jobs = [(r.master_id, r.input_path, r.file_sha256, bool(r.is_rep), r.master_id in control, str(work / "prodigal_tmp"))
            for r in sel.itertuples()]
    out_g, out_c, recs = [], [], []
    with ProcessPoolExecutor(a.threads) as ex:
        for res, g, c in ex.map(genome_job, jobs, chunksize=1):
            recs.append(res); out_g.append(g); out_c.append(c)
    rec = pd.DataFrame(recs)
    genes = pd.concat(out_g, ignore_index=True); contigs = pd.concat(out_c, ignore_index=True)
    write_parquet(genes, work / "genes_raw.parquet"); write_parquet(contigs, work / "contigs_all_raw.parquet")
    write_parquet(rec, work / "genome_gene_records.parquet")
    ctl = rec[rec.get("control_gtdb_genes").notna()] if "control_gtdb_genes" in rec else rec.iloc[:0]
    receipt(a.pkg, "genes", dict(
        genomes=len(rec), sha_mismatch=int((~rec.sha_ok).sum()), genes=len(genes),
        gene_source=rec.gene_source.value_counts().to_dict(), translation_table=rec.translation_table.value_counts().to_dict(),
        contigs=int(rec.n_contigs.sum()), dup_contig_names=int(rec.dup_contig_names.sum()),
        genes_on_unknown_contig=int(rec.contigs_unknown.sum()), genes_end_beyond_contig=int(rec.end_beyond.sum()),
        control=ctl.drop(columns=[c for c in ("file_sha256",) if c in ctl]).to_dict("records")))
    assert (rec.sha_ok).all() and rec.contigs_unknown.sum() == 0 and rec.end_beyond.sum() == 0


def strip(s):
    return s.split("__", 1)[1] if isinstance(s, str) and "__" in s else (s if isinstance(s, str) else "")


def st_universe(a, sel):
    import duckdb
    pkg, work = Path(a.pkg), Path(a.work)
    con = duckdb.connect(); con.execute(f"PRAGMA threads={a.threads}; SET temp_directory='{work}/duck_tmp'")
    con.execute(f"CREATE TABLE p AS SELECT * EXCLUDE (seq), length(seq) AS length FROM read_parquet('{work}/genes_raw.parquet')")
    con.execute(f"CREATE TABLE ca AS SELECT * FROM read_parquet('{work}/contigs_all_raw.parquet')")
    con.execute("CREATE TABLE gm AS SELECT genome, row_number() OVER (ORDER BY genome) - 1 AS genome_code FROM (SELECT DISTINCT genome FROM ca)")
    con.execute("""CREATE TABLE cg AS SELECT genome, contig, row_number() OVER (ORDER BY genome, contig) - 1 AS contig_code
                   FROM (SELECT DISTINCT genome, contig FROM p)""")
    con.execute("""CREATE TABLE u AS
        SELECT row_number() OVER (ORDER BY gm.genome_code, cg.contig_code, p.start, p."end", p.protein_id) - 1 AS gid,
               p.protein_id, p.genome, gm.genome_code::INTEGER AS genome_code, p.contig, cg.contig_code,
               row_number() OVER (PARTITION BY cg.contig_code ORDER BY p.start, p."end", p.protein_id) - 1 AS position_index,
               p.strand::TINYINT AS strand, p.length::BIGINT AS length, p.start::BIGINT AS start, p."end"::BIGINT AS "end",
               p.partial, p.start_type, p.prodigal_id, ca.length_nt AS contig_length, ca.file_order AS contig_file_order
        FROM p JOIN gm USING (genome) JOIN cg USING (genome, contig) JOIN ca USING (genome, contig)""")
    q = lambda s: con.execute(s).fetchone()
    n, npid = q("SELECT count(*), count(DISTINCT protein_id) FROM u"); assert n == npid
    assert q("SELECT min(gid) = 0 AND max(gid) = count(*) - 1 FROM u")[0]
    assert q("""SELECT count(*) FROM (SELECT contig_code, max(gid) - min(gid) + 1 w, count(*) n, max(position_index) mp
                FROM u GROUP BY 1) WHERE w <> n OR mp <> n - 1""")[0] == 0
    con.execute(f"COPY (SELECT * FROM u ORDER BY gid) TO '{work}/universe_full.parquet' (FORMAT parquet, COMPRESSION zstd)")
    (pkg / "universe").mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (SELECT gid::BIGINT AS gid, protein_id, genome, genome_code, contig, contig_code::BIGINT AS contig_code,
                        position_index::BIGINT AS position_index, strand, length, start, "end" FROM u ORDER BY gid)
                    TO '{pkg}/universe/genes.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"""COPY (SELECT genome, contig, contig_code::BIGINT AS contig_code, count(*)::BIGINT AS n_genes,
                        any_value(contig_length) AS length_nt, min(gid)::BIGINT AS first_gid,
                        any_value(contig_file_order)::BIGINT AS file_order
                        FROM u GROUP BY ALL ORDER BY contig_code)
                    TO '{pkg}/universe/contigs.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    # every contig of every bin, in FASTA order, including contigs without a gene call
    con.execute(f"""COPY (SELECT ca.genome, gm.genome_code::INTEGER AS genome_code, ca.contig, ca.file_order::BIGINT AS file_order,
                        ca.length_nt::BIGINT AS length_nt, cg.contig_code::BIGINT AS contig_code,
                        coalesce(k.n, 0)::BIGINT AS n_genes
                        FROM ca JOIN gm USING (genome) LEFT JOIN cg USING (genome, contig)
                        LEFT JOIN (SELECT contig_code, count(*) n FROM u GROUP BY 1) k USING (contig_code)
                        ORDER BY genome_code, file_order)
                    TO '{pkg}/universe/contigs_all.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    # gene flags: Prodigal partial flag, truncation (DPANN repair convention), contig-edge position
    con.execute(f"""COPY (SELECT gid::BIGINT AS gid, partial, start_type,
                        CASE WHEN partial IS NOT NULL THEN partial IN ('10','01','11')
                             ELSE (start <= 3 OR "end" >= contig_length - 2) END AS truncated,
                        (start <= 3 OR "end" >= contig_length - 2) AS at_contig_end_nt,
                        (position_index = 0) AS first_on_contig,
                        (position_index = max(position_index) OVER (PARTITION BY contig_code)) AS last_on_contig,
                        NULL::BOOLEAN AS contig_circular
                        FROM u ORDER BY gid) TO '{pkg}/universe/gene_flags.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    # genome table
    rec = pd.read_parquet(work / "genome_gene_records.parquet")
    gm = con.execute("SELECT * FROM gm").df()
    gstat = con.execute("""SELECT genome, count(*) AS n_genes, count(DISTINCT contig_code) AS n_contigs_with_genes,
                           avg((strand = -1)::DOUBLE) AS minus_fraction,
                           sum((partial IN ('10','01','11'))::INT) AS n_partial FROM u GROUP BY 1""").df()
    r232 = pd.read_csv(R232, sep="\t", dtype=str, keep_default_na=False)
    cl = pd.read_csv(CLUST, sep="\t", dtype=str, keep_default_na=False)[["cluster_id", "members", "pacbio_members", "illumina_members"]]
    s = sel.rename(columns={"master_id": "genome"})
    g = (gm.merge(s, on="genome", how="left").merge(gstat, on="genome", how="left")
         .merge(rec.rename(columns={"master_id": "genome"})[["genome", "n_contigs", "genome_bp", "translation_table",
                                                              "coding_density_11", "coding_density_4", "gene_source"]],
                on="genome", how="left", suffixes=("_register", ""))
         .merge(r232[["master_id", "classification_r232", "gtdb_has_warning", "gtdb_closest_genome_reference",
                      "gtdb_closest_genome_ani", "representative_graph_supported_circle", "representative_finished_status"]]
                .rename(columns={"master_id": "genome"}), on="genome", how="left", suffixes=("", "_r232"))
         .merge(cl, on="cluster_id", how="left"))
    own = g["classification_r232"].fillna("")
    r220 = np.where(g.taxonomy_release.fillna("") == "r220", g.classification.fillna(""), "")
    g["tax"] = np.where(own != "", own, np.where(r220 != "", r220, g.cluster_rep_classification_r232.fillna("")))
    g["tax_source"] = np.where(own != "", "gtdbtk_2.7.2_r232_own", np.where(r220 != "", "gtdbtk_r220_own",
                               "gtdbtk_r232_cluster_representative"))
    for i, rk in enumerate(["domain", "phylum", "class", "ord", "family", "genus", "species"]):
        g[rk] = [strip(t.split(";")[i]) if len(t.split(";")) > i else "" for t in g.tax]
    g = g.rename(columns={"genome_id": "genome_name", "sample": "sample_provisional", "cluster_id": "derep_cluster_95ani",
                          "is_rep": "is_drep_representative", "members": "derep_cluster_members"})
    keep = ["genome", "genome_code", "genome_name", "n_genes", "n_contigs", "n_contigs_with_genes", "genome_bp", "n50_bp",
            "completeness", "contamination", "quality_source_path", "tax", "tax_source", "domain", "phylum", "class", "ord",
            "family", "genus", "species", "gtdb_has_warning", "gtdb_closest_genome_reference", "gtdb_closest_genome_ani",
            "platform", "cohort", "sample_provisional", "sample_key", "derep_cluster_95ani", "derep_cluster_members",
            "pacbio_members", "illumina_members", "is_drep_representative", "representative_master_id", "reported_circular",
            "representative_graph_supported_circle", "representative_finished_status", "translation_table",
            "coding_density_11", "coding_density_4", "gene_source", "n_partial", "minus_fraction", "input_path",
            "preferred_path", "file_sha256", "sequence_sha256"]
    g = g[keep].sort_values("genome_code")
    for c in ("completeness", "contamination", "n50_bp", "genome_bp"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    write_parquet(g, pkg / "universe" / "genomes.parquet")
    qc = dict(genes=n, genomes=len(g), contigs_with_genes=q("SELECT count(DISTINCT contig_code) FROM u")[0],
              contigs_all=q("SELECT count(*) FROM ca")[0],
              contigs_without_genes=q("SELECT count(*) FROM ca LEFT JOIN cg USING (genome, contig) WHERE cg.contig_code IS NULL")[0],
              strand_minus_fraction=q("SELECT avg((strand=-1)::DOUBLE) FROM u")[0],
              partial_values=con.execute("SELECT partial, count(*) FROM u GROUP BY 1 ORDER BY 2 DESC").fetchall(),
              single_gene_contigs=q("SELECT count(*) FROM (SELECT contig_code FROM u GROUP BY 1 HAVING count(*) = 1)")[0],
              contig_size_hist=con.execute("""SELECT CASE WHEN n=1 THEN '1' WHEN n<=4 THEN '2-4' WHEN n<=9 THEN '5-9'
                    WHEN n<=49 THEN '10-49' ELSE '50+' END b, count(*), sum(n) FROM (SELECT contig_code, count(*) n FROM u
                    GROUP BY 1) GROUP BY 1 ORDER BY 1""").fetchall(),
              cds_length_not_3x=q("""SELECT count(*) FROM u WHERE ("end" - start + 1) <> 3 * (length + 1)
                                     AND ("end" - start + 1) <> 3 * length""")[0],
              tax_source=g.tax_source.value_counts().to_dict(), domains=g.domain.value_counts().to_dict(),
              phyla=int(g.phylum.replace("", np.nan).nunique()), platform=g.platform.value_counts().to_dict(),
              single_contig_genomes=int((g.n_contigs == 1).sum()),
              register_vs_measured_bp_mismatch=int((sel.set_index("master_id").genome_bp.astype(float)
                                                    .reindex(g.genome).to_numpy() != g.genome_bp.to_numpy()).sum()))
    receipt(pkg, "universe", qc)


def st_fasta(a, sel):
    pkg, work = Path(a.pkg), Path(a.work)
    raw = pq.read_table(work / "genes_raw.parquet", columns=["protein_id", "seq"]).to_pandas()
    u = pq.read_table(pkg / "universe" / "genes.parquet", columns=["gid", "protein_id"]).to_pandas()
    m = u.merge(raw, on="protein_id", how="left").sort_values("gid")
    assert len(m) == len(u) and m.seq.notna().all()
    out = pkg / "universe" / "proteins.faa"; tmp = out.with_name(".proteins.faa.tmp")
    with open(tmp, "w") as f:
        for gid, s in zip(m.gid.to_numpy(), m.seq.to_numpy()):
            f.write(f">{gid}\n{s}\n")
    tmp.rename(out)
    lens = m.seq.str.len()
    h = m.seq.map(lambda s: hashlib.md5(s.encode()).digest())
    first = ~h.duplicated()
    receipt(pkg, "fasta", dict(path=str(out), records=len(m), residues=int(lens.sum()), empty=int((lens == 0).sum()),
                               internal_stop=int(m.seq.str.contains(r"\*").sum()),
                               unique_sequences=int(first.sum()), unique_residues=int(lens[first].sum()),
                               sha256=sha(out)))


def st_ribosomal(a, sel):
    """hmmsearch of the 121 Pfam 38.1 ^ribosom(al)?_ models (--cut_ga) over every package protein, then runs by the
    standing definition (ribosomal/code/ribosomal_check.py): per contig, maximal runs of marked genes with <= gap unmarked
    genes inside (g0, g2), runs with >= 8 marked kept, m16 = >= 16 marked (primary set g2_m16)."""
    pkg, work = Path(a.pkg), Path(a.work)
    W = work / "ribo"; shutil.rmtree(W, ignore_errors=True); (W / "parts").mkdir(parents=True); (W / "out").mkdir()
    T = a.threads
    handles = [open(W / "parts" / f"p_{k}.faa", "w") for k in range(T)]
    k = -1
    with open(pkg / "universe" / "proteins.faa") as f:
        for line in f:
            if line.startswith(">"):
                k = (k + 1) % T
            handles[k].write(line)
    for h in handles:
        h.close()
    t0 = time.time()
    procs = [subprocess.Popen(["hmmsearch", "--cut_ga", "--cpu", "1", "--noali", "-o", "/dev/null", "--domtblout",
                               str(W / "out" / f"p_{k}.domtbl"), RIBO_HMM, str(W / "parts" / f"p_{k}.faa")])
             for k in range(T)]
    rc = [p.wait() for p in procs]; assert all(r == 0 for r in rc), rc
    hmm_s = time.time() - t0
    rows = []
    for k in range(T):
        for line in open(W / "out" / f"p_{k}.domtbl"):
            if line.startswith("#"):
                continue
            f = line.split()
            rows.append((int(f[0]), f[3], f[4], float(f[7])))
    h = pd.DataFrame(rows, columns=["gid", "name", "acc", "score"])
    genes = pq.read_table(pkg / "universe" / "genes.parquet", columns=["gid", "genome_code", "contig_code", "position_index"]).to_pandas().sort_values("gid")
    assert np.array_equal(genes.gid.to_numpy(), np.arange(len(genes)))
    contigs = pd.read_parquet(pkg / "universe" / "contigs.parquet").set_index("contig_code")
    gm = pd.read_parquet(pkg / "universe" / "genomes.parquet").set_index("genome_code")
    assert h.name.str.lower().str.match(r"^ribosom(al)?_").all()
    top = h.sort_values("score", ascending=False).drop_duplicates("gid").set_index("gid").name
    alln = h.groupby("gid").name.apply(lambda s: ";".join(sorted(set(s))))
    marked = pd.DataFrame({"gid": top.index.to_numpy(), "marker": top.to_numpy()})
    marked["markers"] = alln.reindex(marked.gid).to_numpy()
    marked = marked.sort_values("gid").reset_index(drop=True)
    (pkg / "ribosomal").mkdir(exist_ok=True)
    write_parquet(marked, pkg / "ribosomal" / "marked.parquet")
    cc = genes.contig_code.to_numpy(); pos = genes.position_index.to_numpy(); gcode = genes.genome_code.to_numpy()
    frames = []
    for gap in (0, 2):
        m = np.sort(marked.gid.to_numpy())
        new = np.r_[True, (np.diff(m) > gap + 1) | (np.diff(cc[m]) != 0)]
        rid = np.cumsum(new) - 1; last = np.r_[new[1:], True]
        r = pd.DataFrame(dict(gid_start=m[new], gid_end=m[last], n_marked=np.bincount(rid)))
        r = r[r.n_marked >= 8].reset_index(drop=True); r.insert(0, "run_set", f"g{gap}"); frames.append(r)
    runs = pd.concat(frames, ignore_index=True)
    runs.insert(1, "run_id", range(len(runs)))
    runs["m16"] = runs.n_marked >= 16
    runs["n_genes"] = runs.gid_end - runs.gid_start + 1
    runs["contig_code"] = cc[runs.gid_start]; runs["genome_code"] = gcode[runs.gid_start]
    runs["pos_start"] = pos[runs.gid_start]; runs["pos_end"] = pos[runs.gid_end]
    ci = contigs.loc[runs.contig_code]
    runs["contig"] = ci.contig.to_numpy(); runs["genome"] = ci.genome.to_numpy()
    runs["contig_first_gid"] = ci.first_gid.to_numpy(); runs["contig_n_genes"] = ci.n_genes.to_numpy()
    runs["at_contig_edge"] = (runs.pos_start == 0) | (runs.pos_end == runs.contig_n_genes - 1)
    gg = gm.loc[runs.genome_code]
    for c in ["domain", "phylum", "class", "ord", "family", "genus", "completeness", "platform"]:
        runs[c] = gg[c].to_numpy()
    write_parquet(runs, pkg / "ribosomal" / "runs.parquet")
    p = runs[(runs.run_set == "g2") & runs.m16]
    s = dict(hmm=RIBO_HMM, hmm_sha256=sha(RIBO_HMM), hmmsearch_seconds=round(hmm_s, 1), threads=T, hits=len(h),
             marked_genes=len(marked), models_hit=int(h.name.nunique()),
             runs={f"{rs}_m{mm}": int(((runs.run_set == rs) & (runs.n_marked >= mm)).sum()) for rs in ("g0", "g2") for mm in (8, 16)},
             g2_m16_genomes=int(p.genome_code.nunique()), genomes=len(gm),
             g2_m16_at_contig_edge=int(p.at_contig_edge.sum()),
             g2_m16_by_domain=p.domain.value_counts().to_dict(), g2_m16_by_platform=p.platform.value_counts().to_dict(),
             g2_m16_n_marked_quantiles=p.n_marked.quantile([0, 0.5, 1]).tolist(),
             genomes_with_ge16_marked_anywhere=int((marked.merge(genes, on="gid").groupby("genome_code").size() >= 16).sum()))
    receipt(pkg, "ribosomal", s)
    shutil.rmtree(W / "parts", ignore_errors=True)


def st_split(a, sel):
    pkg = Path(a.pkg)
    g = pd.read_parquet(pkg / "universe" / "genomes.parquet", columns=["genome", "genome_code", "n_genes", "phylum", "platform"])
    g["half"] = np.where(g.genome_code % 2 == 0, "test", "tune")
    write_parquet(g[["genome", "genome_code", "half"]], pkg / "split" / "genome_halves.parquet")
    receipt(pkg, "split", dict(rule="test = even genome_code, tune = odd genome_code (DPANN repair/predeclared.json)",
                               genomes=g.groupby("half").size().to_dict(), genes=g.groupby("half").n_genes.sum().to_dict(),
                               phyla=g.groupby("half").phylum.nunique().to_dict(),
                               platform={f"{h}|{p}": int(n) for (h, p), n in g.groupby(["half", "platform"]).size().items()}))


def st_seal(a, sel):
    pkg = Path(a.pkg)
    files = []
    for root, _, fs in os.walk(pkg):
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(root, f), pkg)
            if rel.startswith(".") or "/." in rel or rel in ("SEAL.txt", "seal.json", "MANIFEST.tsv") or rel.startswith("receipts/"):
                continue
            files.append(rel)
    files.sort()
    rows = []
    for rel in files:
        p = pkg / rel
        n = pq.ParquetFile(p).metadata.num_rows if rel.endswith(".parquet") else (
            sum(1 for l in open(p) if l.startswith(">")) if rel.endswith(".faa") else "")
        rows.append((rel, n, p.stat().st_size, sha(p)))
    combined = hashlib.sha256("".join(f"{r[0]}\t{r[3]}\n" for r in rows).encode()).hexdigest()
    seal = dict(package=str(pkg), sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                files={r[0]: r[3] for r in rows}, seal_sha256=combined,
                rule="sha256 over the sorted lines '<path>\\t<sha256>\\n' of every package file except receipts and seal files")
    (pkg / "seal.json").write_text(json.dumps(seal, indent=1) + "\n")
    lines = ["SR-VP genome-bin community test package (dereplicated primary tier): SEAL",
             f"written_utc\t{seal['sealed_utc']}", f"seal_sha256\t{combined}", "",
             "[status]",
             "No ELSA method has been run on this universe, nothing has been embedded and nothing has been scored.",
             "", "[files]\tsha256\tbytes\trows\tpath"]
    lines += [f"file\t{r[3]}\t{r[2]}\t{r[1]}\t{r[0]}" for r in rows]
    (pkg / "SEAL.txt").write_text("\n".join(lines) + "\n")
    with open(pkg / "MANIFEST.tsv", "w") as f:
        f.write("path\trows\tbytes\tsha256\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print("\n".join(lines), flush=True)


STAGES = dict(genes=st_genes, universe=st_universe, fasta=st_fasta, ribosomal=st_ribosomal, split=st_split, seal=st_seal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection"); ap.add_argument("--pkg", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    a = ap.parse_args()
    Path(a.work, "duck_tmp").mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(a.selection, sep="\t", dtype=str, keep_default_na=False)
    sel["is_rep"] = sel.is_rep == "True"
    for c in ("completeness", "contamination", "genome_bp", "contigs", "n50_bp"):
        sel[c] = pd.to_numeric(sel[c], errors="coerce")
    assert sel.master_id.is_unique
    for s in a.stages:
        t0 = time.time(); print(f"== stage {s} start {time.strftime('%H:%M:%S')}", flush=True)
        STAGES[s](a, sel)
        print(f"== stage {s} done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
