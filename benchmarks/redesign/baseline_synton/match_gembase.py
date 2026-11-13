#!/usr/bin/env python
"""Map the gembase genome codes of Herve et al. 2026 to RefSeq assemblies (locked-test preparation; no method run).

Evidence: ST1 rows whose context cluster has exactly one member give an exact pair (gembase gene id of the
representative = the gene itself, RefSeq WP accession). For each gembase genome code (SPEC###.0523.#####) its singleton
WPs are matched against the protein accessions of every candidate assembly in the universe; the assembly with the most
shared WPs is the call. Reported per code: n singleton WPs, best and second-best shared counts, best assembly, and the
rank agreement between gembase gene numbers and universe positions on the best assembly (Spearman over shared WPs on
the code's first replicon, a tie-breaker diagnostic for near-identical strains).

usage (Slurm): match_gembase.py --w $W --threads $SLURM_CPUS_ON_NODE
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    W = Path(a.w)
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"set threads={a.threads}; set temp_directory='{W / 'tmp_duck'}'; set preserve_insertion_order=false")
    st1 = f"read_csv('{W}/zenodo/Supplementary_Table_1_context_clusters.csv', header=true, all_varchar=true)"
    con.execute(f"""
        create temp table sing as
        with c as (select context_cluster, count(*) n, any_value(refseq) wp from {st1} group by 1)
        select substr(context_cluster, 1, 18) code, context_cluster gid_gembase, wp,
               cast(regexp_extract(context_cluster, '_([0-9]+)$', 1) as bigint) gene_no,
               regexp_extract(context_cluster, '\\.([0-9]{{3}}[A-Z])_', 1) replicon
        from c where n = 1""")
    log("singletons", con.execute("select count(*), count(distinct code) from sing").fetchone(), f"{time.time()-t0:.0f}s")
    con.execute(f"""
        create temp table uni as
        select distinct genome, protein_id wp from read_parquet('{W}/universe/genes.parquet')""")
    log("universe genome-WP pairs", con.execute("select count(*) from uni").fetchone(), f"{time.time()-t0:.0f}s")
    # informative WPs only: carried by <= 20 candidate assemblies (keeps the join linear; strain-level evidence)
    con.execute("create temp table wpf as select wp, count(*) nf from uni group by wp")
    con.execute("""
        create temp table hits as
        select s.code, u.genome, count(distinct s.wp) shared
        from (select distinct code, wp from sing) s join wpf f using (wp) join uni u using (wp)
        where f.nf <= 20
        group by 1, 2""")
    con.execute("""
        create temp table best as
        with tot as (select s.code, count(distinct s.wp) n_wp from sing s join wpf f using (wp)
                     where f.nf <= 20 group by 1),
        r as (select code, genome, shared, row_number() over (partition by code order by shared desc, genome) rk
              from hits)
        select t.code, t.n_wp, b.genome best_genome, coalesce(b.shared, 0) best_shared,
               coalesce(s.shared, 0) second_shared, s.genome second_genome
        from tot t left join (select * from r where rk = 1) b using (code)
                   left join (select * from r where rk = 2) s using (code)""")
    # rank agreement of gembase gene numbers with universe positions on the best assembly (main replicon 001C)
    con.execute(f"""
        create temp table rk as
        select b.code, corr(rank_g, rank_u) spearman, count(*) n
        from (select s.code, s.wp, s.gene_no, g.position_index,
                     rank() over (partition by s.code order by s.gene_no) rank_g,
                     rank() over (partition by s.code order by g.contig, g.position_index) rank_u
              from sing s join best b on b.code = s.code
              join read_parquet('{W}/universe/genes.parquet') g on g.genome = b.best_genome and g.protein_id = s.wp
              where s.replicon = '001C') x
        join best b using (code) group by b.code""")
    out = con.execute("""select b.*, rk.spearman, rk.n n_rank from best b left join rk using (code)
                         order by code""").df()
    out.to_parquet(W / "meta/gembase_to_assembly.parquet", index=False)
    frac = out.best_shared / out.n_wp
    margin = out.best_shared - out.second_shared
    summ = dict(
        codes_with_singletons=int(con.execute("select count(distinct code) from sing").fetchone()[0]),
        genome_codes=int(len(out)), codes_with_any_hit=int((out.best_shared > 0).sum()),
        best_frac_quantiles=[float(x) for x in np.nanquantile(frac, [.01, .05, .25, .5])],
        codes_best_frac_ge_0_95=int((frac >= .95).sum()), codes_best_frac_ge_0_8=int((frac >= .8).sum()),
        codes_unique_best=int((margin > 0).sum()), codes_tied_best=int(((margin == 0) & (out.best_shared > 0)).sum()),
        codes_unique_and_ge_0_95=int(((margin > 0) & (frac >= .95)).sum()),
        distinct_best_assemblies=int(out.best_genome.nunique()),
        spearman_quantiles=[float(x) for x in np.nanquantile(out.spearman, [.01, .05, .5])],
        seconds=round(time.time() - t0, 1))
    json.dump(summ, open(W / "meta/gembase_to_assembly.json", "w"), indent=1)
    log("summary", summ)


if __name__ == "__main__":
    main()
