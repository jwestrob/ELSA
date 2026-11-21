#!/usr/bin/env python3
"""Where the lanmodulin-like proteins of the lut collection fall in the frozen embedding-list loci, and what sits with them."""
import duckdb, pandas as pd, numpy as np, collections, sys
RD="/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
con=duckdb.connect(); con.execute("PRAGMA threads=4")
L=pd.read_csv("lanm_gids.tsv",sep="\t"); con.register("lanm",L)
con.execute(f"create table loci as select s,e,e-s+1 n from read_parquet('{RD}/long_loci/frozen/loci/lut_Z/L4bp_t0.02.parquet')")
con.execute(f"create table genes as select * from read_parquet('{RD}/slice_lut/testset/package_k8/universe/genes.parquet')")
grp=con.execute(f"describe select * from read_parquet('{RD}/long_loci/frozen/groups/emb_hom_average_t0.75.parquet')").df()
print("groups cols", grp.column_name.tolist()); gcols=grp.column_name.tolist()
con.execute(f"create table grp as select * from read_parquet('{RD}/long_loci/frozen/groups/emb_hom_average_t0.75.parquet')")
# loci containing each lanmodulin-like gene
con.execute("create table hit as select l.gid, o.s, o.e, o.n from lanm l join loci o on o.s<=l.gid and o.e>=l.gid")
con.execute("create table outer_ as select gid, arg_max(s,n) s, arg_max(e,n) e, max(n) n, count(*) nodes from hit group by gid")
n_in=con.execute("select count(*) from outer_").fetchone()[0]
print(f"lanmodulin-like proteins: {len(L)}; inside a locus: {n_in}; in no locus: {len(L)-n_in}")
print("outermost locus size:", con.execute("select sum((n=2)::int), sum((n between 3 and 4)::int), sum((n between 5 and 9)::int), sum((n between 10 and 19)::int), sum((n>=20)::int), median(n), max(n) from outer_").fetchone(), "(2 / 3-4 / 5-9 / 10-19 / 20+ / median / max)")
# members with Pfam and labels
con.execute(f"create table pf as select cast(protein_id as bigint) gid, string_agg(distinct name, ',' order by name) pfam from read_parquet('{RD}/annot_gpu/lut/PFAM.parquet') group by 1")
con.execute(f"create table lab as select gid, proposed_label_name lab from read_parquet('{RD}/dpann_leads/pass_port/out/lut/lut_slot_rescue_pass.parquet')")
con.execute("""create table mem as select o.gid lanm, g.gid, g.genome, g.contig, g.position_index pos, g.strand, g.length, o.n, pf.pfam, lab.lab
  from outer_ o join genes g on g.gid between o.s and o.e left join pf on pf.gid=g.gid left join lab on lab.gid=g.gid""")
con.execute(f"create table lg as select o.gid lanm, g.genome, l.genus from outer_ o join genes g on g.gid=o.gid join lanm l on l.gid=o.gid")
gkey=[c for c in gcols if c not in ('genome','genome_code')]
print("group table sample:", con.execute("select * from grp limit 2").fetchall())
# partner domains across lanmodulin loci
con.execute("create table dom as select distinct lanm, unnest(string_split(coalesce(pfam, case when lab is not null then 'label:'||lab else 'none' end), ',')) d from mem where gid<>lanm")
tot=con.execute("select count(distinct lanm) from mem").fetchone()[0]
print(f"\nPartner domains in the outermost locus of {tot} lanmodulin-like proteins (loci, genera):")
print(con.execute("""select d, count(distinct dom.lanm) loci, count(distinct lg.genus) genera from dom join lg on lg.lanm=dom.lanm group by 1 order by genera desc, loci desc limit 45""").df().to_string(index=False))
con.execute("copy mem to 'lanm_members.parquet' (format parquet)")
# locus types by partner-domain sets
M=con.execute("select lanm, list(distinct d) ds from dom group by 1").df()
gen=dict(con.execute("select lanm, genus from lg").fetchall()); size=dict(con.execute("select gid, n from outer_").fetchall())
sets={r.lanm:set(x for x in r.ds if x!='none') for r in M.itertuples()}
ids=list(sets); parent={i:i for i in ids}
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
for i,a in enumerate(ids):
    for b in ids[i+1:]:
        A,B=sets[a],sets[b]
        if A and B and len(A&B)/len(A|B)>=0.5: parent[find(a)]=find(b)
cl=collections.defaultdict(list)
for i in ids: cl[find(i)].append(i)
order=sorted(cl.values(), key=lambda v:(-len(set(gen[x] for x in v)),-len(v)))
print(f"\nLocus types (Jaccard >= 0.5 on partner domains, single linkage): {len(order)} types")
for k,v in enumerate(order[:14]):
    c=collections.Counter(d for x in v for d in sets[x])
    core=[f"{d} {n}/{len(v)}" for d,n in c.most_common(14) if n>=max(2,0.4*len(v))]
    gn=collections.Counter(gen[x] for x in v)
    print(f"\nTYPE {k+1}: {len(v)} loci, {len(gn)} genera ({', '.join(f'{g} {n}' for g,n in gn.most_common(5))}); locus size median {int(np.median([size[x] for x in v]))} (range {min(size[x] for x in v)}-{max(size[x] for x in v)})")
    print("   recurring partners:", "; ".join(core) if core else "(none recurring)")
    rep=max(v,key=lambda x:size[x]) if len(v)<3 else sorted(v,key=lambda x:size[x])[len(v)//2]
    m=con.execute("select pos, strand, length, coalesce(pfam, case when lab is not null then '['||lab||']' else '-' end) a, gid=lanm is_lanm, genome from mem where lanm=? order by pos",[rep]).df()
    print("   example", m.genome.iloc[0], ":", " | ".join(("**LanM**" if r.is_lanm else r.a[:38])+f"({r.strand}{r.length})" for r in m.itertuples()))
