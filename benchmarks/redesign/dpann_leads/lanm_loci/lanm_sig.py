#!/usr/bin/env python3
"""EF-loop signature of the lanmodulin-like proteins (loops, loops with Pro at loop position 2, loop spacing) by locus type."""
import duckdb, pandas as pd, numpy as np, re, collections
RD="/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
L=pd.read_csv("lanm_gids.tsv",sep="\t"); want=set(map(str,L.gid))
seq={}; cur=None
with open(f"{RD}/slice_lut/testset/package_k8/proteins_gid.faa") as fh:
    for line in fh:
        if line[0]==">":
            cur=line[1:].strip() if line[1:].strip() in want else None
            if cur: seq[cur]=""
        elif cur: seq[cur]+=line.strip()
pat=re.compile(r"(?=([DN].[DNS].[DNSG]G.[ILVMFYC]...[ED]))")
rows=[]
for g,s in seq.items():
    hits=[(m.start(),m.group(1)) for m in pat.finditer(s)]
    # keep non-overlapping loops
    loops=[]; last=-99
    for st,x in hits:
        if st-last>=12: loops.append((st,x)); last=st
    sp=[b[0]-a[0] for a,b in zip(loops,loops[1:])]
    rows.append(dict(gid=int(g),len=len(s),loops=len(loops),p2=sum(x[1]=="P" for _,x in loops),spacing=float(np.median(sp)) if sp else np.nan))
S=pd.DataFrame(rows); S["lanm_type"]=(S.p2>=2)
con=duckdb.connect(); con.register("S",S); con.register("L",L)
M=con.execute("select * from read_parquet('lanm_members.parquet')").df()
inloc=set(M.lanm.unique()); S["in_locus"]=S.gid.isin(inloc)
S=S.merge(L,on="gid")
print("all 347: loops distribution", S.loops.value_counts().sort_index().to_dict())
print("Pro at loop position 2 in >= 2 loops (lanmodulin-type):", int(S.lanm_type.sum()), "of", len(S), "| in a locus:", int((S.lanm_type&S.in_locus).sum()), "| not in a locus:", int((S.lanm_type&~S.in_locus).sum()))
print("other EF-hand proteins:", int((~S.lanm_type).sum()), "| in a locus:", int(((~S.lanm_type)&S.in_locus).sum()))
print("median loop spacing: lanmodulin-type", S[S.lanm_type].spacing.median(), "| others", S[~S.lanm_type].spacing.median(), "| median length", S[S.lanm_type].len.median(), S[~S.lanm_type].len.median())
print("lanmodulin-type by genus (top):", S[S.lanm_type].genus.value_counts().head(8).to_dict())
# partner classes per protein
def cls(df):
    d=set(x for p in df.pfam.dropna() for x in p.split(","))
    out=[]
    if {"Response_reg","Trans_reg_C"}<=d and ("HisKA" in d or "HATPase_c" in d): out.append("OmpR-type two-component pair")
    if "Sigma70_r2" in d: out.append("ECF sigma factor")
    if "TonB_dep_Rec_b-barrel" in d: out.append("TonB receptor")
    if "Peripla_BP_6" in d and "Lactonase" in d: out.append("lut transporter genes (lutA/lutB domains)")
    if "PQQ" in d or "PQQ_2" in d: out.append("PQQ dehydrogenase")
    if "HisKA_3" in d and "GerE" in d: out.append("NarL-type pair (HisKA_3 + GerE)")
    if "SoxY" in d or "SoxZ" in d: out.append("SoxYZ")
    if "4HFCP_synth" in d: out.append("4HFCP_synth block")
    return out or ["none of these"]
c=collections.Counter(); cl=collections.Counter(); gen=collections.defaultdict(set); genl=collections.defaultdict(set)
sig=dict(zip(S.gid,S.lanm_type)); genus=dict(zip(S.gid,S.genus))
for lanm,df in M.groupby("lanm"):
    for k in cls(df[df.gid!=lanm]):
        if sig.get(lanm): cl[k]+=1; genl[k].add(genus[lanm])
        else: c[k]+=1; gen[k].add(genus[lanm])
print("\npartner class | lanmodulin-type loci (genera) | other EF-hand loci (genera)")
for k in sorted(set(c)|set(cl), key=lambda k:-(c[k]+cl[k])):
    print(f"  {k:48s} {cl[k]:4d} ({len(genl[k])})   {c[k]:4d} ({len(gen[k])})")
S.to_csv("lanm_signature.tsv",sep="\t",index=False)
# sizes of loci holding lanmodulin-type proteins
sz=M.groupby("lanm").n.first(); t=sz[[g for g in sz.index if sig.get(g)]]
print("\nlocus sizes for lanmodulin-type proteins:", {"2":int((t==2).sum()),"3-4":int(((t>=3)&(t<=4)).sum()),"5-9":int(((t>=5)&(t<=9)).sum()),"10-19":int(((t>=10)&(t<=19)).sum()),"20+":int((t>=20).sum())})
