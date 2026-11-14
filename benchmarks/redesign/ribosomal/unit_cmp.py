import json, numpy as np, pandas as pd
RD="/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
A=RD+"/ribosomal/arms/"
BASE="results_dpann_v5__dpann__mmseqs__seq__e3c50s75__w3"
HYB="results_dpann_emb1__dpann__hybrid__esmpp_small__genomecentred__e3c50cpmr0p05__w3"
CTX="results_dpann_ctx__dpann__ctx__esmpp_small__matched_w5_simctx_sup_r10__w3"
rng=np.random.default_rng(17)
b=pd.read_parquet(A+BASE+"/ribosomal_runs.parquet").sort_values("run_id")
h=pd.read_parquet(A+HYB+"/ribosomal_runs.parquet").sort_values("run_id")
c=pd.read_parquet(A+CTX+"/ribosomal_runs.parquet").sort_values("run_id")
n=len(b); boot=rng.integers(0,n,size=(2000,n))
def cmp(x,cx,y,cy,label):
    d=x[cx+":resolved"].to_numpy().astype(int)-y[cy+":resolved"].to_numpy().astype(int)
    s=d[boot].sum(1); print(f"{label}: {x[cx+':resolved'].sum()} vs {y[cy+':resolved'].sum()}  diff {d.sum():+d} [{np.quantile(s,.025):.0f}, {np.quantile(s,.975):.0f}]")
cmp(h,"unit@0.5",b,"ladder_conserved@80","hybrid r0p05 unit@0.5 vs base conserved@80")
cmp(h,"unit@0.5",b,"unit@0.5","hybrid r0p05 unit@0.5 vs base unit@0.5")
cmp(h,"unit@0.5",b,"conserved@2","hybrid r0p05 unit@0.5 vs base conserved@2")
cmp(c,"ladder_conserved@80",b,"ladder_conserved@80","ctx w5 simctx sup r10 conserved@80 vs base conserved@80")
# base-arm per-run oracle vs fixed @80
for arm,df in (("hyb",h),):
    print(arm,"unit frag median", df["unit@0.5:fragments"].median(), "best module", df["unit@0.5:module"].value_counts().head(3).to_dict())
for nm in (BASE,HYB):
    j=json.load(open(A+nm+"/ribosomal_check.json"))
    u=j["controls"]["g2_m16"]["unit@0.5"]["random"]; t=j["controls"]["g2_m16"]["unit@0.5"]["transplant"]
    r=j["results"]["g2_m16"]["unit@0.5"]
    print(nm[:40], "unit@0.5 obs rec/res", r["recovered"], r["resolved"], "purity", round(r["module_purity"],3), r["module_top3"], "parv", r["parvarchaeaceae"], r["parvarchaeaceae_resolved"], "| random rec/res", round(u["recovered_frac"],4), round(u["resolved_frac"],4), "| transplant", round(t["recovered_frac"],4), round(t["resolved_frac"],4))
    print("   by-phylum not stored for unit; genomes/genera/phyla", r["genomes"], r["genera"], r["phyla"])
# where do base units break inside runs: flanking marker families at internal unit boundaries
runs=pd.read_parquet(RD+"/ribosomal/gold/runs.parquet"); runs=runs[(runs.run_set=="g2")&(runs.n_marked>=16)]
mk=pd.read_parquet(RD+"/ribosomal/gold/marked.parquet")
contigs=pd.read_parquet(RD+"/ws_dpann/package/universe/contigs.parquet")
marker=pd.Series(mk.marker.to_numpy(), index=mk.gid.to_numpy())
for nm in (BASE,HYB):
    l=pd.read_parquet(RD+"/engine/"+nm.split("__",1)[0].replace("results_dpann_","results_dpann_")+"/"+nm.split("__",1)[1]+"/loci_all_genomes.parquet")
    l=l[l.kind=="unit"].merge(contigs[["genome","contig","first_gid"]],on=["genome","contig"])
    l["a"]=l.first_gid+l.start; l["b"]=l.first_gid+l.end
    l=l.sort_values("a")
    a=l.a.to_numpy(); bb=l.b.to_numpy()
    cuts=[]
    for r in runs.itertuples():
        i0=np.searchsorted(bb, r.gid_start); 
        j=i0
        while j < len(a) and a[j] <= r.gid_end:
            # unit end strictly inside run -> cut between bb[j] and bb[j]+1 (units share endpoints: cut at gene bb[j])
            if r.gid_start <= bb[j] < r.gid_end:
                cuts.append((r.run_id, int(bb[j])))
            j+=1
    cdf=pd.DataFrame(cuts,columns=["run_id","g"])
    cdf["left"]=marker.reindex(cdf.g.to_numpy()).fillna("-").to_numpy(); cdf["right"]=marker.reindex((cdf.g+1).to_numpy()).fillna("-").to_numpy()
    print(nm[:40], "internal unit cuts", len(cdf), "runs with cut", cdf.run_id.nunique())
    print("   top cut pairs (unit end gene | next gene):", (cdf.left+" | "+cdf.right).value_counts().head(8).to_dict())
