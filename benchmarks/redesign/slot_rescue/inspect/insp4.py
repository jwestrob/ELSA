# Check B (ESMFold2 structures, US-align): labeled gene vs its annotated partner (true) and vs a different-label partner (null).
import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
u=pd.read_csv("verify/B/checkB_pairs.tsv",sep="\t",dtype={"kind":str},keep_default_na=False,na_values=[""]); u["kind"]=u.kind.str.lower().fillna("null"); print(u.kind.value_counts().to_dict()); print(u.shape); print(list(u.columns)); print(u.status.value_counts().to_dict())
k="kind"
sup=pd.read_parquet("inspect/support.parquet")
u=u.merge(sup,on="protein_id",how="left")
ok=u[u.status=="ok"].copy()
ok["tm_max"]=ok[["TM1","TM2"]].max(axis=1); ok["tm_min"]=ok[["TM1","TM2"]].min(axis=1)
ok["vbin"]=pd.cut(ok.vfrac,[0,0.05,0.2,0.5,1.01],labels=["<5%","5-20%","20-50%",">=50%"],right=False)
ok["hi"]=ok.vote_share>0.8
def tab(g): return pd.Series({"pairs":len(g),"TMmax>=0.5_%":round(100*(g.tm_max>=0.5).mean(),1),"TMmin>=0.5_%":round(100*(g.tm_min>=0.5).mean(),1),"median_TMmax":round(g.tm_max.median(),2)})
for name,sub in (("ALL PAIRS",ok),("BOTH pLDDT>=70",ok[ok.both_plddt_ge70])):
    print(name); print(sub.groupby(k).apply(tab).to_string())
    print(sub.groupby([k,"hi"]).apply(tab).to_string())
    print(sub.groupby([k,"vbin"],observed=True).apply(tab).to_string()); print()
t=ok[ok[k]=="true"]
print("pLDDT>=70 both:",int(t.both_plddt_ge70.sum()),"of",len(t),"true pairs")
lab=t.groupby("label_name").apply(lambda g: pd.Series({"pairs":len(g),"TMmax>=0.5":int((g.tm_max>=0.5).sum()),"med_TM":round(g.tm_max.median(),2),"med_vfrac":round(g.vfrac.median(),3)}))
print(lab.sort_values("pairs",ascending=False).head(25).to_string())
bad=t[(t.tm_max<0.4)&t.both_plddt_ge70][["protein_id","label_name","length","partner_length","tm_max","vote_share","vfrac","amp"]]
print("true pairs, both confident, TM<0.4:",len(bad)); print(bad.sort_values("tm_max").head(25).to_string(index=False))
