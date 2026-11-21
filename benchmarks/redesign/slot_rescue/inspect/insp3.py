# Check A (weak profile hits) broken down by label support = votes_for / supporting_genomes and by label amplification.
import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
a=pd.read_parquet("verify/A/checkA_per_protein.parquet")
s=pd.read_parquet("pass/dpann_slot_rescue_pass.parquet")[["protein_id","supporting_genomes","voters_annotated","proposed_label_name","alignment_lists_same_label"]]
lx=pd.read_parquet("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad/label_stats.parquet")[["lab","amp","annotated"]]
d=a.merge(s,on="protein_id").merge(lx,left_on="label",right_on="lab",how="left")
d["vfrac"]=d.votes_for/d.supporting_genomes
d["vbin"]=pd.cut(d.vfrac,[0,0.05,0.2,0.5,1.01],labels=["<5%","5-20%","20-50%",">=50%"],right=False)
d["abin"]=pd.cut(d.amp,[0,0.2,1,10,1e9],labels=["<0.2x","0.2-1x","1-10x",">=10x"],right=False)
d["other_only"]=(~d.no_hit_any)&(~d.proposed_any)
def tab(g):
    n=len(g); hit=(~g.no_hit_any).sum()
    return pd.Series({"n":n,"proposed_top_%":round(100*g.proposed_top.mean(),1),"proposed_any_%":round(100*g.proposed_any.mean(),1),
        "other_label_only_%":round(100*g.other_only.mean(),1),"no_hit_%":round(100*g.no_hit_any.mean(),1),
        "top_given_any_hit_%":round(100*g.proposed_top.sum()/max(hit,1),1),"null_any_%":round(100*g.null_any.mean(),2)})
print("ALL"); print(tab(d).to_string()); print()
for b in ["vbin","abin"]:
    print(b); print(d.groupby(b,observed=True).apply(tab).to_string()); print()
print("vbin x share>0.8"); print(d.groupby(["vbin","share_gt_0.8"],observed=True).apply(tab).to_string()); print()
# no own KO only (own-KO rows are near-trivially right)
print("rows without own KO, by vbin"); print(d[d.own_ko_agrees=="no_ko"].groupby("vbin",observed=True).apply(tab).to_string()); print()
# candidate rule: support >= x
for x in (0.05,0.1,0.2,0.3,0.5):
    g=d[d.vfrac>=x]; print(f"support>={x}: n={len(g)} top_given_hit={100*g.proposed_top.sum()/(~g.no_hit_any).sum():.1f}% proposed_any={100*g.proposed_any.mean():.1f}%")
g=d[d.vfrac<0.05]; print(f"support<0.05: n={len(g)} top_given_hit={100*g.proposed_top.sum()/(~g.no_hit_any).sum():.1f}%")
d[["protein_id","vfrac","amp","annotated"]].to_parquet("inspect/support.parquet")
