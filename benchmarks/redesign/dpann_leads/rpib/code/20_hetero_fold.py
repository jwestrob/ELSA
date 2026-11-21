#!/usr/bin/env python3
"""Frame-challenge test (orthogonal hypothesis: the long-form protein is a partner-binding module of its co-directional
+1 gene, COG2522 HTH regulator, 93% of loci). Hetero co-folds through the Biohub API (ESMFold2 all-atom), 1:1:
4 long-form genes + their cognate +1 COG2522 gene, and the same 4 long-form genes + a same-genome control protein of
similar length (Pfam Ribosomal_L7Ae, unrelated). Reuses the request and PDB-writing functions of
Sharur/scripts/fold_biohub.py (imported read-only). Writes folds/hetero/*.pdb, *.json, data/hetero_pairs.tsv."""
import sys, json, os, time, urllib.request, duckdb, pandas as pd
sys.path.insert(0, "/Users/jacob/Documents/Sandbox/Sharur/Sharur/scripts")
import fold_biohub as FB
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"; O=R+"/folds/hetero"; os.makedirs(O,exist_ok=True)
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
N=pd.read_csv(D+"/neighbors_offsets.tsv",sep="\t")
LF=["DAHWRM010000003.1_10","DAHXEZ010000002.1_75","DAHWRX010000014.1_34","MEM3181264.1"]
rows=[]
for q in LF:
    nb=N[(N.fid==q)&(N.o==1)&(N.rel_strand=="same")&(N.cdd=="COG2522")].protein_id.iloc[0]
    bin_=con.execute("SELECT bin_id FROM d.proteins WHERE protein_id=?",[q]).fetchone()[0]
    ctl=con.execute("""SELECT pr.protein_id FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE pr.bin_id=? AND a.source='pfam' AND a.name='Ribosomal_L7Ae' ORDER BY a.evalue LIMIT 1""",[bin_]).fetchone()
    rows.append(dict(lf=q,partner=nb,kind="cognate_plus1_COG2522")); 
    if ctl: rows.append(dict(lf=q,partner=ctl[0],kind="control_Ribosomal_L7Ae_same_genome"))
P=pd.DataFrame(rows)
seq=con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))",[list(set(P.lf)|set(P.partner))]).df().set_index("protein_id").sequence.str.rstrip("*")
P["name"]=P.lf+"__"+P.partner; P["lf_len"]=P.lf.map(seq).str.len(); P["partner_len"]=P.partner.map(seq).str.len()
H=FB.headers(); model="esmfold2-2026-05"
payload=[{"model":model,"all_atom_input":{"sequences":[{"type":"protein","id":"A","sequence":seq[r.lf]},{"type":"protein","id":"B","sequence":seq[r.partner]}]},
          "include_pae":False,"include_pair_chains_iptm":True} for r in P.itertuples()]
batch=FB.post("/submit",{"endpoint":"fold_all_atom","payload":payload},H)["data"]["batch_id"]; print("batch",batch,len(payload),"items",flush=True)
pending={i:f"{batch}-{i:05d}" for i in range(len(payload))}; t0=time.time(); out={}
while pending and time.time()-t0<1800:
    time.sleep(15)
    for i,tid in list(pending.items()):
        dd=FB.post("/status",{"task_id":tid},H)["data"]; st=dd.get("status")
        if st in ("failed","cancelled"): print("FAILED",P.name[i],dd.get("error")); del pending[i]; continue
        if st!="done": continue
        j=json.loads(urllib.request.urlopen(dd["response"],timeout=120).read())
        pdb,atoms=FB.to_pdb_complex(j["complex"],{},("A","B")); open(f"{O}/{P.name[i]}.pdb","w").write(pdb)
        json.dump({k:j.get(k) for k in ("model","ptm","interface_ptm","pair_chains_iptm")},open(f"{O}/{P.name[i]}.json","w"))
        out[i]=(j.get("ptm"),j.get("interface_ptm")); print("done",P.name[i],out[i],flush=True); del pending[i]
P["ptm"]=[out.get(i,(None,None))[0] for i in range(len(P))]; P["iptm"]=[out.get(i,(None,None))[1] for i in range(len(P))]
P.to_csv(D+"/hetero_pairs.tsv",sep="\t",index=False); print(P.drop(columns="name").to_string(index=False))
