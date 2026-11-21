#!/usr/bin/env python
"""Leads pipeline step 5 (biotite login node; one small Aksha search within the login-node thread cap).
usage: r5_context.py --collection C --leads REMOTE/data/leads_in.tsv [--threads 4]
leads_in.tsv columns: gid, func, koset (space list), pfset (space list).
Per passing gene: taxonomy (genomes table), completeness, neighbors +-5 genes on the contig with their Pfam names,
other proteins of the genome with a PFSET Pfam (collection Pfam table, GA), genomes of the same order carrying a PFSET
Pfam (prevalence of the family in the clade), and a picked-profile KOfam search (Aksha
--hmm_in <KOSET profiles copied from the installed KOFAM store> --cut_ga) of every protein of the lead genomes: hits for the
KO and for KOs sharing its EC (non-homologous enzymes of the same reaction included when KEGG files them under the EC).
Writes REMOTE/data/leads_context.tsv and REMOTE/data/leads_neighbors.tsv."""
import argparse, json, os, shutil, subprocess
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
HERE = os.path.dirname(os.path.abspath(__file__))
KOFAM = "/home/jwestrob/.config/Astra/KOFAM"
AKSHA = "/home/jwestrob/.local/bin/aksha"
ap = argparse.ArgumentParser()
ap.add_argument("--collection", required=True); ap.add_argument("--leads", required=True); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
cfg = {k: (v.replace("RD/", RD + "/") if isinstance(v, str) else v)
       for k, v in json.load(open(os.path.join(HERE, "collections_leads.json")))[a.collection].items()}
R = cfg["remote"]
W = f"{R}/context"
os.makedirs(W, exist_ok=True)
L = pd.read_csv(a.leads, sep="\t", dtype={"koset": str, "pfset": str})
genes = pd.read_parquet(cfg["genes"], columns=["gid", "genome", "contig", "position_index", "strand", "length"])
G = pd.read_parquet(cfg["genomes"]).set_index("genome")
gi = genes.set_index("gid")
L["genome"] = L.gid.map(gi.genome)
lead_genomes = sorted(set(L.genome))
gg = genes[genes.genome.isin(lead_genomes)]
want = set(gg.gid.astype(int))
seq, keep, name, buf = {}, False, None, []
for line in open(cfg["fasta"]):
    if line.startswith(">"):
        if keep:
            seq[name] = "".join(buf).rstrip("*")
        name = line[1:].split()[0]; keep = int(name) in want; buf = []
    elif keep:
        buf.append(line.strip())
if keep:
    seq[name] = "".join(buf).rstrip("*")
with open(f"{W}/lead_genomes.faa", "w") as fh:
    for g in sorted(want):
        fh.write(f">{g}\n{seq[str(g)]}\n")
kos = sorted({k for s in L.koset for k in str(s).split()})
with open(f"{W}/picked.hmm", "w") as fh:
    for k in kos:
        p = f"{KOFAM}/{k}.hmm"
        if os.path.exists(p):
            t = open(p).read(); fh.write(t if t.endswith("\n") else t + "\n")
shutil.rmtree(f"{W}/kofam_picked", ignore_errors=True)
subprocess.run([AKSHA, "search", "--hmm_in", f"{W}/picked.hmm", "--cut_ga", "--prot_in", f"{W}/lead_genomes.faa", "--outdir",
                f"{W}/kofam_picked", "--threads", str(a.threads)], check=True, capture_output=True)
hf = [f for f in os.listdir(f"{W}/kofam_picked") if f.endswith("_hits_df.tsv")]
KH = pd.concat([pd.read_csv(f"{W}/kofam_picked/{f}", sep="\t", dtype={"sequence_id": str}) for f in hf], ignore_index=True) if hf else pd.DataFrame(columns=["sequence_id", "hmm_name"])
KH["gid"] = KH.sequence_id.astype(int)
KH["ko"] = KH.hmm_name.astype(str).str.extract(r"(K\d{5})")[0]
KH["genome"] = KH.gid.map(gi.genome)
pfall = pd.read_parquet(cfg["pfam"], columns=["gid", "accession", "name"])
pfall["pf"] = pfall.accession.str.split(".").str[0]
pf = pfall[pfall.gid.isin(want)].copy()
pf["genome"] = pf.gid.map(gi.genome)
ocol = "ord" if "ord" in G.columns else "order"
genome_of_gid = genes.set_index("gid").genome
rows, nb = [], []
for r in L.itertuples():
    ks, ps = set(str(r.koset).split()), set(str(r.pfset).split())
    me = gi.loc[r.gid]
    kh = KH[(KH.genome == r.genome) & KH.ko.isin(ks)]
    ph = pf[(pf.genome == r.genome) & pf.pf.isin(ps)]
    own_k = sorted(set(kh[kh.gid == r.gid].ko))
    other_k = [f"{x.gid}:{x.ko}" for x in kh[kh.gid != r.gid].drop_duplicates(["gid", "ko"]).itertuples()]
    other_p = [f"{x.gid}:{x.name}" for x in ph[ph.gid != r.gid].drop_duplicates(["gid", "pf"]).itertuples()]
    ctg = genes[(genes.contig == me.contig) & ((genes.position_index - me.position_index).abs() <= 5) & (genes.gid != r.gid)]
    txt = []
    for x in ctg.sort_values("position_index").itertuples():
        nm = ",".join(sorted(set(pf[pf.gid == x.gid].name)))[:50] or "-"
        txt.append(f"{x.position_index - me.position_index:+d}:{nm}")
        nb.append(dict(lead=r.gid, offset=x.position_index - me.position_index, neighbor=x.gid, pfam=nm))
    t = G.loc[r.genome] if r.genome in G.index else None
    og = set(G.index[G[ocol] == t[ocol]]) if t is not None else set()
    carriers_pf = set(genome_of_gid.reindex(pfall[pfall.pf.isin(ps)].gid).dropna())
    order_prev = f"{len(og & carriers_pf)}/{len(og)}"
    rows.append(dict(gid=r.gid, genome=r.genome, phylum=t["phylum"] if t is not None else "", tclass=t["class"] if t is not None else "",
                     order=t["ord"] if t is not None and "ord" in t else "", family=t["family"] if t is not None else "",
                     genus=t["genus"] if t is not None else "", species=t["species"] if t is not None and "species" in t else "",
                     completeness=t["completeness"] if t is not None else "", own_kofam_cutga=";".join(own_k),
                     other_kofam_cutga=";".join(other_k)[:300], other_pfam_ga=";".join(other_p)[:300],
                     function_absent_otherwise=not other_k and not other_p, order_genomes=len(og),
                     order_genomes_with_family_pfam=order_prev, neighbors=" ".join(txt)))
pd.DataFrame(rows).to_csv(f"{R}/data/leads_context.tsv", sep="\t", index=False)
pd.DataFrame(nb).to_csv(f"{R}/data/leads_neighbors.tsv", sep="\t", index=False)
print(len(rows), "leads;", len(lead_genomes), "genomes;", len(KH), "picked KOfam hits")
