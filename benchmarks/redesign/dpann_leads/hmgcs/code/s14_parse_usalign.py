#!/usr/bin/env python3
"""Parse the US-align text outputs: TM-scores per pair; and for every model x reference pair, the model residue that
superposes onto each curated catalytic residue of the reference (UniProt ACT_SITE numbering mapped onto the PDB chain
by global alignment of the chain sequence to the UniProt sequence), with the CA-CA distance flag (':' = d < 5 A)."""
import os, re, glob, pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
AA1 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")
# reference chain -> (UniProt accession, catalytic residues in UniProt numbering, class)
REF = {"1YSL_A": ("Q9FD71", {"E": 79, "C": 111, "H": 233}, "HMGS (E. faecalis MvaS)"),
       "2P8U_A": ("Q01581", {"E": 95, "C": 129, "H": 264}, "HMGS (human HMGCS1)"),
       "6ESQ_I": ("A0A384E143", {"E": 82, "C": 114, "H": 238}, "HMGS (archaeal, M. thermolithotrophicus)"),
       "4YXT_A": ("P40830", {"E": 82, "C": 114, "H": 250}, "HMG-ACP synthase (PksG)"),
       "1HNJ_A": ("P0A6R0", {"C": 112, "H": 244, "N": 274}, "KAS III (FabH)"),
       "1DM3_A": ("P07097", {"C": 89, "H": 348, "C2": 378}, "thiolase (PhaA)"),
       "2GFW_A": ("P0AAI5", {"C": 164, "H": 304, "H2": 341}, "KAS II (FabF)"),
       "1BI5_A": ("P30074", {"C": 164, "H": 303, "N": 336}, "type III PKS (CHS2)")}
al = Align.PairwiseAligner(); al.mode = "global"; al.substitution_matrix = substitution_matrices.load("BLOSUM62")
al.open_gap_score = -10; al.extend_gap_score = -0.5
def chain_seq(pdb):
    res = []; seen = set()
    for l in open(pdb):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            key = (l[21], l[22:27])
            if key not in seen:
                seen.add(key); res.append((AA1.get(l[17:20], "X"), l[22:27].strip()))
    return res
def uni(acc):
    return "".join(x.strip() for x in open(f"{W}/ref/{acc}.faa") if not x.startswith(">"))
def uni2idx(chain, useq):          # UniProt residue number -> index in chain residue list
    cs = "".join(c for c, _ in chain)
    a = al.align(cs, useq)[0]
    m = {}
    for (cs0, ce0), (us0, ue0) in zip(*a.aligned):
        for k in range(ce0 - cs0):
            m[us0 + k + 1] = cs0 + k
    return m
def parse(fn):
    t = open(fn).read()
    if "TM-score=" not in t:
        return None
    L = re.findall(r"Length of Structure_[12]:\s+(\d+)", t)
    al_ = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([\d.]+), Seq_ID=n_identical/n_aligned=\s*([\d.]+)", t)
    tms = re.findall(r"TM-score= ([\d.]+)", t)
    lines = t.rstrip().splitlines()
    i = [k for k, l in enumerate(lines) if l.startswith('(":"')][0]
    s1, mk, s2 = lines[i + 1], lines[i + 2], lines[i + 3]
    return dict(L1=int(L[0]), L2=int(L[1]), Lali=int(al_.group(1)), RMSD=float(al_.group(2)), SeqID=float(al_.group(3)),
                TM1=float(tms[0]), TM2=float(tms[1]), s1=s1, mk=mk, s2=s2)
sets = pd.read_csv(f"{W}/fold_in/fold_sets.tsv", sep="\t").set_index("protein_id")["set"].to_dict()
rows, cat = [], []
refchain = {k: chain_seq(f"{W}/refchains/{k}.pdb") for k in REF}
for fn in sorted(glob.glob(f"{W}/us_raw/*.txt")):
    tag = os.path.basename(fn)[:-4]
    a, b = tag.split("__")
    p = parse(fn)
    if p is None:
        continue
    rows.append(dict(model=a, target=b, set=sets.get(a, a), **{k: p[k] for k in ("L1", "L2", "Lali", "RMSD", "SeqID", "TM1", "TM2")}))
    if b in REF and not a.startswith("cofold"):
        acc, sites, cls = REF[b]
        u2i = uni2idx(refchain[b], uni(acc))
        mres = chain_seq(f"{W}/folds/single/{a}.pdb")
        # walk the alignment: column -> (model idx, ref idx)
        mi = ri = -1; ref2model = {}
        for x, f, y in zip(p["s1"], p["mk"], p["s2"]):
            if x != "-": mi += 1
            if y != "-": ri += 1
            if x != "-" and y != "-":
                ref2model[ri] = (mi, f)
        rec = dict(model=a, set=sets.get(a), target=b, ref_class=cls, TM_ref=p["TM2"])
        for nm, r in sites.items():
            ci = u2i.get(r)
            refaa = refchain[b][ci][0] if ci is not None else "?"
            if ci is not None and ci in ref2model:
                mi_, f = ref2model[ci]
                rec[f"{nm}{r}"] = f"{refaa}->{mres[mi_][0]}{mi_ + 1}{'' if f == ':' else '(far)'}"
            else:
                rec[f"{nm}{r}"] = f"{refaa}->gap"
        cat.append(rec)
T = pd.DataFrame(rows)
T.to_csv(f"{W}/usalign_pairs.tsv", sep="\t", index=False)
C = pd.DataFrame(cat)
C.to_csv(f"{W}/usalign_catalytic_mapping.tsv", sep="\t", index=False)
print("# TM-score normalized by the experimental reference (TM2), model x reference")
piv = T[T.target.isin(REF)].pivot_table(index=["set", "model"], columns="target", values="TM2").round(3)
print(piv[list(REF)].to_string())
print("\n# catalytic residues superposed (reference residue -> model residue; '(far)' = CA-CA >= 5 A)")
for b in REF:
    sub = C[C.target == b].dropna(axis=1, how="all")
    print(f"\n## {b} {REF[b][2]}")
    print(sub.drop(columns=["target", "ref_class"]).to_string(index=False))
print("\n# model-vs-model TM (normalized by the second model) for lead/controls vs canonical DPANN HMGCS CAISET010000016.1_25 and thiolase")
mm = T[~T.target.isin(REF) & ~T.model.str.startswith("cofold")]
print(mm[(mm.target.isin(["CAISET010000016.1_25", "WP_203134943.1", "WP_203134463.1", "WP_226803576.1"])) | (mm.model.isin(["CAISET010000016.1_25"]))][["model", "target", "TM1", "TM2", "RMSD", "SeqID", "Lali"]].to_string(index=False))
print("\n# cofold dimers vs 1YSL A+B")
print(T[T.model.str.startswith("cofold")].to_string(index=False))
