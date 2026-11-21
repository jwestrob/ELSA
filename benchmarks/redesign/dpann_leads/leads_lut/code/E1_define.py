#!/usr/bin/env python
"""Extended tier step 1 (local; rule: assay_rule_extended.json): in-scope families, annotated members, dereplication,
held-out split, definition alignment, consensus, reference-member choice, invariant and curated site positions,
experimental structure for the fold comparison.
usage: E1_define.py COLLECTION
Writes data/ext/{families.tsv, site.tsv, members_split.tsv}, FOLD_DIR/in/ext_refs0.faa (rank-0 reference members, fold before
E2_select.py) and ext_refs1.faa (rank 1, folded only where E2 reports 'no_reference_model'), ext_refs.faa (both), ref/ext_<KO>_exp.pdb (experimental chains, fold comparison only)."""
import json, re, subprocess, sys, time, urllib.parse, urllib.request
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import Align

C = sys.argv[1]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CODE = Path(__file__).resolve().parent
CFG = json.load(open(CODE / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds"
RS = CFG["remote"].replace("RD/", RD + "/")
E = H / "data/ext"; E.mkdir(parents=True, exist_ok=True)
W = FD / "ext_work"; W.mkdir(parents=True, exist_ok=True)
(FD / "in").mkdir(parents=True, exist_ok=True)
B, PY = "jwestrob@biotite.berkeley.edu", "/home/jwestrob/miniconda3/envs/elsa/bin/python"
CDHIT, MAFFT = "/Users/jacob/Documents/Software/cdhit/cd-hit", "mafft"
CHEM = set("DEHKRCSTYNQW")
AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M",
           PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")
al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)
al.end_insertion_score = al.end_deletion_score = 0


def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=90) as r:
                return r.read().decode()
        except Exception as e:
            if "404" in str(e):
                return ""
            time.sleep(2 + 3 * i)
    return ""


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


def u2t(useq, tseq):
    a = al.align(useq, tseq)[0]
    m = {}
    for (us, ue), (ts, te) in zip(*a.aligned):
        for k in range(ue - us):
            m[us + k + 1] = ts + k + 1
    return m


REF = pd.read_csv(H / "data/references.tsv", sep="\t")
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"])]
fail = REF[~REF.status.isin(["ok", "ok_but_length_incompatible"])]
fam = pop[pop.func.isin(fail.func)].groupby("func").agg(label=("proposed_label_positional_homology", lambda s: s.value_counts().index[0]),
                                                         label_name=("proposed_label_name", lambda s: s.value_counts().index[0]),
                                                         genes=("gid", "size"), unannotated=("named", lambda s: int((~s).sum())),
                                                         median_len=("seq_len", "median")).reset_index()
fam = fam.merge(fail[["func", "ecs", "status"]], on="func")
fam[["func", "label"]].to_csv(E / "ext_families.tsv", sep="\t", index=False)
subprocess.run(["rsync", "-a", str(E / "ext_families.tsv"), f"{B}:{RS}/data/"], check=True)
subprocess.run(["rsync", "-a", f"{CODE}/", f"{B}:{RS}/code/"], check=True)
print(subprocess.run(["ssh", B, f"cd {RS} && {PY} code/r6_members.py --collection {C} --families {RS}/data/ext_families.tsv"],
                     check=True, capture_output=True, text=True).stdout)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/ext_members.tsv", str(E) + "/"], check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/ext_members.faa", str(FD / "seqs") + "/"], check=True)
M = pd.read_csv(E / "ext_members.tsv", sep="\t")
SEQ = read_faa(FD / "seqs/ext_members.faa")
frows, srows, mrows, refseqs = [], [], [], {}
for r in fam.itertuples():
    f = r.func
    d = M[M.func == f].sort_values("gid")
    rec = dict(func=f, label=r.label, label_name=r.label_name, genes=r.genes, unannotated=r.unannotated, ecs=r.ecs,
               fixed_status=r.status, n_members=len(d))
    if len(d) < 15:
        frows.append({**rec, "in_scope": False, "reason": "fewer than 15 annotated members"}); continue
    fa = W / f"{f}.faa"
    fa.write_text("".join(f">{g}\n{SEQ[str(g)]}\n" for g in d.gid))
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(W / f"{f}.c70"), "-c", "0.7", "-n", "5", "-T", "4", "-M", "2000", "-d", "0"],
                   capture_output=True, check=True)
    reps = sorted(int(x) for x in read_faa(W / f"{f}.c70"))
    rec["n_reps"] = len(reps)
    if len(reps) < 15:
        frows.append({**rec, "in_scope": False, "reason": "fewer than 15 representatives at 70% identity"}); continue
    rng = np.random.default_rng(17)
    perm = [reps[i] for i in rng.permutation(len(reps))]
    nh = min(5, max(3, int(round(0.2 * len(reps)))))
    held, defn = perm[:nh], sorted(perm[nh:])
    for g in reps:
        mrows.append(dict(func=f, gid=g, set="heldout" if g in held else "definition"))
    dfa = W / f"{f}.def.faa"
    dfa.write_text("".join(f">{g}\n{SEQ[str(g)]}\n" for g in defn))
    aln = subprocess.run([MAFFT, "--auto", "--thread", "4", "--quiet", str(dfa)], capture_output=True, text=True, check=True).stdout
    A, n = {}, None
    for l in aln.splitlines():
        if l.startswith(">"):
            n = l[1:].strip(); A[n] = []
        else:
            A[n].append(l.strip())
    A = {k: "".join(v).upper() for k, v in A.items()}
    rows_ = list(A.values()); L = len(rows_[0]); N = len(rows_)
    cols = [[s[i] for s in rows_] for i in range(L)]
    cons = {}
    for i, c in enumerate(cols):
        res = [x for x in c if x != "-"]
        if len(res) >= 0.5 * N:
            cons[i] = pd.Series(res).value_counts().index[0]
    ident = {}
    for g, s in A.items():
        L_ = len(SEQ[g])
        if 100 <= L_ <= 600:
            ident[g] = sum(s[i] == a for i, a in cons.items()) / max(1, len(cons))
    if not ident:
        frows.append({**rec, "in_scope": False, "reason": "no definition member of 100-600 aa"}); continue
    top = sorted(ident, key=lambda g: (-ident[g], int(g)))[:2]
    for k, g in enumerate(top):
        refseqs[g] = SEQ[g]
    inv = []
    for i, c in enumerate(cols):
        vc = pd.Series(c).value_counts()
        a, cnt = vc.index[0], vc.iloc[0]
        if a != "-" and a in CHEM and cnt >= 0.9 * N:
            inv.append((i, a))
    for k, g in enumerate(top):
        s = A[g]; pos, mp = 0, {}
        for i, ch in enumerate(s):
            if ch != "-":
                pos += 1; mp[i] = pos
        mm = d.set_index("gid").loc[int(g)]
        lo, hi = max(1, int(mm.env_from) - 10), min(len(SEQ[g]), int(mm.env_to) + 10)
        for i, a in inv:
            if i in mp and s[i] == a and lo <= mp[i] <= hi:
                srows.append(dict(func=f, ref_member=int(g), ref_rank=k, member_pos=mp[i], residue=a, kind="invariant", note=""))
        rec[f"ref{k}"] = int(g); rec[f"ref{k}_identity"] = round(ident[g], 3); rec[f"ref{k}_region"] = f"{lo}-{hi}"
    # curated features and experimental structure (reviewed entry with an EC of the family and the label Pfam)
    ecs = [e for e in str(r.ecs).split(";") if e and "-" not in e]
    curated_acc, exp_chain = "", ""
    if ecs:
        q = "(" + " OR ".join(f"ec:{e}" for e in ecs) + f") AND reviewed:true AND xref:pfam-{r.label}"
        t = get("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            dict(query=q, fields="accession,xref_pdb,ft_act_site,ft_binding,length", format="tsv", size=200)))
        if t and len(t.splitlines()) > 1:
            U = pd.read_csv(StringIO(t), sep="\t", keep_default_na=False)
            U["n_pdb"] = U.PDB.str.count(";")
            U["has_ft"] = (U["Active site"].str.len() + U["Binding site"].str.len()) > 0
            U = U.sort_values(["has_ft", "n_pdb", "Entry"], ascending=[False, False, True])
            if U.iloc[0].has_ft:
                acc = U.iloc[0].Entry
                j = json.loads(get(f"https://rest.uniprot.org/uniprotkb/{acc}.json") or "{}")
                if j:
                    json.dump(j, open(H / f"ref/{acc}.json", "w"))
                    useq = j["sequence"]["value"]
                    for k, g in enumerate(top):
                        mp = u2t(useq, SEQ[g])
                        lo, hi = [int(x) for x in rec[f"ref{k}_region"].split("-")]
                        for ft in j.get("features", []):
                            if ft["type"] not in ("Active site", "Binding site"):
                                continue
                            s0, e0 = ft["location"]["start"]["value"], ft["location"]["end"]["value"]
                            if s0 is None or e0 is None:
                                continue
                            for p in range(s0, e0 + 1):
                                tp = mp.get(p)
                                if tp and lo <= tp <= hi and SEQ[g][tp - 1] == useq[p - 1]:
                                    srows.append(dict(func=f, ref_member=int(g), ref_rank=k, member_pos=tp, residue=useq[p - 1],
                                                      kind="curated", note=f"{acc} {ft['type']} {useq[p - 1]}{p} " +
                                                      (ft.get("ligand", {}).get("name", "") if ft["type"] == "Binding site" else "")))
                    curated_acc = acc
            S_ = U[U.n_pdb > 0]
            for acc in list(S_.Entry)[:3]:
                bs = get(f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{acc}")
                ip = get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{acc}")
                if not bs or not ip:
                    continue
                span = [f2 for e in json.loads(ip).get("results", []) if e["metadata"]["accession"] == r.label
                        for loc in e["proteins"][0]["entry_protein_locations"] for f2 in (loc["fragments"][0]["start"], loc["fragments"][-1]["end"])]
                best = [b for b in json.loads(bs).get(acc, []) if b.get("experimental_method", "").lower().startswith(("x-ray", "electron"))]
                if not span or not best:
                    continue
                best.sort(key=lambda b: (b.get("resolution") or 99, -b.get("coverage", 0)))
                b = best[0]; pdb, ch = b["pdb_id"].lower(), b["chain_id"]
                raw = get(f"https://files.rcsb.org/download/{pdb}.pdb")
                lines = [l for l in raw.splitlines() if l.startswith("ATOM") and l[21] == ch and l[16] in " A"] if raw else []
                if not lines:
                    continue
                lo_u, hi_u = min(span) - 10, max(span) + 10
                # chain residue numbers approximately follow UniProt numbering after the SIFTS offset
                off = b["unp_start"] - b["start"] if "start" in b else 0
                keep = [l for l in lines if lo_u <= int(l[22:26]) + off <= hi_u]
                if len(keep) < 100:
                    continue
                (H / "ref").mkdir(exist_ok=True)
                open(H / f"ref/ext_{f}_exp.pdb", "w").write("\n".join(keep) + "\nEND\n")
                exp_chain = f"{pdb.upper()}_{ch} ({acc})"
                break
    rec["curated_record"] = curated_acc; rec["exp_chain"] = exp_chain
    n0 = sum(1 for x in srows if x["func"] == f and x["ref_rank"] == 0)
    rec["n_site_ref0"] = n0
    rec["n_curated_ref0"] = sum(1 for x in srows if x["func"] == f and x["ref_rank"] == 0 and x["kind"] == "curated")
    rec["in_scope"] = n0 >= 3
    rec["reason"] = "" if n0 >= 3 else "fewer than 3 site positions"
    frows.append(rec)
    print(f, r.label_name, len(d), "members", len(reps), "reps", n0, "site", curated_acc, exp_chain, flush=True)
F = pd.DataFrame(frows)
F.to_csv(E / "families.tsv", sep="\t", index=False)
S = pd.DataFrame(srows).drop_duplicates(["func", "ref_rank", "member_pos", "kind"])
S.to_csv(E / "site.tsv", sep="\t", index=False)
pd.DataFrame(mrows).to_csv(E / "members_split.tsv", sep="\t", index=False)
keep = set(F[F.in_scope == True].func)  # noqa: E712
ids = [g for f in keep for g in [F.set_index("func").loc[f, "ref0"], F.set_index("func").loc[f, "ref1"]] if pd.notna(g)]
(FD / "in/ext_refs.faa").write_text("".join(f">{int(g)}\n{SEQ[str(int(g))]}\n" for g in ids))
Fi = F[F.in_scope == True].set_index("func")  # noqa: E712
for k in (0, 1):   # fold ext_refs0.faa first; fold ext_refs1.faa only for families E2_select reports as 'no_reference_model'
    gk = [int(g) for g in Fi[f"ref{k}"] if pd.notna(g)] if f"ref{k}" in Fi.columns else []
    (FD / f"in/ext_refs{k}.faa").write_text("".join(f">{g}\n{SEQ[str(g)]}\n" for g in gk))
print(F.in_scope.value_counts(dropna=False).to_string())
print(F.groupby(["in_scope", "reason"]).size().to_string())
print("reference members to fold:", len(ids), "| unannotated genes in scope:", int(F[F.in_scope == True].unannotated.sum()))  # noqa: E712
