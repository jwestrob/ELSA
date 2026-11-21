#!/usr/bin/env python
"""Leads pipeline step 4a (local), copied from dpann_leads/invisible/code/s02_references.py with only the paths changed
(collection argument; population = data/population.parquet, tiers T2/T2b/rest, both novelty strata). usage: L3_references.py C
Original docstring: one curated, characterized reference per function (family) of the novel population.
Fixed before any fold was requested.

Query (UniProt REST, public metadata only; no DPANN data leaves the machine): reviewed entries with an EC number of the
function, an experimental 3D structure and at least one Active site feature. Eligible when the entry carries the label
Pfam (label kind pfam) or a Pfam of PFSET (s01; also the fallback when no entry carries the label Pfam itself).
Family region of the reference: span of the entry's InterPro-Pfam matches to the label Pfam (else the matched PFSET Pfams), +-10 residues
(whole entry when no Pfam coordinates). The region must hold >= 1 Active site and >= 1 Binding site feature.
When some entry carries the label Pfam, only such entries are eligible (PFSET is the fallback otherwise).
A family is length-compatible when its median candidate length is >= 0.6x the reference family region.
Reference choice order: (1) carries the label Pfam itself; (2) entry length within 0.6-1.7x the family's median candidate
length; (3) archaea > bacteria > eukaryota; (4) closest length; (5) more PDB entries.
Experimental chain: PDBe best_structures for the accession; X-ray or EM; chain covers every in-region Active site
position and every one of them is resolved; best resolution, then coverage. UniProt positions are mapped to chain residues
by global alignment of the UniProt sequence to the chain's ATOM sequence (Biopython PairwiseAligner). The chain is
trimmed to the family region (ref/<PDB>_<CHAIN>_fam.pdb), renumbered from 1 in chain order.
Writes ref/<ACC>.json, ref/<PDB>_<CHAIN>_fam.pdb, data/references.tsv, data/reference_candidates.tsv,
data/ref_features.tsv (one row per curated residue: family, feature type, UniProt position, residue, ligand, index in the
trimmed chain or 'unresolved')."""
import json, sys, time, urllib.parse, urllib.request
from io import StringIO
from pathlib import Path
import pandas as pd
from Bio import Align

C = sys.argv[1]
_R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA"
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace("R/", _R + "/"))
REF = H / "ref"
REF.mkdir(exist_ok=True)
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M",
          PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M", SEC="U", PYL="O")
nv = pd.read_parquet(H / "data/population.parquet")
nv = nv[nv.tier.isin(["T2", "T2b", "rest"])]          # both strata (already named and unannotated) define the families
nv["protein_id"] = nv.protein_id.astype(str)
S = pd.read_csv(H / "data/sets_per_function.tsv", sep="\t", keep_default_na=False)
aligner = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5)
aligner.target_end_gap_score = 0
aligner.query_end_gap_score = 0


def get(url, tries=4):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=90) as r:
                return r.read().decode()
        except Exception as e:
            err = e
            if "404" in str(e):
                break
            time.sleep(2 + 3 * i)
    print("FAILED", url, err, file=sys.stderr)
    return None


def chain_atoms(pdb, ch):
    """protein ATOM lines of chain ch, first model, altloc blank/A; PDB format, else mmCIF."""
    raw = get(f"https://files.rcsb.org/download/{pdb}.pdb")
    out = []
    if raw:
        for l in raw.splitlines():
            if l.startswith("ENDMDL"):
                break
            if l.startswith(("ATOM", "HETATM")) and l[21] == ch and l[16] in " A" and l[17:20] in AA:
                out.append(("ATOM  " + l[6:16] + " " + l[17:]).ljust(80))
        if out:
            return out
    cif = get(f"https://files.rcsb.org/download/{pdb}.cif")
    if not cif:
        return out
    hdr, n = [], 0
    for l in cif.splitlines():
        if l.startswith("_atom_site."):
            hdr.append(l.split(".", 1)[1].strip()); continue
        if hdr and l.startswith(("ATOM", "HETATM")):
            d = dict(zip(hdr, l.split()))
            if d.get("auth_asym_id") != ch or d.get("pdbx_PDB_model_num", "1") != "1" or d.get("label_alt_id", ".") not in (".", "A") \
                    or d["auth_comp_id"] not in AA:
                continue
            n += 1
            nm = d["auth_atom_id"].strip('"')
            ic = d.get("pdbx_PDB_ins_code", "?")
            out.append("ATOM  %5d %-4s %3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s" % (
                n % 100000, nm if len(nm) == 4 else " " + nm, d["auth_comp_id"], "A", int(d["auth_seq_id"]) % 10000,
                " " if ic in ("?", ".") else ic, float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"]), 1.0,
                float(d["B_iso_or_equiv"]), d["type_symbol"]))
    return out


def residues(lines):
    res, seen = [], {}
    for l in lines:
        key = l[22:27]
        if key not in seen:
            seen[key] = len(res)
            res.append([key, AA.get(l[17:20], "X"), []])
        res[seen[key]][2].append(l)
    return res


ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None   # srvp: rerun a subset, merged into the tables
FIELDS = "accession,id,protein_name,organism_name,length,ec,xref_pfam,xref_pdb,ft_act_site,ft_binding,lineage"
cand_rows, ref_rows, feat_rows = [], [], []
fams = nv.groupby("func").agg(genes=("protein_id", "size"), L=("length", "median"),
                              label=("proposed_label_positional_homology", lambda s: s.value_counts().index[0]),
                              label_name=("proposed_label_name", lambda s: s.value_counts().index[0]),
                              kind=("proposed_label_kind", "first")).reset_index()
if ONLY is not None:
    fams = fams[fams.func.isin(ONLY)]
for fr in fams.itertuples():
    srow = S[(S.func == fr.func) & (S.label == fr.label)].iloc[0]
    ecs = [x for e in srow.ecset.split() for x in e.split(";") if x]     # srvp fix: individual ECs
    pfset = set(srow.pfset.split())
    out = dict(func=fr.func, label=fr.label, label_name=fr.label_name, genes=fr.genes, median_len=fr.L, ecs=" ".join(ecs))
    if not ecs:
        ref_rows.append({**out, "status": "no_complete_EC"}); continue
    q = "(" + " OR ".join(f"ec:{e}" for e in ecs) + ") AND reviewed:true AND structure_3d:true AND ft_act_site:*"
    txt = get("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(dict(query=q, fields=FIELDS, format="tsv", size=500)))
    if not txt or len(txt.splitlines()) < 2:
        ref_rows.append({**out, "status": "no_reviewed_entry_with_EC_active_site_structure"}); continue
    U = pd.read_csv(StringIO(txt), sep="\t", keep_default_na=False)
    U["pfams"] = U.Pfam.str.strip(";").str.split(";")
    U["has_label"] = U.pfams.apply(lambda p: fr.label in p)
    U["pf_match"] = U.pfams.apply(lambda p: sorted(set(p) & (pfset | {fr.label})))
    U["domain"] = U["Taxonomic lineage"].apply(lambda s: "Archaea" if "Archaea" in s else "Bacteria" if "Bacteria" in s else "Eukaryota")
    U["n_pdb"] = U.PDB.str.count(";")
    for r in U.itertuples():
        cand_rows.append(dict(func=fr.func, label=fr.label, acc=r.Entry, length=r.Length, pfam=r.Pfam, has_label=r.has_label,
                              pf_match=",".join(r.pf_match), domain=r.domain, n_pdb=r.n_pdb))
    # PFSET is a fallback only when no entry carries the label Pfam itself (label kind pfam)
    K = U[U.has_label].copy() if (fr.kind == "pfam" and U.has_label.any()) else U[U.pf_match.str.len() > 0].copy()
    if K.empty:
        ref_rows.append({**out, "status": f"no_entry_in_label_family ({len(U)} with EC+structure+active site)"}); continue
    K["dom_rank"] = K.domain.map({"Archaea": 0, "Bacteria": 1, "Eukaryota": 2})
    K["len_ok"] = ((K.Length >= 0.6 * fr.L) & (K.Length <= 1.7 * fr.L)).astype(int)
    K["len_diff"] = (K.Length - fr.L).abs()
    K = K.sort_values(["has_label", "len_ok", "dom_rank", "len_diff", "n_pdb"], ascending=[False, False, True, True, False])
    chosen, why = None, []
    for r in K.head(8).itertuples():
        j = get(f"https://rest.uniprot.org/uniprotkb/{r.Entry}.json")
        ip = get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{r.Entry}")
        if not j:
            continue
        j = json.loads(j)
        useq = j["sequence"]["value"]
        span = []
        if ip:
            for e in json.loads(ip).get("results", []):
                if e["metadata"]["accession"] in ([fr.label] if r.has_label else r.pf_match):
                    for loc in e["proteins"][0]["entry_protein_locations"]:
                        for f in loc["fragments"]:
                            span += [f["start"], f["end"]]
        lo, hi = (max(1, min(span) - 10), min(len(useq), max(span) + 10)) if span else (1, len(useq))
        feats = []
        for f in j.get("features", []):
            if f["type"] not in ("Active site", "Binding site"):
                continue
            s, e = f["location"]["start"]["value"], f["location"]["end"]["value"]
            if s is None or e is None:
                continue
            lig = f.get("ligand", {}).get("name", "") if f["type"] == "Binding site" else f.get("description", "")
            for p in range(s, e + 1):
                if lo <= p <= hi:
                    feats.append((f["type"], p, useq[p - 1], lig))
        act = sorted({p for t, p, a, l in feats if t == "Active site"})
        if not act or not any(t == "Binding site" for t, *_ in feats):
            why.append(f"{r.Entry}:no_active_or_binding_in_region"); continue
        bs = get(f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{r.Entry}")
        if not bs:
            why.append(f"{r.Entry}:no_best_structures"); continue
        best = [b for b in json.loads(bs).get(r.Entry, []) if b.get("experimental_method", "").lower().startswith(("x-ray", "electron"))
                and all(b["unp_start"] <= a <= b["unp_end"] for a in act)]
        best.sort(key=lambda b: (b.get("resolution") or 99, -b.get("coverage", 0)))
        for b in best[:4]:
            pdb, ch = b["pdb_id"].lower(), b["chain_id"]
            lines = chain_atoms(pdb, ch)
            res = residues(lines)
            if len(res) < 30:
                continue
            cseq = "".join(x[1] for x in res)
            aln = aligner.align(useq, cseq)[0]
            u2c = {}
            for (us, ue), (cs, ce) in zip(*aln.aligned):
                for k in range(ue - us):
                    u2c[us + k + 1] = cs + k          # UniProt 1-based -> chain residue index (0-based)
            ident = sum(useq[u - 1] == cseq[c] for u, c in u2c.items()) / max(1, len(u2c))
            if ident < 0.9 or any(a not in u2c or cseq[u2c[a]] != useq[a - 1] for a in act):
                continue
            keep = sorted(c for u, c in u2c.items() if lo <= u <= hi)
            cmin, cmax = keep[0], keep[-1]
            trimmed, newidx = [], {}
            for k, ci in enumerate(range(cmin, cmax + 1)):
                newidx[ci] = k
                for l in res[ci][2]:
                    trimmed.append(l[:22] + "%4d " % (k + 1) + l[27:])
            chosen = (r, j, b, pdb, ch, trimmed, u2c, newidx, cseq, feats, lo, hi, ident)
            break
        if chosen:
            break
        why.append(f"{r.Entry}:no_chain")
    if not chosen:
        ref_rows.append({**out, "status": "no_experimental_chain_with_active_site", "notes": ";".join(why)}); continue
    r, j, b, pdb, ch, trimmed, u2c, newidx, cseq, feats, lo, hi, ident = chosen
    json.dump(j, open(REF / f"{r.Entry}.json", "w"))
    name = f"{pdb.upper()}_{ch}_fam"
    open(REF / f"{name}.pdb", "w").write("\n".join(trimmed) + "\nEND\n")
    for t, p, a, lig in feats:
        ci = u2c.get(p)
        idx = newidx.get(ci) if ci is not None else None
        feat_rows.append(dict(func=fr.func, ref_acc=r.Entry, ref_chain=name, feature=t, uniprot_pos=p, residue=a, ligand=lig,
                              chain_index=idx if idx is not None else "unresolved",
                              chain_residue=cseq[ci] if ci is not None else ""))
    na = sum(1 for t, *_ in feats if t == "Active site"); nb = sum(1 for t, *_ in feats if t == "Binding site")
    ref_rows.append({**out, "status": "ok", "ref_acc": r.Entry, "ref_entry": r._2, "ref_organism": r.Organism, "ref_domain": r.domain,
                     "ref_len": r.Length, "ref_has_label_pfam": r.has_label, "pf_match": ",".join(r.pf_match), "region": f"{lo}-{hi}",
                     "pdb_chain": f"{pdb.upper()}_{ch}", "ref_file": f"ref/{name}.pdb", "ref_fam_len": len(newidx),
                     "resolution": b.get("resolution"), "method": b.get("experimental_method"), "chain_uniprot_identity": round(ident, 3),
                     "n_active_residues": na, "n_binding_residues": nb, "notes": ";".join(why)})
    print(fr.func, fr.label_name, r.Entry, name, len(newidx), na, nb, flush=True)
CR, R, FR = pd.DataFrame(cand_rows), pd.DataFrame(ref_rows), pd.DataFrame(feat_rows)
if ONLY is not None:      # merge: keep the earlier rows of every function outside the subset
    keep = lambda f, new: pd.concat([pd.read_csv(H / f, sep="\t")[lambda d: ~d.func.isin(ONLY)], new], ignore_index=True)
    CR, FR = keep("data/reference_candidates.tsv", CR), keep("data/ref_features.tsv", FR)
    R = pd.concat([pd.read_csv(H / "data/references.tsv", sep="\t")[lambda d: ~d.func.isin(ONLY)].drop(columns=["length_compatible"]), R],
                  ignore_index=True)
CR.to_csv(H / "data/reference_candidates.tsv", sep="\t", index=False)
# a gene shorter than ~0.5x the reference family region cannot reach TM 0.5 normalized by the reference
R["length_compatible"] = R.median_len >= 0.6 * R.ref_fam_len
R.loc[(R.status == "ok") & ~R.length_compatible, "status"] = "ok_but_length_incompatible"
R.to_csv(H / "data/references.tsv", sep="\t", index=False)
FR.to_csv(H / "data/ref_features.tsv", sep="\t", index=False)
print(R.status.str.split(" ").str[0].value_counts())
print(R[R.status == "ok"].genes.sum(), "genes in", (R.status == "ok").sum(), "families with a reference")
