#!/usr/bin/env python3
"""slot_rescue step 1 (Mac; read-only on the Sharur DPANN DB and the Sharur study folders): truth table + label tables.

Nothing here reads an ELSA output; no call is computed. Writes
  data/truth.tsv        one row per (set, truth protein): protein_id, set, unit (lead or family), study call (R sets),
                        study evidence, flank protein IDs, verification tag
  data/units.tsv        one row per lead / detection-limited family: R sets, broad class, evidence summary
  data/ko_proxy.tsv     KO -> Pfam accessions carried by > 50% of the KO's proteins (whole DB)
  data/labels.tsv       every Pfam accession / KO in the DB with the Sharur predicates used as broad classes
  data/annot.parquet    protein_id -> Pfam accessions, KOs (sources pfam, kofam, kegg)
  data/truth_meta.json  counts and reproduction checks against the study's stored numbers

    cd ~/Documents/Sandbox/Sharur/Sharur && python3 <this file>
"""
import collections
import csv
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SH = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur")
DB = SH / "data/dpann/sharur.duckdb"
RS = Path("/Users/jacob/Documents/Berkeley/Sharur/replacement_screen_2026-11-11")
RP = SH / "data/dpann/analysis/ribosomal_protein_absence_2026-10-12"
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
csv.field_size_limit(10 ** 9)
os.chdir(SH)                                   # rp_cohort opens data/dpann/sharur.duckdb relative to the repo root
sys.path.insert(0, str(RP))
sys.path.insert(0, str(SH))
from rp_cohort import COHORT  # noqa: E402

con = duckdb.connect(str(DB), read_only=True)
con.execute("PRAGMA threads=4; PRAGMA memory_limit='8GB'")
meta = {}

# ------------------------------------------------------------------ label tables (whole DB)
ko_proxy = con.execute(r"""
  WITH k AS (SELECT DISTINCT protein_id, accession AS ko FROM annotations
             WHERE source IN ('kofam','kegg') AND regexp_matches(accession, '^K\d{5}$')),
       f AS (SELECT DISTINCT protein_id, accession AS pf FROM annotations WHERE source = 'pfam' AND regexp_matches(accession, '^PF\d{5}$')),
       nk AS (SELECT ko, COUNT(*) AS n FROM k GROUP BY 1)
  SELECT k.ko, f.pf, COUNT(*) AS c, any_value(nk.n) AS n FROM k JOIN f USING (protein_id) JOIN nk USING (ko)
  GROUP BY 1, 2 HAVING COUNT(*) > 0.5 * any_value(nk.n) ORDER BY 1, 2""").df()
ko_proxy.to_csv(OUT / "ko_proxy.tsv", sep="\t", index=False)
proxy_of = ko_proxy.groupby("ko").pf.apply(set).to_dict()
kos_of_pf = collections.defaultdict(set)
for k, pfs in proxy_of.items():
    for p in pfs:
        kos_of_pf[p].add(k)

from sharur.predicates.mappings.kegg_map import KEGG_TO_PREDICATES  # noqa: E402
from sharur.predicates.mappings.pfam_map import PFAM_TO_PREDICATES  # noqa: E402
PRED = ["ribosomal_protein", "sec_pathway", "rna_polymerase", "atp_synthesis", "trna_modification", "membrane",
        "cytochrome_bd_oxidase"]
labs = con.execute(r"""SELECT DISTINCT CASE WHEN source = 'pfam' THEN 'pfam' ELSE 'ko' END AS kind, accession, name
                       FROM annotations WHERE (source = 'pfam' AND regexp_matches(accession, '^PF\d{5}$'))
                          OR (source IN ('kofam','kegg') AND regexp_matches(accession, '^K\d{5}$'))""").df()
labs = labs.drop_duplicates("accession").reset_index(drop=True)
rows = []
for kind, acc, name in labs.itertuples(index=False):
    pr = set(PFAM_TO_PREDICATES.get(acc, [])) | set(PFAM_TO_PREDICATES.get(name, [])) if kind == "pfam" else set(KEGG_TO_PREDICATES.get(acc, []))
    rows.append(dict(label=acc, kind=kind, name=name, **{p: int(p in pr) for p in PRED}))
L = pd.DataFrame(rows)
L.to_csv(OUT / "labels.tsv", sep="\t", index=False)
pf_name = dict(zip(L.label, L.name))
meta["labels"] = dict(pfam=int((L.kind == "pfam").sum()), ko=int((L.kind == "ko").sum()),
                      ribosomal_protein=int(L.ribosomal_protein.sum()), ko_with_proxy=len(proxy_of))

ann = con.execute(r"""SELECT protein_id,
       list(DISTINCT accession) FILTER (WHERE source = 'pfam' AND regexp_matches(accession, '^PF\d{5}$')) AS pfam,
       list(DISTINCT accession) FILTER (WHERE source IN ('kofam','kegg') AND regexp_matches(accession, '^K\d{5}$')) AS ko
       FROM annotations WHERE source IN ('pfam','kofam','kegg') GROUP BY 1""").df()
ann["pfam"] = [sorted(v) if isinstance(v, (list, np.ndarray)) else [] for v in ann.pfam]
ann["ko"] = [sorted(v) if isinstance(v, (list, np.ndarray)) else [] for v in ann.ko]
ann = ann[(ann.pfam.str.len() > 0) | (ann.ko.str.len() > 0)].reset_index(drop=True)
ann.to_parquet(OUT / "annot.parquet", index=False)
meta["annot_proteins"] = len(ann)
has_pfam = set(ann.protein_id[ann.pfam.str.len() > 0])
pf_of = dict(zip(ann.protein_id, ann.pfam))
ko_of = dict(zip(ann.protein_id, ann.ko))


def r_ko_for(pfs, kos):
    out = set(kos)
    for p in pfs:
        out |= kos_of_pf.get(p, set())
    return out


def r_pf_for(pfs, kos):
    out = set(pfs)
    for k in kos:
        out |= proxy_of.get(k, set())
    return out


# ------------------------------------------------------------------ the 30 dark leads (52 rows)
D = pd.read_csv(RS / "tables/dark_newcomers.tsv", sep="\t", keep_default_na=False, na_values=[""])
I = pd.read_csv(RS / "work/dark_instances.tsv", sep="\t", keep_default_na=False, na_values=[""])
assert len(D) == len(I) == 52
for c in ("dark_lead", "X_members", "Y_members", "offset_in_Y_orientation"):
    assert (D[c].astype(str).values == I[c].astype(str).values).all(), c
D["dark_instances"] = I.dark_instances.values

BROAD = {"ribosomal": "ribosomal_protein", 3: "sec_pathway", 6: "rna_polymerase", 9: "atp_synthesis", 24: "atp_synthesis",
         10: "trna_modification", 11: "membrane", 25: None, 29: "cytochrome_bd_oxidase"}
units, lead_rows = [], collections.defaultdict(list)
RIBO = dict(zip(L.label, L.ribosomal_protein))
for r in D.itertuples(index=False):
    lead_rows[int(r.dark_lead)].append(r)
for lead, rs in sorted(lead_rows.items()):
    pf, ko = set(), set()
    for r in rs:
        mem = str(r.X_members).split("+")
        pf |= {a for a in mem if a.startswith("PF")}
        ko |= {a for a in mem if a.startswith("K")}
        pf |= {a for a in str(r.X_proxy_pfam or "").split("+") if a.startswith("PF")}
        ko |= {a for a in str(r.X_dominant_KO or "").split("+") if a.startswith("K")}
    Rpf, Rko = r_pf_for(pf, ko), r_ko_for(pf, ko)
    ev = [(max(float(r.dark_hit_by_X_profile_relaxed or 0), float(r.dark_with_X_dominant_KO or 0)), float(r.dark_to_X_length_ratio), r) for r in rs]
    best = max(ev, key=lambda t: t[0])
    is_x = bool(rs[0].dark_lead_is_X)
    ribo = any(RIBO.get(a, 0) for a in pf | ko)
    broad = "ribosomal_protein" if ribo else BROAD.get(lead) if is_x else None
    weak = is_x and (best[0] < 0.40 or not (0.8 <= best[1] <= 1.25))
    units.append(dict(unit=f"L{lead:02d}", set="lead_X" if is_x else "lead_negative", study_call="+".join(sorted({str(r.X_names) for r in rs})),
                      flank_names="+".join(dict.fromkeys(str(r.Y_names) for r in rs)),
                      R_pfam=";".join(sorted(Rpf)), R_ko=";".join(sorted(Rko)), broad_class=broad or "",
                      evidence=(f"dark proteins in X's slot in X-absent species reps; best row: {best[0]:.3f} of a 60-protein sample "
                                f"score X's Pfam profile (relaxed, >=60% coverage, >=0.4 GA) or carry X's dominant KO; length ratio "
                                f"{best[1]:.2f}; dark_is_X rule >= 0.20"),
                      best_evidence_fraction=round(best[0], 3), best_length_ratio=best[1],
                      verification=("UNVERIFIED" if weak else "study_call") if is_x else "study_negative",
                      rows=len(rs), instances_listed=int(sum(len(str(r.dark_instances).split(";")) for r in rs))))
U = pd.DataFrame(units)

inst = collections.defaultdict(list)            # protein -> [(lead, Y members, offset)]
for r in D.itertuples(index=False):
    for p in str(r.dark_instances).split(";"):
        inst[p].append((int(r.dark_lead), str(r.Y_members), int(r.offset_in_Y_orientation)))
pids = sorted(inst)
con.execute("CREATE OR REPLACE TEMP TABLE q AS SELECT unnest(?::VARCHAR[]) AS protein_id", [pids])
P = con.execute("""SELECT p.protein_id, p.bin_id, p.contig_id, p.gene_index, p.strand, p.sequence_length
                   FROM proteins p JOIN q USING (protein_id)""").df()
assert len(P) == len(pids), (len(P), len(pids))
C = con.execute("""SELECT p.protein_id, p.contig_id, p.gene_index, p.strand FROM proteins p
                   WHERE p.contig_id IN (SELECT DISTINCT contig_id FROM proteins JOIN q USING (protein_id))""").df()
at = {(c, g): (pid, s) for pid, c, g, s in C.itertuples(index=False)}
Pi = P.set_index("protein_id")
lu = U.set_index("unit")
trows = []
conflict = 0
for p in pids:
    c, g = Pi.at[p, "contig_id"], int(Pi.at[p, "gene_index"])
    flanks, leads = [], sorted({l for l, _, _ in inst[p]})
    for lead, ym, rel in inst[p]:
        ymem = set(ym.split("+"))
        hit = []
        for gy, sy in ((g - rel, "+"), (g + rel, "-")):
            x = at.get((c, gy))
            if x and x[1] == sy and set(pf_of.get(x[0], [])) & ymem:
                hit.append(x[0])
        flanks += hit
    flanks = sorted(set(flanks))
    calls = {(lu.at[f"L{l:02d}", "R_pfam"], lu.at[f"L{l:02d}", "R_ko"]) for l in leads}
    if len(calls) > 1:
        conflict += 1
    for l in leads:
        u = lu.loc[f"L{l:02d}"]
        trows.append(dict(protein_id=p, set=u.set, unit=f"L{l:02d}", study_call=u.study_call, R_pfam=u.R_pfam, R_ko=u.R_ko,
                          broad_class=u.broad_class, verification=u.verification, flanks=";".join(flanks), n_flanks_found=len(flanks),
                          length=int(Pi.at[p, "sequence_length"]), bin_id=Pi.at[p, "bin_id"], has_pfam=int(p in has_pfam),
                          carries_R_ko=int(bool(set(ko_of.get(p, [])) & set(u.R_ko.split(";")))) if u.R_ko else 0,
                          multi_lead=len(leads), conflicting_calls=int(len(calls) > 1),
                          evidence="lead-level (see units.tsv)", order=""))
meta["leads"] = dict(proteins_listed=len(pids), protein_lead_pairs=len(trows), proteins_in_conflicting_leads=conflict,
                     proteins_carrying_pfam=int(sum(p in has_pfam for p in pids)))

# ------------------------------------------------------------------ detection-limited slot sets (rp_report_facts logic, cohort)
def genes(order):
    rows = con.execute(f"""
      WITH g AS (SELECT bin_id FROM bins WHERE {COHORT} AND taxonomy LIKE '%o__{order};%'),
      pr AS (SELECT protein_id, bin_id, contig_id, gene_index, sequence_length, strand FROM proteins WHERE bin_id IN (SELECT bin_id FROM g)),
      ann AS (SELECT protein_id, string_agg(DISTINCT name, ',') FILTER (WHERE LOWER(source)='pfam') AS nm,
                     bool_or(LOWER(source) IN ('kofam','kegg')) AS ko FROM annotations WHERE protein_id IN (SELECT protein_id FROM pr) GROUP BY 1)
      SELECT bin_id, contig_id, gene_index, sequence_length, strand, COALESCE(ann.nm,''), COALESCE(ann.ko, false), protein_id
      FROM pr LEFT JOIN ann USING (protein_id) ORDER BY 1,2,3""").fetchall()
    by = collections.defaultdict(list)
    for r in rows:
        by[(r[0], r[1])].append(r)
    return by, len({r[0] for r in rows})


def after_us3(order):
    by, ng = genes(order)
    c, out = collections.Counter(), []
    for gs in by.values():
        for i, g in enumerate(gs):
            if "Ribosomal_S3_C" in g[5].split(","):
                j = i + (1 if g[4] == "+" else -1)
                if not 0 <= j < len(gs):
                    continue
                x = gs[j]
                if "Ribosomal_L29" in x[5].split(","):
                    c["Ribosomal_L29"] += 1
                elif not x[5] and not x[6] and x[3] <= 80 and x[4] == g[4]:
                    c["unannotated_le80_same_strand"] += 1
                    out.append((x[7], [g[7]], x[0], x[3]))
                else:
                    c["other"] += 1
    return dict(genomes=ng, us3_with_downstream_gene=sum(c.values()), **c), out


def ul15_slot(order, maxgap=4):
    by, ng = genes(order)
    c, out = collections.Counter(), []
    for gs in by.values():
        for i, g in enumerate(gs):
            if "Ribosomal_L30" not in g[5].split(","):
                continue
            for j in list(range(i + 1, min(i + maxgap + 1, len(gs)))) + list(range(max(0, i - maxgap), i)):
                if "SecY" in gs[j][5].split(","):
                    lo, hi = sorted((i, j))
                    between = gs[lo + 1:hi]
                    if any("Ribosomal_L27A" in x[5].split(",") for x in between):
                        c["Ribosomal_L27A"] += 1
                    elif between and all(not x[5] and not x[6] for x in between):
                        c["unannotated"] += 1
                        out += [(x[7], [g[7], gs[j][7]], x[0], x[3]) for x in between]
                    else:
                        c["other"] += 1
                    break
    return dict(genomes=ng, pairs=sum(c.values()), **c), out


facts = json.load(open(RP / "rp_report_facts.json"))
dl = []
for order in ("Parvarchaeales", "Pacearchaeales"):
    cnt, out = after_us3(order)
    stored = facts["gene_order"]["uL29_after_uS3"][order]
    meta.setdefault("reproduce", {})[f"uL29_{order}"] = dict(recomputed=cnt, stored={k: v for k, v in stored.items() if not k.startswith("orf_")},
                                                            equal=all(cnt.get(k) == v for k, v in stored.items() if not k.startswith("orf_")))
    dl += [dict(protein_id=p, set="DL_slot", unit="uL29", order=order, flanks=";".join(f), bin_id=b, length=ln, evidence="slot after uS3, <=80 aa, same strand, no Pfam, no KO") for p, f, b, ln in out]
for order in ("Parvarchaeales", "Iainarchaeales"):
    cnt, out = ul15_slot(order)
    stored = facts["gene_order"]["uL15_between_uL30_SecY"][order]
    meta["reproduce"][f"uL15_{order}"] = dict(recomputed=cnt, stored={k: v for k, v in stored.items() if "length" not in k},
                                            equal=all(cnt.get(k) == v for k, v in stored.items() if "length" not in k))
    dl += [dict(protein_id=p, set="DL_slot", unit="uL15", order=order, flanks=";".join(f), bin_id=b, length=ln, evidence="only gene(s) between uL30 and SecY, no Pfam, no KO") for p, f, b, ln in out]

# eL39 slot: first L31e / dsDNA_bind pair 1-2 genes apart per genome, the gene between
rep = json.load(open(RP / "rp_report_data.json"))
el39_row = next(r for r in rep["rows"] if r["protein"] == "eL39")
dpann_el39 = {l.split()[0] for l in open(RP / "search_outputs/custom_profiles/custom.tbl")
              if not l.startswith("#") and l.split()[2] == "dpann_eL39" and float(l.split()[4]) <= 1e-5}
slot_rows = con.execute(f"""
    WITH coh AS (SELECT bin_id, split_part(taxonomy,';',2) || '/' || split_part(taxonomy,';',4) AS ord FROM bins WHERE {COHORT}),
    fl AS (SELECT DISTINCT p.bin_id, p.contig_id, p.gene_index, a.name, p.protein_id FROM annotations a JOIN proteins p USING (protein_id)
           JOIN coh USING (bin_id) WHERE LOWER(a.source)='pfam' AND a.name IN ('Ribosomal_L31e','dsDNA_bind')),
    pr AS (SELECT x.bin_id, x.contig_id, LEAST(x.gene_index, y.gene_index) AS lo, ABS(x.gene_index - y.gene_index) AS d,
                  x.protein_id AS fx, y.protein_id AS fy
           FROM fl x JOIN fl y ON x.bin_id = y.bin_id AND x.contig_id = y.contig_id AND x.name = 'Ribosomal_L31e'
           AND y.name = 'dsDNA_bind' AND ABS(x.gene_index - y.gene_index) BETWEEN 1 AND 2),
    one AS (SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY bin_id ORDER BY d, contig_id, lo) rn FROM pr) WHERE rn = 1),
    ann AS (SELECT protein_id, bool_or(LOWER(source)='pfam' AND split_part(accession,'.',1)='PF00832') AS l39, COUNT(*) AS n
            FROM annotations GROUP BY 1)
    SELECT replace(replace(coh.ord, 'p__', ''), 'o__', ''), one.bin_id, one.d, m.protein_id, m.sequence_length,
           COALESCE(ann.l39, false), COALESCE(ann.n, 0), one.fx, one.fy
    FROM one JOIN coh USING (bin_id)
    LEFT JOIN proteins m ON one.d = 2 AND m.bin_id = one.bin_id AND m.contig_id = one.contig_id AND m.gene_index = one.lo + 1
    LEFT JOIN ann ON ann.protein_id = m.protein_id""").fetchall()
slot = collections.defaultdict(collections.Counter)
el39 = []
for ordkey, b, d, pid, ln, l39, n_ann, fx, fy in slot_rows:
    c = slot[ordkey]
    c["genomes_with_flanks"] += 1
    if d == 1:
        c["closed"] += 1
    elif l39:
        c["eL39_stored"] += 1
    elif pid in dpann_el39:
        c["eL39_dpann_profile"] += 1
    elif n_ann == 0:
        c["single_unannotated_orf"] += 1
        el39.append((ordkey, pid, [fx, fy], b, ln))
    else:
        c["other_annotated"] += 1
st = facts["el39_slot"]["by_order"]
meta["reproduce"]["eL39_slot_by_order"] = dict(equal=all(dict(slot.get(o, {})) == v for o, v in st.items()),
                                               recomputed={o: dict(v) for o, v in slot.items()})
neg = [o["key"] for o in rep["orders"] if el39_row["by_order"][o["key"]] <= 0.10]
flagged = [o["key"] for o in rep["orders"] if el39_row["by_order"][o["key"]] < 0.70 and st.get(o["key"], {}).get("genomes_with_flanks", 0) >= 5
           and st[o["key"]].get("single_unannotated_orf", 0) / st[o["key"]]["genomes_with_flanks"] >= 0.5]
faa_ids = {l[1:].split()[0] for l in open(RP / "search_outputs/el39_slot/slot_orfs.faa") if l.startswith(">")}
ev_slot = {l.split()[0]: float(l.split()[4]) for l in open(RP / "search_outputs/el39_slot/slot_orfs.tbl") if not l.startswith("#")}
neg_ids = {pid for o, pid, _, _, _ in el39 if o in neg}
meta["reproduce"]["eL39_negative_order_orfs"] = dict(recomputed=len(neg_ids), stored_faa=len(faa_ids), equal=neg_ids == faa_ids,
                                                    flagged_orders=flagged, negative_orders=neg)
for o, pid, f, b, ln in el39:
    if o in flagged or o in neg:
        e = ev_slot.get(pid)
        dl.append(dict(protein_id=pid, set="DL_slot", unit="eL39", order=o.split("/")[1], flanks=";".join(f), bin_id=b, length=ln,
                       evidence=("single unannotated ORF in the eL31-dsDNA_bind slot; " +
                                 ("negative order; DPANN-eL39 profile " + (f"E={e:.2g} (weak hit, E<=1)" if e is not None and e <= 1 else "no hit at E<=1")
                                  if o in neg else "flagged order (eL39 < 70%), slot rule only"))))

# DPANN-specific profile rescues (rp_custom_tabulate rule): best hit E <= 1e-5, cohort genome, protein lacks the stored family Pfam
fam_of = {"uL29": "PF00831", "eS17": "PF00833", "eS27": "PF01667", "eL31": "PF01198", "eL34": "PF01199", "eL37": "PF01907",
          "eL33": "PF01247", "eL39": "PF00832", "eL20": "PF01775", "eS26": "PF01283", "eL24": "PF01246", "eS31": "PF01599",
          "eL42": "PF00935", "uS15": "PF00312"}
best = {}
for line in open(RP / "search_outputs/custom_profiles/custom.tbl"):
    if line.startswith("#"):
        continue
    x = line.split()
    if float(x[4]) <= 1e-5 and float(x[5]) > best.get(x[0], ("", -1, 1))[1]:
        best[x[0]] = (x[2].removeprefix("dpann_"), float(x[5]), float(x[4]))
coh = {b for (b,) in con.execute(f"SELECT bin_id FROM bins WHERE {COHORT}").fetchall()}
con.execute("CREATE OR REPLACE TEMP TABLE h AS SELECT unnest(?::VARCHAR[]) AS protein_id", [list(best)])
hp = con.execute("SELECT p.protein_id, p.bin_id, p.sequence_length FROM proteins p JOIN h USING (protein_id)").fetchall()
n_prof = collections.Counter()
n_prof_all = collections.Counter()
for pid, b, ln in hp:
    prot, sc, e = best[pid]
    if b not in coh or fam_of[prot] in pf_of.get(pid, []):
        continue
    n_prof_all[prot] += 1
    if pid in has_pfam:                       # dark only (no Pfam annotation at all), as the leads' dark definition
        continue
    n_prof[prot] += 1
    dl.append(dict(protein_id=pid, set="DL_profile", unit=prot, order="", flanks="", bin_id=b, length=ln,
                   evidence=f"DPANN-specific {prot} profile best hit E={e:.2g}, {sc:.1f} bits; no stored Pfam"))
meta["DL_profile"] = dict(lacking_family_pfam=dict(n_prof_all), dark_kept=dict(n_prof))

DLt = pd.DataFrame(dl)
for prot in sorted(set(DLt.unit)):
    pf = {fam_of.get(prot) or {"uL15": "PF00828"}[prot]}
    Rpf, Rko = r_pf_for(pf, set()), r_ko_for(pf, set())
    for s in sorted(set(DLt.set[DLt.unit == prot])):
        units.append(dict(unit=prot, set=s, study_call=prot, flank_names={"uL29": "Ribosomal_S3_C", "uL15": "Ribosomal_L30+SecY",
                          "eL39": "Ribosomal_L31e+dsDNA_bind"}.get(prot, "") if s == "DL_slot" else "",
                          R_pfam=";".join(sorted(Rpf)), R_ko=";".join(sorted(Rko)), broad_class="ribosomal_protein",
                          evidence=("gene-order slot + length (study: detection-limited, present)" if s == "DL_slot"
                                    else "DPANN-specific profile, E <= 1e-5"),
                          best_evidence_fraction=np.nan, best_length_ratio=np.nan,
                          verification="study_call_positional" if s == "DL_slot" else "study_call",
                          rows=0, instances_listed=int(((DLt.unit == prot) & (DLt.set == s)).sum())))
U = pd.DataFrame(units)
Ui = U.drop_duplicates(["unit", "set"]).set_index(["unit", "set"])
for r in dl:
    u = Ui.loc[(r["unit"], r["set"])]
    p = r["protein_id"]
    trows.append(dict(protein_id=p, set=r["set"], unit=r["unit"], study_call=r["unit"], R_pfam=u.R_pfam, R_ko=u.R_ko,
                      broad_class=u.broad_class, verification=u.verification, flanks=r["flanks"],
                      n_flanks_found=len([f for f in r["flanks"].split(";") if f]), length=int(r["length"]), bin_id=r["bin_id"],
                      has_pfam=int(p in has_pfam), carries_R_ko=int(bool(set(ko_of.get(p, [])) & set(u.R_ko.split(";")))) if u.R_ko else 0,
                      multi_lead=1, conflicting_calls=0, evidence=r["evidence"], order=r["order"]))
T = pd.DataFrame(trows)
T.to_csv(OUT / "truth.tsv", sep="\t", index=False)
U.to_csv(OUT / "units.tsv", sep="\t", index=False)
meta["truth_rows_by_set"] = T.groupby("set").size().to_dict()
meta["truth_unique_proteins_by_set"] = T.groupby("set").protein_id.nunique().to_dict()
meta["truth_unique_proteins"] = int(T.protein_id.nunique())
meta["DL_by_unit"] = T[T.set.str.startswith("DL")].groupby(["set", "unit"]).size().astype(int).to_dict()
meta["DL_by_unit"] = {f"{a}|{b}": v for (a, b), v in meta["DL_by_unit"].items()}
meta["lead_rows_flank_found"] = int((T[T.set.str.startswith("lead")].n_flanks_found > 0).sum())
json.dump(meta, open(OUT / "truth_meta.json", "w"), indent=1, default=str)
print(json.dumps(meta, indent=1, default=str)[:6000])
print(U[["unit", "set", "study_call", "verification", "best_evidence_fraction", "best_length_ratio", "broad_class", "instances_listed"]].to_string())
