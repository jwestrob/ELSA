"""Task 1, step 2 (local, HTTP lookups by public accession only): genomic loci of the public A/B/fused homologs.

1. NCBI taxonomy lineage of every target taxid (efetch db=taxonomy).
2. Scope: Bacteria and non-DPANN Archaea by NCBI lineage, whose EMBL protein id is absent from the DPANN gene
   universe (work/genes_dark.parquet protein_id). Eukaryotic assemblies and metagenome-only records are listed
   separately (not in the 40).
3. One locus per nucleotide record and target cluster (targets within 5 CDS of each other on one record are one
   locus). Priority: fused single-ORF homologs, then non-DPANN archaea, then bacteria by best evidence (UniRef50
   E-value, then HHblits P >= 95 count). Up to 40 loci with a resolvable record.
4. Per record: efetch db=nuccore rettype=fasta_cds_aa (every CDS translation with its location), parsed into
   work/nb/cds.tsv and work/nb/contig_proteins.faa (all CDS of the record, ids '<locus>|<protein_id>').
Writes work/nb/loci.tsv, work/nb/cds.tsv, work/nb/contig_proteins.faa, work/nb/scope.tsv.
"""
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
W = HERE / "work"
NB = W / "nb"
RAW = W / "nb_raw"
GENES = HERE.parent / "work" / "genes_dark.parquet"
UA = {"User-Agent": "generic-research-client/1.0"}
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?tool=generic_client"
DPANN = ("DPANN group", "Nanoarchaeota", "Woesearchaeota", "Pacearchaeota", "Aenigmarchaeota", "Micrarchaeota",
         "Parvarchaeota", "Diapherotrites", "Iainarchaeota", "Nanohaloarchaeota", "Nanohalarchaeota", "Altiarchaeota",
         "Altarchaeota", "Huberarchaeota", "Undinarchaeota", "Nanobdellota", "Aenigmatarchaeota", "Mamarchaeota")


def get(url, tries=4):
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"{url}: {err}")


def parse_loc(loc):
    strand = -1 if loc.startswith("complement(") else 1
    s = loc.replace("complement(", "").replace("join(", "").replace(")", "")
    nums = [int(x) for x in re.findall(r"\d+", s)]
    return min(nums), max(nums), strand, ("<" in s), (">" in s)


def main():
    T = pd.read_csv(NB / "targets.tsv", sep="\t")
    tx = pd.read_csv(NB / "acc_taxid.tsv", sep="\t", header=None, names=["acc", "taxid2"])
    T = T.merge(tx, on="acc", how="left")
    T["taxid"] = T.taxid.fillna(T.taxid2).astype("Int64")
    ids = ",".join(str(x) for x in T.taxid.dropna().unique())
    xml = get(f"{EU}&db=taxonomy&id={ids}&retmode=xml")
    (RAW / "taxonomy.xml").write_text(xml)
    lin = {}
    for t in ET.fromstring(xml).findall("Taxon"):
        lin[int(t.findtext("TaxId"))] = t.findtext("Lineage", "") + "; " + t.findtext("ScientificName", "")
    T["ncbi_lineage"] = T.taxid.map(lambda x: lin.get(int(x), "") if pd.notna(x) else "")
    g = pd.read_parquet(GENES, columns=["protein_id"])
    dp = set(g.protein_id)
    T["prot_ids"] = T.embl.fillna("").map(lambda s: [x.split("|")[1] for x in s.split(";") if x and x.split("|")[1] != "-"])
    T["in_dpann_data"] = T.prot_ids.map(lambda L: any(p in dp for p in L))

    def cls(r):
        L = r.ncbi_lineage
        if r.in_dpann_data:
            return "DPANN(in dataset)"
        if "Eukaryota" in L:
            return "Eukaryota"
        if "Archaea" in L:
            return "Archaea_DPANN" if any(t in L for t in DPANN) else "Archaea_nonDPANN"
        if "Bacteria" in L:
            return "Bacteria"
        return "metagenome/unclassified"

    T["scope_cls"] = T.apply(cls, axis=1)
    T["nuc"] = T.embl.fillna("").map(lambda s: s.split(";")[0].split("|")[0] if s else "")
    T["pid"] = T.prot_ids.map(lambda L: L[0] if L else "")
    T["fused"] = T.acc.isin(["A0A661IN52", "A0A2W7RED9"])
    T.drop(columns=["prot_ids"]).to_csv(NB / "scope.tsv", sep="\t", index=False)
    print(T.scope_cls.value_counts().to_string())
    S = T[T.scope_cls.isin(["Bacteria", "Archaea_nonDPANN"]) & (T.nuc != "")].copy()
    S["prio"] = [0 if f else (1 if c == "Archaea_nonDPANN" else 2) for f, c in zip(S.fused, S.scope_cls)]
    S = S.sort_values(["prio", "best_u50_E", "n_hh95"], ascending=[True, True, False])

    cds_rows, faa, loci = [], [], []
    done_nuc = {}
    for _, r in S.iterrows():
        if len({l["locus"] for l in loci}) >= 40 and r.nuc not in done_nuc:
            break
        if r.nuc not in done_nuc:
            try:
                txt = get(f"{EU}&db=nuccore&id={r.nuc}&rettype=fasta_cds_aa&retmode=text")
            except RuntimeError as e:
                print("unresolvable", r.acc, r.nuc, e)
                done_nuc[r.nuc] = None
                continue
            time.sleep(0.4)
            (RAW / f"cds_{r.nuc}.faa").write_text(txt)
            recs = []
            for blk in txt.split(">")[1:]:
                head, *seq = blk.split("\n")
                pid = re.search(r"\[protein_id=([^\]]+)\]", head)
                loc = re.search(r"\[location=([^\]]+)\]", head)
                prod = re.search(r"\[protein=([^\]]+)\]", head)
                if not loc:
                    continue
                s, e, st, p5, p3 = parse_loc(loc.group(1))
                recs.append(dict(nuc=r.nuc, protein_id=pid.group(1) if pid else head.split()[0], start=s, end=e,
                                 strand=st, partial5=p5, partial3=p3, product=prod.group(1) if prod else "",
                                 seq="".join(seq).strip().rstrip("*")))
            recs.sort(key=lambda x: (x["start"], x["end"]))
            for i, x in enumerate(recs):
                x["idx"] = i
            done_nuc[r.nuc] = recs
        recs = done_nuc[r.nuc]
        if not recs:
            continue
        hit = [x for x in recs if x["protein_id"] == r.pid]
        if not hit:
            print("target cds not found", r.acc, r.pid, r.nuc)
            continue
        i0 = hit[0]["idx"]
        # merge with an existing locus on the same record within 5 CDS
        prev = [l for l in loci if l["nuc"] == r.nuc and abs(l["idx"] - i0) <= 5]
        locus = prev[0]["locus"] if prev else f"L{len({l['locus'] for l in loci}) + 1:02d}"
        loci.append(dict(locus=locus, acc=r.acc, pid=r.pid, nuc=r.nuc, idx=i0, n_cds_record=len(recs),
                         target_strand=hit[0]["strand"], target_len=len(hit[0]["seq"]), scope_cls=r.scope_cls,
                         organism=r.organism, ncbi_lineage=r.ncbi_lineage, fused=r.fused, hits_A=r.hits_A,
                         hits_B=r.hits_B, hits_AB=r.hits_AB, best_u50_E=r.best_u50_E, n_hh95=r.n_hh95,
                         evidence=r.evidence))
    L = pd.DataFrame(loci)
    L.to_csv(NB / "loci.tsv", sep="\t", index=False)
    for nuc, recs in done_nuc.items():
        if not recs or nuc not in set(L.nuc):
            continue
        for x in recs:
            cds_rows.append({k: v for k, v in x.items() if k != "seq"} | dict(length=len(x["seq"])))
            faa.append(f">{nuc}|{x['protein_id']}\n{x['seq']}\n")
    pd.DataFrame(cds_rows).to_csv(NB / "cds.tsv", sep="\t", index=False)
    (NB / "contig_proteins.faa").write_text("".join(faa))
    print("loci", L.locus.nunique(), "targets", len(L), "records", L.nuc.nunique(), "CDS", len(cds_rows))
    print(L.groupby("locus").agg(nuc=("nuc", "first"), org=("organism", "first"), cls=("scope_cls", "first"),
                                 accs=("acc", ",".join), ncds=("n_cds_record", "first")).to_string())


if __name__ == "__main__":
    main()
