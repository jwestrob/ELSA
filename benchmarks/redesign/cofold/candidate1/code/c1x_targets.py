"""Task 1, step 1 (local, HTTP lookups only): public homologs of A, B and fused A|B outside DPANN.

Sources (deep dive, deepdive/work/):
  hh_hits.tsv      HHblits UniRef30_2020_06 hits (search 'u') of the A, B, A|B, F1, F2 profiles; P >= 95 kept
  u50_c1_hits.tsv  hmmsearch of the A, B, A|B HMMs against UniRef50; E <= 1e-5 kept
Every kept UniProt accession is looked up by accession at UniProt REST (lineage, length, EMBL cross-references
with ProteinId). Deleted entries are read from UniSave (last version). Nothing but public accessions is sent.
Writes work/nb/targets.tsv (one row per accession: source hits, best evidence, lineage class, EMBL protein ids).
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DD = HERE.parent / "deepdive" / "work"
OUT = HERE / "work" / "nb"
RAW = HERE / "work" / "nb_raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "generic-research-client/1.0"}
DPANN_TAXA = ("DPANN group", "Nanoarchaeota", "Woesearchaeota", "Pacearchaeota", "Aenigmarchaeota", "Micrarchaeota",
              "Parvarchaeota", "Diapherotrites", "Iainarchaeota", "Nanohaloarchaeota", "Altiarchaeota",
              "Altarchaeota", "Huberarchaeota", "Undinarchaeota", "Nanobdellota", "Aenigmatarchaeota")


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"{url}: {err}")


def main():
    hh = pd.read_csv(DD / "hh_hits.tsv", sep="\t")
    hh = hh[(hh.search == "u") & (hh.prob >= 95)].copy()
    hh["acc"] = hh.id.str.replace("UniRef100_", "", regex=False)
    u = pd.read_csv(DD / "u50_c1_hits.tsv", sep="\t")
    u = u[u.E <= 1e-5].copy()
    u["acc"] = u.target.str.replace("UniRef50_", "", regex=False)
    ev = {}
    for _, r in hh.iterrows():
        ev.setdefault(r.acc, []).append(f"hh:{r.query}:P{r.prob}:q{r.qs}-{r.qe}:t{r.ts}-{r.te}/{r.tlen}")
    for _, r in u.iterrows():
        ev.setdefault(r.acc, []).append(f"u50:{r.hmm}:E{r.E:.1e}")
    accs = sorted(a for a in ev if not a.startswith("UPI"))
    print("accessions", len(accs), "UniParc-only skipped", sum(a.startswith("UPI") for a in ev))
    rows = []
    for i in range(0, len(accs), 50):
        chunk = accs[i:i + 50]
        url = ("https://rest.uniprot.org/uniprotkb/accessions?accessions=" + ",".join(chunk) + "&format=json")
        js = json.loads(get(url))
        (RAW / f"uniprot_{i:03d}.json").write_text(json.dumps(js))
        got = {}
        for e in js.get("results", []):
            acc = e["primaryAccession"]
            lin = e.get("organism", {}).get("lineage", [])
            embl = [(x["id"], {p["key"]: p["value"] for p in x.get("properties", [])})
                    for x in e.get("uniProtKBCrossReferences", []) if x["database"] == "EMBL"]
            refseq = [x["id"] for x in e.get("uniProtKBCrossReferences", []) if x["database"] == "RefSeq"]
            got[acc] = dict(acc=acc, organism=e.get("organism", {}).get("scientificName", ""),
                            taxid=e.get("organism", {}).get("taxonId"), lineage=";".join(lin),
                            length=e.get("sequence", {}).get("length"),
                            embl=";".join(f"{n}|{p.get('ProteinId', '-')}|{p.get('MoleculeType', '-')}" for n, p in embl),
                            refseq=";".join(refseq), status="active")
        for acc in chunk:
            if acc not in got:
                got[acc] = dict(acc=acc, status="missing")
        rows += list(got.values())
        time.sleep(0.3)
    T = pd.DataFrame(rows)
    # deleted / missing entries: UniSave last version
    for k, r in T[T.status == "missing"].iterrows():
        try:
            meta = json.loads(get(f"https://rest.uniprot.org/unisave/{r.acc}?format=json"))
            vers = [x["entryVersion"] for x in meta.get("results", [])]
            if not vers:
                continue
            last = max(vers)
            # the last version of a deleted entry is a deletion stub; take the newest version with content
            for v in sorted(vers, reverse=True):
                txt = get(f"https://rest.uniprot.org/unisave/{r.acc}?format=txt&versions={v}")
                if "\nSQ " in txt or txt.startswith("ID "):
                    break
            (RAW / f"unisave_{r.acc}_v{v}.txt").write_text(txt)
            embl = re.findall(r"^DR   EMBL; ([^;]+); ([^;]+); [^;]+; ([^.]+)\.", txt, flags=re.M)
            oc = " ".join(l[5:] for l in txt.splitlines() if l.startswith("OC   ")).replace(".", "").split("; ")
            os_ = " ".join(l[5:] for l in txt.splitlines() if l.startswith("OS   ")).rstrip(".")
            ln = re.search(r"SQ   SEQUENCE\s+(\d+) AA", txt)
            T.loc[k, ["organism", "lineage", "length", "embl", "status"]] = [
                os_, ";".join(x.strip() for x in oc), int(ln.group(1)) if ln else None,
                ";".join(f"{a}|{b}|{c}" for a, b, c in embl), f"unisave_v{v}_of_{last}"]
            time.sleep(0.3)
        except Exception as e:  # noqa: BLE001
            print("unisave failed", r.acc, e, file=sys.stderr)

    def cls(lin):
        lin = str(lin)
        if "Eukaryota" in lin:
            return "Eukaryota"
        if "Viruses" in lin:
            return "virus"
        if "Archaea" in lin:
            return "Archaea_DPANN" if any(t in lin for t in DPANN_TAXA) else "Archaea_nonDPANN"
        if "Bacteria" in lin:
            return "Bacteria"
        return "unclassified/metagenome"

    T["cls"] = T.lineage.map(cls)
    T["evidence"] = T.acc.map(lambda a: ";".join(ev[a]))
    T["n_hh95"] = T.acc.map(lambda a: sum(x.startswith("hh:") for x in ev[a]))
    T["best_u50_E"] = T.acc.map(lambda a: min([float(x.split(":E")[1]) for x in ev[a] if x.startswith("u50:")] or [1.0]))
    T["hits_A"] = T.evidence.str.contains(r":c1A:", regex=True)
    T["hits_B"] = T.evidence.str.contains(r":c1B:", regex=True)
    T["hits_AB"] = T.evidence.str.contains(r":c1AB:|:c1F1:|:c1F2:", regex=True)
    T.to_csv(OUT / "targets.tsv", sep="\t", index=False)
    print(T.cls.value_counts().to_string())
    print(T[["acc", "cls", "organism", "length", "status", "embl", "best_u50_E", "n_hh95"]].to_string())


if __name__ == "__main__":
    main()
