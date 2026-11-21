"""Structure files for the remote Foldseek / US-align job (local, seconds).

work/fs_c1/chains/   single chains: A and B of every cognate MSA fold (c1_cogmsa_*), C of every ABC MSA fold
                     (c1_abcmsa_*), the DPANN fused homologs (c1_fusedmsa_*) and the two UniProt fused homologs
                     (c1_extfusedmsa_*), and AB_concat_<slot>: chain B appended to chain A as one chain (residues
                     renumbered), i.e. the complex presented as a single chain
work/fs_c1/complex/  c1_cogmsa_* (AB) and c1_abcmsa_* (ABC) multi-chain models
work/fs_c1/usalign/  fused models and AB_concat models for pairwise US-align
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE / "work"
O = W / "fs_c1"


def chains(pdb):
    out = {}
    for line in open(pdb):
        if line.startswith("ATOM"):
            out.setdefault(line[21], []).append(line)
    return out


def write(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)
        f.write("END\n")


def concat(lines_by_chain, order):
    out, off, n = [], 0, 1
    for c in order:
        last = None
        for line in lines_by_chain[c]:
            r = int(line[22:26])
            last = r
            out.append(f"{line[:6]}{n:5d}{line[11:21]}A{r + off:4d}{line[26:]}")
            n += 1
        off += last
    return out


def main():
    for d in ("chains", "complex", "usalign"):
        (O / d).mkdir(parents=True, exist_ok=True)
    for i in range(1, 13):
        p = W / f"folds_c1/pdb/c1_cogmsa_{i:02d}.pdb"
        ch = chains(p)
        write(O / f"chains/c1_cogmsa_{i:02d}_A.pdb", ch["A"])
        write(O / f"chains/c1_cogmsa_{i:02d}_B.pdb", ch["B"])
        cc = concat(ch, "AB")
        write(O / f"chains/AB_concat_{i:02d}.pdb", cc)
        write(O / f"usalign/AB_concat_{i:02d}.pdb", cc)
        shutil.copy(p, O / f"complex/c1_cogmsa_{i:02d}.pdb")
        q = W / f"folds_c1abc/pdb/c1_abcmsa_{i:02d}.pdb"
        if q.exists():
            chq = chains(q)
            write(O / f"chains/c1_abcmsa_{i:02d}_C.pdb", chq["C"])
            shutil.copy(q, O / f"complex/c1_abcmsa_{i:02d}.pdb")
    for p in list((W / "folds_c1/pdb").glob("c1_fused*.pdb")) + list((W / "folds_c1ext/pdb").glob("*.pdb")):
        ch = chains(p)
        if "msa" in p.name:
            write(O / f"chains/{p.stem}.pdb", ch["A"])
        write(O / f"usalign/{p.stem}.pdb", ch["A"])
    print(len(list((O / "chains").glob("*.pdb"))), "chains;", len(list((O / "complex").glob("*.pdb"))), "complexes;",
          len(list((O / "usalign").glob("*.pdb"))), "us-align files")


if __name__ == "__main__":
    main()
