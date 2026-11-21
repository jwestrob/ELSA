"""Deep dive, step 3 (local, seconds): query FASTAs for the remote searches (biotite jobs dd_hh / dd_hmm).

hh/  c1A.fa      unique token-81718 sequences, replicate-1 chain A (HIH59103.1, gid 366161) first
     c1B.fa      unique token-393054 sequences <= 100 aa (fused homologs left out), HIH59104.1 (366162) first
     c1A_inst.fa, c1B_inst.fa   one row per block instance (key = instance_id), same order in both files; the
                                job aligns each and concatenates rows into c1AB.fa (instance pairing kept)
     c1F1.fa, c1F2.fa           fused homologs MBU0906634.1 (1218213, 133 aa) and MCK5039384.1 (1650759)
sp/  signalp_in.fa  SignalP 6 input: every candidate-2 instance chain, every candidate-1 instance chain, the
                    curated Archaeal-T4P members <= 400 aa and every protein with a class III / N_methyl /
                    Pilin_N Pfam row (raw rows; used only to choose control proteins)
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"


def read_fasta(p):
    seqs, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            seqs[h] = []
        elif line:
            seqs[h].append(line)
    return {k: "".join(v).rstrip("*") for k, v in seqs.items()}


def write(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, s in items:
            f.write(f">{k}\n{s}\n")


def main():
    S = read_fasta(W / "dd_seqs.faa")
    g = pd.read_parquet(HERE.parent / "work/genes_dark.parquet", columns=["gid", "token", "length"])
    c1 = pd.read_csv(W / "c1_instances.tsv", sep="\t")
    hh = W / "hh_in"

    def uniq(gids, first):
        out, seen = [(str(first), S[str(first)])], {S[str(first)]}
        for x in gids:
            s = S[str(x)]
            if s not in seen and "X" not in s:
                seen.add(s)
                out.append((str(x), s))
        return out

    a = g.gid[g.token == 81718].tolist()
    b = g.gid[(g.token == 393054) & (g.length <= 100)].tolist()
    write(hh / "c1A.fa", uniq(a, 366161))
    write(hh / "c1B.fa", uniq(b, 366162))
    ia, ib = [], []
    rep1 = c1[c1.gids == "366161;366162"].instance_id.iloc[0]
    order = [rep1] + [i for i in c1.instance_id if i != rep1]
    c1i = c1.set_index("instance_id")
    for i in order:
        x, y = c1i.at[i, "gids"].split(";")
        if "X" in S[x] or "X" in S[y]:
            continue
        ia.append((i, S[x]))
        ib.append((i, S[y]))
    write(hh / "c1A_inst.fa", ia)
    write(hh / "c1B_inst.fa", ib)
    write(hh / "c1F1.fa", [("1218213", S["1218213"])])
    write(hh / "c1F2.fa", [("1650759", S["1650759"])])
    print("c1A", len(uniq(a, 366161)), "c1B", len(uniq(b, 366162)), "instances", len(ia))

    # SignalP input
    sp = {}
    c2 = pd.read_csv(W / "c2_instances.tsv", sep="\t")
    for s_ in list(c2.gids) + list(c1.gids):
        for x in s_.split(";"):
            sp[x] = S[x]
    t4p = pd.read_csv(W / "t4p_members.tsv", sep="\t")
    for x in t4p.gid[t4p.sequence_length <= 400].astype(int).astype(str):
        sp[x] = S[x]
    c3 = pd.read_csv(W / "classIII_pfam.tsv", sep="\t")
    for x in c3.gid.astype(int).astype(str):
        sp[x] = S[x]
    write(W / "sp_in/signalp_in.fa", [(k, v) for k, v in sp.items() if "X" not in v[:70]])
    print("signalp sequences", len(sp))


if __name__ == "__main__":
    main()
