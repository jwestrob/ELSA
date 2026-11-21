"""Discovery pass for larger modules on lut (read-only; prints tallies used to write modules_large_rule.json; nothing
here reads any method output). Co-oriented chain = maximal run of genes on one contig and one strand with every
intergenic gap <= 500 bp (the notes' protocol). Anchors: PQQ dehydrogenase (PQQ/PQQ_2, 500-700 aa) typed by the
KOfam hits lut_loci mapped (K23995, K00114, K14028; observation only), PqqC (TENA_THI-4), Peptidase_S58 (Type A),
Lysine_decarbox (Type B), lut spans (flanks). Tokens: frozen rescue3 tokens (orf6-like 298295, orf7-like 361272)."""
import collections
import numpy as np
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
g = pd.read_parquet(PKG + "/universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "strand", "length", "start", "end"])
N = len(g); CC = g.contig_code.to_numpy(); ST = g.strand.to_numpy(); S = g.start.to_numpy(); E = g.end.to_numpy(); L = g.length.to_numpy()
P = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "name", "env_from"]).sort_values(["gid", "env_from"])
NAMES = P.groupby("gid").name.agg(lambda s: "+".join(dict.fromkeys(s)))
NM = np.full(N, "-", dtype=object); NM[NAMES.index.to_numpy()] = NAMES.to_numpy()
TOK = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
NM = np.where((NM == "-") & (TOK == 298295), "[orf6-like]", NM)
NM = np.where((NM == "-") & (TOK == 361272), "[orf7-like]", NM)
mdh = pd.read_parquet("mdh_pkg_from_lut_loci.parquet")
KO = dict(zip(mdh.gid, mdh.ko))
same = np.zeros(N, bool); same[:-1] = (CC[1:] == CC[:-1]) & (ST[1:] == ST[:-1]) & (S[1:] - E[:-1] - 1 <= 500)
chain = np.concatenate([[0], np.cumsum(~same[:-1])])        # chain id per gene
cstart = pd.Series(np.arange(N)).groupby(chain).min().to_numpy(); cend = pd.Series(np.arange(N)).groupby(chain).max().to_numpy()
has = lambda x, pat: pat in NM[x]
pq = np.flatnonzero(np.array([("PQQ" in n and "PqqD" not in n) for n in NM]) & (L >= 500) & (L <= 700))
print("PQQDH", len(pq))

def oriented(x):
    c = chain[x]; a, b = cstart[c], cend[c]
    mem = list(range(a, b + 1))
    if ST[x] < 0:
        mem = mem[::-1]
    return mem

def short(n):
    return n.split("+")[0] if n != "-" else "-"

for ko in ["K23995", "K00114", "K14028", None]:
    sub = [x for x in pq if KO.get(x) == ko] if ko else [x for x in pq if x not in KO]
    lens = [cend[chain[x]] - cstart[chain[x]] + 1 for x in sub]
    arch = collections.Counter(" ".join(short(NM[y]) for y in oriented(x)) for x in sub if cend[chain[x]] - cstart[chain[x]] + 1 >= 3)
    comp = collections.Counter(n for x in sub for y in oriented(x) if y != x for n in set(NM[y].split("+")))
    print(f"\n== PQQDH typed {ko}: n={len(sub)}; chain length quantiles {np.quantile(lens,[.25,.5,.75,.9]).tolist() if lens else []}; >=5: {sum(l>=5 for l in lens)}")
    print(" top chain-member Pfam:", comp.most_common(25))
    print(" top oriented chains (>=3 genes):"); [print("   ", c, a) for a, c in arch.most_common(12)]

# xoxFGJ core and its extensions
core = [x for x in pq if any(has(y, "Cytochrome_CBB3") for y in oriented(x) if y != x) and any(has(y, "SBP_bac_3") for y in oriented(x))]
print("\n== xoxF chains holding CBB3 and SBP_bac_3:", len(core), "genomes", len(set(g.genome_code.to_numpy()[core])))
print(" chain lengths:", collections.Counter(cend[chain[x]] - cstart[chain[x]] + 1 for x in core).most_common(10))
ext = collections.Counter(); opp = collections.Counter()
for x in core:
    o = ST[x]
    for d in range(-12, 13):
        y = x + d
        if 0 <= y < N and CC[y] == CC[x] and chain[y] != chain[x]:
            k = (d * o, "same" if ST[y] == o else "opp", short(NM[y]))
            (ext if ST[y] == o else opp)[k] += 1
print(" outside the chain, co-oriented (oriented offset, name):", [(k[0], k[2], v) for k, v in ext.most_common(15)])
print(" outside the chain, opposite strand:", [(k[0], k[2], v) for k, v in opp.most_common(15)])
pos = collections.Counter()
for x in core:
    for y in oriented(x):
        pos[(int((y - x) * ST[x]), short(NM[y]))] += 1
print(" inside chain (oriented offset, name):", pos.most_common(25))
o6 = np.flatnonzero(TOK == 298295); o7 = np.flatnonzero(TOK == 361272)
def rel(xs, ys):
    c = collections.Counter()
    for y in ys:
        near = [x for x in xs if CC[x] == CC[y] and abs(x - y) <= 10]
        if near:
            x = min(near, key=lambda x: abs(x - y))
            c[(int((y - x) * ST[x]), "same" if ST[y] == ST[x] else "opp", "samechain" if chain[y] == chain[x] else "other")] += 1
    return c.most_common(8)
print(" orf6-like vs nearest PQQDH:", rel(pq, o6)); print(" orf7-like vs nearest PQQDH:", rel(pq, o7))
print(" orf7 vs orf6:", rel(o6, o7))

# PqqC anchor
pc = np.flatnonzero(np.array(["TENA_THI-4" in n for n in NM]))
lens = [cend[chain[x]] - cstart[chain[x]] + 1 for x in pc]
print("\n== PqqC (TENA_THI-4):", len(pc), "chain length quantiles", np.quantile(lens, [.25, .5, .75, .9]).tolist())
print(" top chains:", collections.Counter(" ".join(short(NM[y]) for y in oriented(x)) for x in pc).most_common(12))

for anc, nm in [("Peptidase_S58", "Type A"), ("Lysine_decarbox", "Type B"), ("Ribul_P_3_epim", "R3P")]:
    xs = np.flatnonzero(np.array([anc in n for n in NM]))
    w = collections.Counter()
    for x in xs:
        for d in range(-8, 9):
            y = x + d
            if d and 0 <= y < N and CC[y] == CC[x]:
                w[(int(d * ST[x]), "s" if ST[y] == ST[x] else "o", short(NM[y]))] += 1
    lens = [cend[chain[x]] - cstart[chain[x]] + 1 for x in xs]
    print(f"\n== {nm} anchor {anc}: {len(xs)}; chain length quantiles {np.quantile(lens,[.25,.5,.75,.9]).tolist()}")
    print(" top (oriented offset, strand, name):", [(k, v) for k, v in w.most_common(30) if k[2] != "-"][:22])
    print(" top chains:", collections.Counter(" ".join(short(NM[y]) for y in oriented(x)) for x in xs if cend[chain[x]] - cstart[chain[x]] >= 2).most_common(8))

# chains of >= 5 genes flanking lut spans (within 10 genes), outside the span
ins = pd.read_parquet(PKG + "/gold/lut_instances.parquet"); ins = ins[ins.caller == "lut_homology_main"]
fl = collections.Counter(); seen = set()
for a, b in zip(ins.member_gids.map(min), ins.member_gids.map(max)):
    for y in list(range(a - 10, a)) + list(range(b + 1, b + 11)):
        if 0 <= y < N and CC[y] == CC[a]:
            c = chain[y]
            if c in seen or cend[c] - cstart[c] + 1 < 5 or (cstart[c] <= b and cend[c] >= a):
                continue
            seen.add(c)
            mem = list(range(cstart[c], cend[c] + 1))
            fl[" ".join(short(NM[z]) for z in (mem if ST[mem[0]] > 0 else mem[::-1]))] += 1
print("\n== co-oriented chains >= 5 genes flanking lut spans:", sum(fl.values()))
for a, c in fl.most_common(15):
    print("   ", c, a[:200])
