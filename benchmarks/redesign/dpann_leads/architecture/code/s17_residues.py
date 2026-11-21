#!/usr/bin/env python3
"""Step 17 (ladder rung 3): residues of characterized relatives at the aligned positions of lead members.
Per lead: MAFFT L-INS-i of up to 8 members (distinct 95% clusters first) with the reference UniProt sequences; each
annotated reference residue (UniProt features, data/uniprot_features.tsv) is read out in every member. For the bicupin
the two cupin signature motifs are tested by regular expression in each half. Writes data/residues_<lead>.tsv and
data/aln_<lead>.faa."""
import duckdb, pandas as pd, os, re, subprocess, tempfile
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
F = pd.read_csv("data/uniprot_features.tsv", sep="\t")
refseq = {}
for blk in open(f"{SP}/ref/ref.faa").read().split(">")[1:]:
    h, *s = blk.split("\n"); refseq[h.split()[0]] = "".join(s)
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
LEADS = {
    "L1_NTase_kinase": dict(reps=["HLC90278.1"], refs=["P32173", "A0A0H2UQB5", "P0A3Y5", "P35790", "Q7WYX5"]),
    "L2_Tmk_NTase5": dict(reps=["MBN2421503.1", "HIG96325.1"], refs=["P0A720", "P23919"]),
    "L4_KamA_like": dict(reps=["MBT6744359.1"], refs=["Q9XBQ8", "Q6LYX4"]),
    "L5_M29_long": dict(reps=["MBI2650380.1"], refs=["P39762"]),
}
def pick(reps, n=8):
    g = A[A.rep.isin(reps)].copy(); g["genus"] = g.taxonomy.str.split(";").str[5]
    return list(g.sort_values("len").drop_duplicates("genus").protein_id[:n])
def mafft(recs):
    with tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False, dir=SP) as f:
        for k, s in recs: f.write(f">{k}\n{s}\n")
    out = subprocess.run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", "4", "--quiet", f.name], capture_output=True, text=True).stdout
    al = {}
    for blk in out.split(">")[1:]:
        h, *s = blk.split("\n"); al[h.split()[0]] = "".join(s).upper()
    return al
for lead, spec in LEADS.items():
    mem = pick(spec["reps"])
    al = mafft([(m, seqs[m].rstrip("*")) for m in mem] + [(r, refseq[r]) for r in spec["refs"]])
    with open(f"data/aln_{lead}.faa", "w") as f:
        for k, s in al.items(): f.write(f">{k}\n{s}\n")
    def col_of(seq_al, pos):  # 1-based residue -> alignment column
        c = 0
        for i, ch in enumerate(seq_al):
            if ch != "-":
                c += 1
                if c == pos: return i
    def res_at(seq_al, col):
        if seq_al[col] == "-": return "-"
        return f"{seq_al[col]}{len(seq_al[:col + 1].replace('-', ''))}"
    rows = []
    for r in spec["refs"]:
        for f in F[(F.acc == r) & F.type.isin(["Active site", "Binding site", "Site", "Modified residue"])].itertuples():
            if f.type == "Modified residue" and "pyridoxal" not in str(f.note): continue
            for pos in range(int(f.start), int(f.end) + 1):
                c = col_of(al[r], pos)
                row = dict(ref=r, type=f.type, ligand=f.ligand if isinstance(f.ligand, str) else f.note, ref_res=res_at(al[r], c))
                for m in mem: row[m] = res_at(al[m], c)
                rows.append(row)
    R = pd.DataFrame(rows)
    if len(R):
        refaa = R.ref_res.str[0]
        R["n_identical"] = [sum(str(R.loc[i, m])[0] == refaa[i] for m in mem) for i in R.index]
        R["n_members"] = len(mem)
    R.to_csv(f"data/residues_{lead}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"\n##### {lead}: members {mem}")
    print(R.to_string(index=False))
# bicupin motifs
print("\n##### L3_bicupin cupin motifs")
m1 = re.compile(r"G.{5}H.H.{3,4}E.{6}G"); m2 = re.compile(r"G.{5}P.G.{2}H.{3}N")
g = A[A.rep == "MBS3154103.1"]
rows = []
for pid in list(g.protein_id) + ["MDO8634247.1"]:
    s = seqs[pid].rstrip("*"); half = len(s) // 2
    for nm, rx in (("motif1 G(X)5HXH(X)3,4E(X)6G", m1), ("motif2 G(X)5PXG(X)2H(X)3N", m2)):
        hits = [(mm.start() + 1, mm.group()) for mm in rx.finditer(s)]
        rows.append(dict(protein_id=pid, motif=nm, n=len(hits), N_half=sum(h[0] <= half for h in hits), C_half=sum(h[0] > half for h in hits), hits=";".join(f"{a}:{b}" for a, b in hits)))
M = pd.DataFrame(rows).drop_duplicates(["motif", "hits"]); M.to_csv("data/residues_L3_bicupin.tsv", sep="\t", index=False)
print(M.to_string(index=False))
# relaxed cupin metal motifs (added after the strict motifs returned 0 hits in every member): H.H.{3,4}E (motif 1 core) and
# H.{3}N (motif 2 core) per half
r1 = re.compile(r"H.H.{3,4}E"); r2 = re.compile(r"H.{3}N")
rows = []
for pid in ["MBS3154103.1", "JAHFJQ010000006.1_10", "MDO8634247.1", "DAISQB010000011.1_40"]:
    s = seqs[pid].rstrip("*") if pid in seqs.index else None
    if s is None:
        import duckdb
        c = duckdb.connect(); c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
        s = c.execute("SELECT sequence FROM d.proteins WHERE protein_id=?", [pid]).fetchone()[0].rstrip("*")
    rows.append(dict(protein_id=pid, len=len(s), core1_HxH_E=[(m.start() + 1, m.group()) for m in r1.finditer(s)], core2_HxxxN=[(m.start() + 1, m.group()) for m in r2.finditer(s)]))
print(pd.DataFrame(rows).to_string(index=False))
