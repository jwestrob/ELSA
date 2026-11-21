#!/usr/bin/env python3
"""Step 16: fold inputs for the lead groups: 2-3 members per lead (distinct genera), positive controls (DPANN proteins
annotated with each part's family at the gathering threshold, highest score near the typical length, single-domain), and
negative controls (DPANN proteins of similar length from unrelated families). Writes folds/fold_input.faa and
folds/fold_plan.tsv."""
import duckdb, pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
os.makedirs("folds", exist_ok=True)
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
def ga(name, lo, hi, src="pfam", acc=None):
    q = f"""SELECT a.protein_id, p.sequence_length len, a.score, p.sequence,
              (SELECT count(DISTINCT b."name") FROM d.annotations b WHERE b.protein_id=a.protein_id AND b.source='pfam') npf
            FROM d.annotations a JOIN d.proteins p USING (protein_id)
            WHERE a.source='{src}' AND {"a.name=?" if acc is None else "a.accession=?"} AND p.sequence_length BETWEEN {lo} AND {hi}
            ORDER BY npf, a.score DESC LIMIT 1"""
    return con.execute(q, [name if acc is None else acc]).df().iloc[0]
plan = []
mem = {"L1_NTase_kinase": ["HII14378.1", "MBX4196761.1", "HLC90278.1"],
       "L2_Tmk_NTase5": ["MBT7106118.1", "MBN2421503.1", "MDP3698733.1"],
       "L3_bicupin": ["MBS3154103.1", "MDO8634247.1"],
       "L4_KamA_like": ["MBT6744359.1", "MFA5796516.1"]}
seq = lambda pid: con.execute("SELECT sequence, sequence_length FROM d.proteins WHERE protein_id=?", [pid]).fetchone()
for lead, ps in mem.items():
    for p in ps:
        s, L = seq(p); plan.append(dict(lead=lead, role="member", protein_id=p, len=L, family="", sequence=s))
ctrl = [("L1_NTase_kinase", "pos_NTP_transf_3", ga("NTP_transf_3", 220, 260)),
        ("L1_NTase_kinase", "pos_Choline_kinase", ga("Choline_kinase", 260, 340)),
        ("L1_NTase_kinase", "neg_Cpn60_TCP1", ga("Cpn60_TCP1", 530, 570)),
        ("L2_Tmk_NTase5", "pos_Thymidylate_kin", ga("Thymidylate_kin", 190, 220)),
        ("L2_Tmk_NTase5", "pos_NTP_transf_5", ga("NTP_transf_5", 180, 260)),
        ("L2_Tmk_NTase5", "neg_Enolase_C", ga("Enolase_C", 400, 460)),
        ("L3_bicupin", "pos_MannoseP_isomer", ga("MannoseP_isomer", 110, 140)),
        ("L3_bicupin", "neg_Methyltransf_11", ga("Methyltransf_11", 230, 270)),
        ("L4_KamA_like", "pos_KamA_K01843", ga(None, 380, 480, "kofam", "K01843")),
        ("L4_KamA_like", "neg_tRNA-synt_2", ga("tRNA-synt_2", 500, 560))]
for lead, role, r in ctrl:
    plan.append(dict(lead=lead, role=role, protein_id=r.protein_id, len=int(r.len), family=role.split("_", 1)[1], sequence=r.sequence))
P = pd.DataFrame(plan)
P[["lead", "role", "protein_id", "len", "family"]].to_csv("folds/fold_plan.tsv", sep="\t", index=False)
with open("folds/fold_input.faa", "w") as f:
    for r in P.drop_duplicates("protein_id").itertuples(): f.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
print(P[["lead", "role", "protein_id", "len"]].to_string(index=False)); print(P.protein_id.nunique(), "folds")
