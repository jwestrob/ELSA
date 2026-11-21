#!/usr/bin/env python3
"""Write the ESMFold2 inputs: single chains (15 lead-1 representatives covering every mmseqs family and clade, positive
controls = DPANN proteins with Pfam Thy1 rows at the gathering threshold [two-domain Pacearchaeales MBU2497176.1,
two-domain Micrarchaeota NYZ74576.1, one-domain Woesearchaeales MBN2880691.1], negative controls = DPANN proteins of
similar length from unrelated families [Aldedh DAHVAY010000002.1_53, 538 aa; TatD_DNase MBS3126271.1, 248 aa], and the
public T. maritima ThyX Q9WYT0), plus the oligomer sets.
usage: python code/07_fold_inputs.py"""
import duckdb
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
LEAD = ["NMB66601.1", "MFA5724626.1", "MEM4152697.1", "DAOVIF010000002.1_60", "JAFCEG010000023.1_4", "HJO15200.1",
        "PWLK01000080.1_2", "JAGVGF010000004.1_10", "JAHFJQ010000034.1_2", "JACWAL010000022.1_32", "MEK6947948.1",
        "JAUWXZ010000042.1_5", "NPA86484.1", "MBS3141614.1", "MBS3141613.1"]
POS = ["MBU2497176.1", "NYZ74576.1", "MBN2880691.1"]
NEG = ["DAHVAY010000002.1_53", "MBS3126271.1"]
ids = LEAD + POS + NEG
S = dict(con.execute(f"SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN {tuple(ids)}").fetchall())
tm = "".join(open("data/ref/Q9WYT0.faa").read().split("\n")[1:])
S["THYX_THEMA_Q9WYT0"] = tm
def w(fn, keys):
    with open(fn, "w") as f:
        for k in keys:
            f.write(f">{k}\n{S[k].rstrip('*')}\n")
w("data/fold/singles.faa", ids + ["THYX_THEMA_Q9WYT0"])
w("data/fold/dimer_lig.faa", ["NMB66601.1", "MEM4152697.1", "JAGVGF010000004.1_10", "HJO15200.1", "MBU2497176.1", "NYZ74576.1", "DAHVAY010000002.1_53"])
w("data/fold/tetramer_lig.faa", ["MBN2880691.1", "THYX_THEMA_Q9WYT0"])
w("data/fold/monomer_lig.faa", ["NMB66601.1", "MEM4152697.1"])
w("data/fold/long_tetramer.faa", ["NMB66601.1", "MEM4152697.1"])
print({k: len(v) for k, v in S.items()})
