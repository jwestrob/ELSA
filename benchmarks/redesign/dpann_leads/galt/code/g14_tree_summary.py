#!/usr/bin/env python
"""galt step 14 (question 1): clades of the IQ-TREE ML tree (aln/galt_tree_treefile.txt, Q.pfam+F+R8, 273 taxa, 312 trimmed
columns; node labels 'SH-aLRT/UFBoot').
Supports are read as bipartitions of the unrooted tree as written (every edge is the edge above one node of the IQ-TREE
output), so no label moves under rerooting. Rooting assumption for 'smallest clade holding X': the GDP-hexose
phosphorylase references (100/100 monophyletic) are the outgroup; the clade is the smallest bipartition side that holds
X and excludes them. Reported per query set: that clade's size, composition, orders and edge support, and whether X
is exactly a bipartition side (monophyletic). Taxon classes from data/tree_taxa.tsv; named genes split by profile tier
(P = Pfam PF01087/PF02744 or KOfam K00965 row; C = CDD-only cd00608/COG1085).
Writes data/q1_clades.tsv and aln/galt_tree_annotated_nwk.txt (leaf names class|order|id; topology as written)."""
from pathlib import Path
import pandas as pd
from ete3 import Tree

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
T = Tree(str(H / "aln/galt_tree_treefile.txt"), format=1)
X = pd.read_csv(H / "data/tree_taxa.tsv", sep="\t").set_index("taxon")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t").set_index("protein_id")


def cls(t):
    g = X.loc[t, "group"]
    if g.startswith("REF_"):
        return "REF_ADPGP_IPR053177" if g == "REF_ADPGP" else g
    if g in ("named_rep50", "labeled_named", "ap4a_branch"):
        fl = C.fl.get(t, "")
        return f"{g}_{'P' if any(k in fl for k in ('pfam:PF01087', 'pfam:PF02744', 'kofam:K00965')) else 'C'}"
    return g


ALL = frozenset(T.get_leaf_names())
CL = {t: cls(t) for t in ALL}
BIP = []
for n in T.traverse():
    if n.is_root():
        continue
    s = frozenset(n.get_leaf_names())
    BIP.append((s, n.name or ""))
OUT = frozenset(t for t in ALL if CL[t] == "REF_GDPhexose_phosphorylase")


def smallest_clade(S):
    best = None
    for s, sup in BIP:
        for side in (s, ALL - s):
            if S <= side and not (side & OUT):
                if best is None or len(side) < len(best[0]):
                    best = (side, sup)
    return best


def describe(name, S):
    side, sup = smallest_clade(frozenset(S))
    comp = pd.Series([CL[t] for t in side]).value_counts().to_dict()
    orders = pd.Series([X.loc[t, "order"] for t in side if isinstance(X.loc[t, "order"], str)]).value_counts().head(6).to_dict()
    return dict(query=name, query_size=len(S), clade_size=len(side), monophyletic=len(side) == len(S), support_alrt_ufb=sup,
                composition=comp, orders=orders)


Q = {
    "lead15": [t for t in ALL if CL[t] == "lead15"],
    "set64": [t for t in ALL if X.loc[t, "set64"]],
    "labeled76": [t for t in ALL if X.loc[t, "group"] in ("lead15", "labeled_named", "labeled_unannotated")],
    "REF GalT class I": [t for t in ALL if CL[t] == "REF_GalT_I"],
    "REF GalT class II": [t for t in ALL if CL[t] == "REF_GalT_II"],
    "REF ADPGP (Q9FK51 + 14 IPR053177)": [t for t in ALL if CL[t] == "REF_ADPGP_IPR053177"],
    "REF Ap4A phosphorylase": [t for t in ALL if CL[t] == "REF_Ap4A_phosphorylase"],
    "named form tier P (119 reps)": [t for t in ALL if CL[t] == "named_rep50_P"],
}
rows = [describe(k, v) for k, v in Q.items()]
# the subfamily clade holding the 64 and its nested clades
s64 = frozenset(Q["set64"])
D, supD = smallest_clade(s64)
sides = [(side, sup) for s, sup in BIP for side in (s, ALL - s)]
nested = sorted({(side, sup) for side, sup in sides if side < D and frozenset(Q["lead15"]) <= side}, key=lambda x: -len(x[0]))
for s, sup in nested[:6]:
    comp = pd.Series([CL[t] for t in s]).value_counts().to_dict()
    orders = pd.Series([X.loc[t, "order"] for t in s if isinstance(X.loc[t, "order"], str)]).value_counts().head(6).to_dict()
    rows.append(dict(query="nested clade holding lead15", query_size=15, clade_size=len(s), monophyletic="", support_alrt_ufb=sup,
                     composition=comp, orders=orders))
# largest clade holding the ADPGP references and excluding GalT class I, subfamily D and the outgroup
adp = frozenset(Q["REF ADPGP (Q9FK51 + 14 IPR053177)"]); gt1 = frozenset(Q["REF GalT class I"])
cands = [(side, sup) for side, sup in sides if adp <= side and not (side & (gt1 | D | OUT))]
if cands:
    s, sup = max(cands, key=lambda x: len(x[0]))
    comp = pd.Series([CL[t] for t in s]).value_counts().to_dict()
    orders = pd.Series([X.loc[t, "order"] for t in s if isinstance(X.loc[t, "order"], str)]).value_counts().head(6).to_dict()
    rows.append(dict(query="largest clade with ADPGP refs, without GalT I / D", query_size=len(adp), clade_size=len(s),
                     monophyletic="", support_alrt_ufb=sup, composition=comp, orders=orders))
R = pd.DataFrame(rows)
R.to_csv(H / "data/q1_clades.tsv", sep="\t", index=False)
for t in T.iter_leaves():
    o = X.loc[t.name, "order"]
    t.name = f"{CL[t.name]}|{o if isinstance(o, str) else ''}|{t.name}"
T.write(outfile=str(H / "aln/galt_tree_annotated_nwk.txt"), format=1)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 230)
print(R.to_string(index=False))
