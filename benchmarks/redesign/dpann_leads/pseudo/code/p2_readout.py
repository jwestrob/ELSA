"""Generic read-out for the pseudoenzyme test (test_rule.json + addendum). Pure python/numpy/pandas; runs on the Mac
(DPANN families) and on the biotite login node (SR-VP families; sequences never leave biotite, only tables come back).
usage: p2_readout.py CONFIG.json   -> writes <out>/<fam>_positions.tsv, <fam>_summary.json, <fam>_own_invariant.tsv,
                                       <fam>_genes.tsv (per-gene residues at the positions, no sequences)
CONFIG keys: fam, title, aln, meta (protein_id, role, bin_id, group, order), ref_id, positions [[pos, residue, kind]],
  roles {labeled, canonical, cdd_only, extra: [role, ...]}, genomes_tsv (bin_id, group, order: every genome of the
  collection, for the paralog clade denominator), pdb (optional), out."""
import json, sys, difflib
from collections import Counter
import numpy as np, pandas as pd

AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
           MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")


def read_fasta(p):
    d, k = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            k = line[1:].split()[0]; d[k] = []
        elif k:
            d[k].append(line)
    return {k: "".join(v).upper() for k, v in d.items()}


def majority(vals):
    c = Counter(vals)
    return sorted(c.items(), key=lambda t: (-t[1], t[0]))[0][0]


def group_votes(df, col):
    """one vote per genome group: the majority value among the group's genes"""
    return df.groupby("group")[col].agg(majority)


def share(series, value):
    return float((series == value).mean()) if len(series) else float("nan")


def pdb_ca(path, chain=None):
    res = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and (chain is None or line[21] == chain):
            if line[16] not in (" ", "A"):
                continue
            aa = AA3.get(line[17:20])
            if aa is None:
                continue
            res.append((int(line[22:26]), aa, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return res


def run(cfg):
    A = read_fasta(cfg["aln"])
    M = pd.read_csv(cfg["meta"], sep="\t", dtype={"protein_id": str})
    M = M[M.protein_id.isin(A)].copy()
    ref = A[cfg["ref_id"]]
    pos2col, k = {}, 0
    for j, ch in enumerate(ref):
        if ch not in "-.":
            k += 1; pos2col[k] = j
    ncol = len(ref)
    refcols = [j for j, ch in enumerate(ref) if ch not in "-."]
    positions = [(int(p), r, kind) for p, r, kind in cfg["positions"]]
    for p, r, kind in positions:
        assert ref[pos2col[p]] == r, (cfg["fam"], p, r, ref[pos2col[p]])
    roles = cfg["roles"]
    M["group"] = M["group"].fillna(-1).astype(int)
    M.loc[M.group < 0, "group"] = -1 - np.arange((M.group < 0).sum())  # genomes without a group count alone

    def ident(s):
        a = [(ref[j], s[j]) for j in refcols if s[j] not in "-."]
        return sum(x == y for x, y in a) / len(a) if a else np.nan

    M["identity"] = [ident(A[i]) for i in M.protein_id]
    for p, r, kind in positions:
        c = pos2col[p]
        lo, hi = pos2col.get(p - 3, pos2col[min(pos2col)]), pos2col.get(p + 3, pos2col[max(pos2col)])
        M[f"r{p}"] = [A[i][c] for i in M.protein_id]
        M[f"s{p}"] = [(A[i][c] != r) and (r in A[i][lo:hi + 1]) for i in M.protein_id]
    lab = M[M.role == roles["labeled"]]
    can = M[M.role == roles["canonical"]]
    lo_id, hi_id = np.nanpercentile(lab.identity, [10, 90])
    win = lambda d: d[(d.identity >= lo_id) & (d.identity <= hi_id)]
    ctrl = win(M[M.role == roles.get("cdd_only", "cdd_only")])
    ctrl_kind = "cdd_only (broad-database named, identity-matched)"
    if ctrl.group.nunique() < 5:
        ctrl = win(can); ctrl_kind = "fallback: GA canonical, identity-matched"
    extra = {e: M[M.role == e] for e in roles.get("extra", [])}

    rows = []
    for p, r, kind in positions:
        col = f"r{p}"
        gv = group_votes(lab, col)
        subs = gv[(gv != r) & (gv != "-")]
        top_sub = majority(list(subs)) if len(subs) else ""
        sv = group_votes(lab, f"s{p}")
        nonc = gv != r
        row = dict(family=cfg["fam"], position=p, kind=kind, canonical_residue=r,
                   top_residue_labeled=majority(list(gv)), top_substitute=top_sub,
                   share_groups_top_substitute=round(share(gv, top_sub), 3) if top_sub else 0.0,
                   share_groups_canonical_labeled=round(share(gv, r), 3), share_groups_gap=round(share(gv, "-"), 3),
                   n_groups=int(gv.size), n_genes=int(len(lab)),
                   labeled_residue_counts_groups=";".join(f"{a}:{b}" for a, b in Counter(gv).most_common(6)),
                   shift_rate=round(float(sv[nonc].mean()), 3) if nonc.any() else 0.0, n_groups_noncanonical=int(nonc.sum()))
        cg = group_votes(can, col) if len(can) else pd.Series(dtype=str)
        row.update(canonical_share_groups=round(share(cg, r), 3), canonical_n_groups=int(cg.size))
        dg = group_votes(ctrl, col) if len(ctrl) else pd.Series(dtype=str)
        row.update(divergence_control_canonical_share=round(share(dg, r), 3), divergence_control_n_groups=int(dg.size))
        for e, d in extra.items():
            eg = group_votes(d, col) if len(d) else pd.Series(dtype=str)
            row[f"{e}_canonical_share"] = round(share(eg, r), 3); row[f"{e}_n_groups"] = int(eg.size)
        rows.append(row)
    P = pd.DataFrame(rows)
    P["verdict_position"] = P.canonical_share_groups >= 0.70
    V = P[P.verdict_position]
    nV = len(V)
    maj = lambda mask: int(mask.sum()) > nV / 2
    lab_replaced = float((1 - V.share_groups_canonical_labeled).mean()) if nV else np.nan
    ctl_replaced = float((1 - V.divergence_control_canonical_share).mean()) if nV and V.divergence_control_n_groups.min() > 0 else np.nan
    # pooled shift rate over verdict positions, in genome groups
    num = float((V.shift_rate * V.n_groups_noncanonical).sum()); den = float(V.n_groups_noncanonical.sum())
    shift = num / den if den else 0.0
    ctrl_keeps = maj(V.divergence_control_canonical_share >= 0.70) if not np.isnan(ctl_replaced) else False
    artifact = (not np.isnan(ctl_replaced) and abs(lab_replaced - ctl_replaced) <= 0.15) or shift > 0.50
    selected = maj(V.share_groups_top_substitute >= 0.70) and shift < 0.20 and ctrl_keeps
    mixed = maj(V.share_groups_top_substitute < 0.50)
    verdict = "ARTIFACT" if artifact else "SELECTED SUBSTITUTE" if selected else "MIXED" if mixed else "INTERMEDIATE"

    # (d) paralog: canonical member with the canonical residue at every verdict position, same genome
    vp = list(V.position)
    can_ok = can.copy()
    can_ok["intact"] = np.all([can_ok[f"r{p}"] == r for p, r in zip(V.position, V.canonical_residue)], axis=0) if nV else False
    act = [(p, r) for p, r, k in positions if k == "active"]
    can_ok["intact_active"] = np.all([can_ok[f"r{p}"] == r for p, r in act], axis=0) if act else True
    intact_bins = set(can_ok[can_ok.intact].bin_id.dropna())
    any_bins = set(can.bin_id.dropna()); act_bins = set(can_ok[can_ok.intact_active].bin_id.dropna())
    lab_bins = set(lab.bin_id.dropna())
    G = pd.read_csv(cfg["genomes_tsv"], sep="\t", dtype={"bin_id": str})
    clade = set(lab["order"].dropna())
    Gc = G[G["order"].isin(clade)]
    non_bins = set(Gc.bin_id) - lab_bins
    lab_with = len(lab_bins & intact_bins); non_with = len(non_bins & intact_bins)
    # group-level version
    Gc2 = Gc.assign(lab=Gc.bin_id.isin(lab_bins), intact=Gc.bin_id.isin(intact_bins))
    gl = Gc2[Gc2.lab].groupby("group").intact.max(); gn = Gc2[~Gc2.lab & ~Gc2.group.isin(gl.index)].groupby("group").intact.max()
    par = dict(labeled_genomes=len(lab_bins), labeled_genomes_with_intact=lab_with,
               clade_genomes_without_labeled=len(non_bins), without_labeled_with_intact=non_with,
               labeled_groups=int(gl.size), labeled_groups_with_intact=int(gl.sum()),
               nonlabeled_groups=int(gn.size), nonlabeled_groups_with_intact=int(gn.sum()), clade=";".join(sorted(clade)),
               canonical_members=int(len(can)), canonical_intact=int(can_ok.intact.sum()) if nV else 0,
               labeled_genomes_with_any_canonical=len(lab_bins & any_bins), without_labeled_with_any_canonical=len(non_bins & any_bins),
               labeled_genomes_with_active_site_canonical=len(lab_bins & act_bins), without_labeled_with_active_site_canonical=len(non_bins & act_bins))

    # (e) own invariant columns among labeled genome groups
    labA = {i: A[i] for i in lab.protein_id}; canA = {i: A[i] for i in can.protein_id}
    col2pos = {j: p for p, j in pos2col.items()}
    lg = lab.groupby("group").protein_id.apply(list); cgp = can.groupby("group").protein_id.apply(list)
    own = []
    for j in range(ncol):
        lv = pd.Series([majority([labA[i][j] for i in ids]) for ids in lg])
        top = majority(list(lv)); f = share(lv, top)
        if top in "-." or f < 0.90:
            continue
        cv = pd.Series([majority([canA[i][j] for i in ids]) for ids in cgp]) if len(cgp) else pd.Series(dtype=str)
        cf = share(cv, top); cmaj = majority(list(cv)) if len(cv) else ""
        if cf >= 0.90 and cmaj == top:
            continue
        own.append(dict(col=j, ref_pos=col2pos.get(j), ref_residue=ref[j] if ref[j] not in "-." else "", residue=top,
                        labeled_share=round(f, 3), canonical_share_same=round(cf, 3), canonical_majority=cmaj,
                        canonical_majority_share=round(share(cv, cmaj), 3) if cmaj else np.nan))
    O = pd.DataFrame(own, columns=["col", "ref_pos", "ref_residue", "residue", "labeled_share", "canonical_share_same",
                                   "canonical_majority", "canonical_majority_share"])
    O["dist_to_site_A"] = np.nan; O["same_pocket"] = ""
    struct = ""
    if cfg.get("pdb"):
        ca = pdb_ca(cfg["pdb"], cfg.get("chain"))
        pseq = "".join(a for _, a, _ in ca)
        sm = difflib.SequenceMatcher(None, pseq, ref.replace("-", "").replace(".", ""), autojunk=False)
        pos2xyz = {}
        for b in sm.get_matching_blocks():
            for t in range(b.size):
                pos2xyz[b.b + t + 1] = ca[b.a + t][2]
        site = [pos2xyz[p] for p in vp if p in pos2xyz]
        if site:
            cen = np.mean(site, axis=0)
            O["dist_to_site_A"] = [round(float(np.linalg.norm(pos2xyz[int(p)] - cen)), 1) if pd.notna(p) and int(p) in pos2xyz else np.nan for p in O.ref_pos]
            O["same_pocket"] = np.where(O.dist_to_site_A <= 12, "same_pocket", np.where(O.dist_to_site_A.notna(), "elsewhere", "unmapped"))
            struct = f"{cfg['pdb'].split('/')[-1]}: {len(site)}/{len(vp)} verdict positions mapped"
    S = dict(family=cfg["fam"], title=cfg["title"], verdict=verdict, n_labeled_genes=int(len(lab)), n_labeled_groups=int(lab.group.nunique()),
             n_canonical_genes=int(len(can)), n_canonical_groups=int(can.group.nunique()),
             positions_total=len(P), verdict_positions=nV, verdict_position_list=";".join(f"{r}{p}" for p, r in zip(V.position, V.canonical_residue)),
             positions_top_sub_ge70=int((V.share_groups_top_substitute >= 0.70).sum()), positions_top_sub_lt50=int((V.share_groups_top_substitute < 0.50).sum()),
             labeled_replaced_rate=round(lab_replaced, 3), control_replaced_rate=None if np.isnan(ctl_replaced) else round(ctl_replaced, 3),
             control_kind=ctrl_kind, control_genes=int(len(ctrl)), control_groups=int(ctrl.group.nunique()),
             identity_window=[round(lo_id, 3), round(hi_id, 3)], labeled_identity_median=round(float(lab.identity.median()), 3),
             control_identity_median=round(float(ctrl.identity.median()), 3) if len(ctrl) else None,
             canonical_identity_median=round(float(can.identity.median()), 3) if len(can) else None,
             shift_rate_pooled=round(shift, 3), control_keeps_canonical=bool(ctrl_keeps), paralog=par,
             own_invariant_n=int(len(O)), own_invariant_residues="".join(sorted(O.residue)) if len(O) else "",
             own_invariant_same_pocket=int((O.same_pocket == "same_pocket").sum()), own_invariant_elsewhere=int((O.same_pocket == "elsewhere").sum()),
             own_invariant_unmapped=int((O.same_pocket.isin(["unmapped", ""])).sum()), structure=struct,
             extra_roles={e: dict(genes=int(len(d)), groups=int(d.group.nunique()),
                                  replaced_rate=round(float((1 - V[f"{e}_canonical_share"]).mean()), 3) if nV and len(d) else None,
                                  identity_median=round(float(d.identity.median()), 3) if len(d) else None) for e, d in extra.items()})
    out = cfg["out"]
    P.to_csv(f"{out}/{cfg['fam']}_positions.tsv", sep="\t", index=False)
    O.to_csv(f"{out}/{cfg['fam']}_own_invariant.tsv", sep="\t", index=False)
    keep = ["protein_id", "role", "bin_id", "group", "order", "identity"] + [f"r{p}" for p, _, _ in positions] + [f"s{p}" for p, _, _ in positions]
    M[keep].to_csv(f"{out}/{cfg['fam']}_genes.tsv", sep="\t", index=False)
    json.dump(S, open(f"{out}/{cfg['fam']}_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    run(json.load(open(sys.argv[1])))
