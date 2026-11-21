#!/usr/bin/env python3
"""gapscreen step 3 (Mac): gene-level function from partners, gaps, pathway context, leads, global numbers, controls.

Definitions (written before the screen was run; also in REPORT.txt):
  function   plurality KO among the supporting partners that carried the winning label (partner_ko.parquet); share =
             ko_top_n / partners-with-label-and-any-KO. STRONG: ko_top_n >= 3 and share >= 0.70.
  gap        the genome holds (a) no gene with the KO (kofam / kegg rows), (b) no KO sharing a complete EC number with it
             (non-homologous isofunctional enzymes), (c) no satisfied alternative for any KEGG module step that contains
             the KO, and (d) no Pfam row for the winning label (accession or name form). gap_ko = (a)-(c) only.
  context    for each KEGG module holding the KO, the share of the module's other top-level steps satisfied by the
             genome's annotated KOs; best module kept. otherwise-present pathway = >= 2 other steps and share >= 0.8.
  clade      bins.taxonomy order. Lead = (KO, order).
  null       KOs shuffled among STRONG genes of the same order (20 permutations, numpy default_rng(20261121)).
Outputs: data/genes_function.parquet, screen_leads.tsv (all leads, ranked), global.json, plus printed tables.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parents[1]
D = H / "data"
MIN_N, MIN_SHARE = 3, 0.70
TIER_RANK = {"T1": 0, "T2": 1, "T2b": 2, "contradicted": 3, "rest": 4}


# ------------------------------------------------------------------ KEGG module definitions
TOK = re.compile(r"K\d{5}|M\d{5}|--|[(),+\- ]")


class P:
    def __init__(self, s):
        self.t = TOK.findall(s.strip())
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        x = self.t[self.i]
        self.i += 1
        return x

    def p_or(self):
        n = [self.p_and()]
        while self.peek() == ",":
            self.take()
            n.append(self.p_and())
        return ("or", n) if len(n) > 1 else n[0]

    def p_and(self):
        n = [self.p_cx()]
        while self.peek() == " ":
            while self.peek() == " ":
                self.take()
            if self.peek() in (None, ",", ")"):
                break
            n.append(self.p_cx())
        return ("and", n) if len(n) > 1 else n[0]

    def p_cx(self):
        items = [(self.p_f(), True)]
        while self.peek() in ("+", "-"):
            op = self.take()
            items.append((self.p_f(), op == "+"))
        return ("cx", items) if len(items) > 1 else items[0][0]

    def p_f(self):
        x = self.peek()
        if x == "(":
            self.take()
            n = self.p_or()
            if self.peek() == ")":
                self.take()
            return n
        if x == "--":
            self.take()
            return ("undef",)
        if x == "-":
            self.take()
            return ("opt", self.p_f())
        self.take()
        if x.startswith("K"):
            return ("ko", x)
        return ("undef",)


def split_top(defn):
    steps, depth, cur = [], 0, ""
    for ch in defn.strip():
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == " " and depth == 0:
            if cur.strip():
                steps.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        steps.append(cur.strip())
    return steps


def ev(n, S):
    k = n[0]
    if k == "ko":
        return n[1] in S
    if k == "or":
        return any(ev(c, S) for c in n[1])
    if k == "and":
        return all(ev(c, S) for c in n[1])
    if k == "cx":
        return all(ev(c, S) for c, req in n[1] if req)
    return True


def kos(n, out):
    if n[0] == "ko":
        out.add(n[1])
    elif n[0] in ("or", "and"):
        for c in n[1]:
            kos(c, out)
    elif n[0] == "cx":
        for c, _ in n[1]:
            kos(c, out)
    elif n[0] == "opt":
        kos(n[1], out)
    return out


def main():
    mods = pd.read_csv(D / "kegg_modules.tsv", sep="\t")
    MOD = {}
    ko2steps = defaultdict(list)
    for m, nm, defn in zip(mods.module, mods.name, mods.definition):
        steps = [P(s).p_or() for s in split_top(str(defn))]
        ks = [kos(s, set()) for s in steps]
        MOD[m] = (nm, steps, ks)
        for i, kk in enumerate(ks):
            for k in kk:
                ko2steps[k].append((m, i))
    kk = pd.read_csv(D / "kegg_ko.tsv", sep="\t", keep_default_na=False)
    koname = dict(zip(kk.ko, kk.name))
    ko_ec = {k: [e for e in ec.split(";") if e and "-" not in e] for k, ec in zip(kk.ko, kk.ec)}
    ec_ko = defaultdict(set)
    for k, es in ko_ec.items():
        for e in es:
            ec_ko[e].add(k)
    gk = pd.read_parquet(D / "genome_ko.parquet")
    KO = gk.groupby("bin_id").ko.apply(set).to_dict()
    gp = pd.read_parquet(D / "genome_pfam.parquet")
    PF = defaultdict(set)
    for b_, a_, n_ in zip(gp.bin_id, gp.pf, gp.pfname):
        PF[b_].add(a_)
        PF[b_].add(n_)
    bins = pd.read_parquet(D / "bins.parquet").set_index("bin_id")
    L = pd.read_parquet(D / "labeled.parquet")
    pk = pd.read_parquet(D / "partner_ko.parquet")
    L = L.merge(pk[["protein_id", "n_supporters", "n_carriers", "n_carriers_ko", "ko_top", "ko_top_n", "ko_second", "ko_second_n",
                    "ko_counts", "rep_partner_gid"]], on="protein_id")
    L["order"] = L.bin_id.map(bins["order"])
    L["phylum"] = L.bin_id.map(bins["phylum"])
    L["completeness"] = L.bin_id.map(bins["completeness"])
    L["ko_share"] = np.where(L.n_carriers_ko > 0, L.ko_top_n / L.n_carriers_ko.clip(lower=1), np.nan)
    L["strong_partner"] = (L.ko_top_n >= MIN_N) & (L.ko_share >= MIN_SHARE)
    L["embedding_only"] = L.alignment_lists_same_label != "yes"
    glob = dict(labeled=len(L), with_carrier_ko=int((L.n_carriers_ko > 0).sum()), strong_partner=int(L.strong_partner.sum()),
                strong_partner_kos=int(L[L.strong_partner].ko_top.nunique()))
    # calibration against own KO (partner-derived function)
    own = L[L.own_ko != ""].copy()
    own["match"] = [k in set(o.split(";")) for k, o in zip(own.ko_top, own.own_ko)]
    glob["calibration_partner"] = dict(own_ko_genes=len(own), any_function=int((own.ko_top != "").sum()),
                                       match_any=round(float(own[own.ko_top != ""].match.mean()), 4),
                                       strong=int(own.strong_partner.sum()), match_strong=round(float(own[own.strong_partner].match.mean()), 4))
    # label-level fallback (data/label_ko.tsv): used only where the partners give no strong function
    lk = pd.read_csv(D / "label_ko.tsv", sep="\t").set_index("label")
    lstrong = lk[(lk.n_with_ko >= 5) & (lk.share >= 0.8)]
    lab_fun = L.proposed_label_positional_homology.map(lstrong.top_ko)
    lab_share = L.proposed_label_positional_homology.map(lstrong.share)
    L["func"] = np.where(L.strong_partner, L.ko_top, np.where(lab_fun.notna(), lab_fun,
                         np.where(L.proposed_label_kind == "ko", L.proposed_label_positional_homology, "")))
    L["func_source"] = np.where(L.strong_partner, "partners", np.where(lab_fun.notna(), "label",
                                np.where(L.proposed_label_kind == "ko", "label_is_ko", "")))
    L["ko_share"] = np.where(L.strong_partner, L.ko_share, np.where(lab_fun.notna(), lab_share, np.where(L.func != "", 1.0, np.nan)))
    own = L[L.own_ko != ""].copy()
    own["match"] = [k in set(o.split(";")) for k, o in zip(own.func, own.own_ko)]
    glob["calibration_by_source"] = {s: dict(n=int((own.func_source == s).sum()), match=round(float(own[own.func_source == s].match.mean()), 4))
                                     for s in ("partners", "label", "label_is_ko") if (own.func_source == s).any()}
    L["ko_top_partner"] = L.ko_top
    L["ko_top"] = L.func
    L["strong"] = L.func != ""
    glob.update(strong=int(L.strong.sum()), strong_by_source=L[L.strong].func_source.value_counts().to_dict(),
                strong_kos=int(L[L.strong].ko_top.nunique()), strong_with_ec=int(L[L.strong].ko_top.map(lambda k: bool(ko_ec.get(k))).sum()))

    def assess(b_, f, label):
        S = KO.get(b_, set())
        pres_ko = f in S
        ecs = ko_ec.get(f, [])
        pres_ec = any((ec_ko[e] - {f}) & S for e in ecs)
        pres_step = False
        best = (np.nan, 0, "")
        for m, i in ko2steps.get(f, []):
            nm, steps, _ = MOD[m]
            if ev(steps[i], S):
                pres_step = True
            other = [j for j in range(len(steps)) if j != i and f not in MOD[m][2][j]]
            if other:
                fr = sum(ev(steps[j], S) for j in other) / len(other)
                if np.isnan(best[0]) or (fr, len(other)) > (best[0], best[1]):
                    best = (fr, len(other), m)
        pres_pf = label in PF.get(b_, set())
        gap_ko = not (pres_ko or pres_ec or pres_step)
        return pres_ko, pres_ec, pres_step, pres_pf, gap_ko, gap_ko and not pres_pf, best

    S_ = L[L.strong].copy()
    res = [assess(b_, f, lab) for b_, f, lab in zip(S_.bin_id, S_.ko_top, S_.proposed_label_positional_homology)]
    for j, c in enumerate(["pres_ko", "pres_ec", "pres_step", "pres_pf", "gap_ko", "gap"]):
        S_[c] = [r[j] for r in res]
    S_["ctx_frac"] = [r[6][0] for r in res]
    S_["ctx_nother"] = [r[6][1] for r in res]
    S_["ctx_module"] = [r[6][2] for r in res]
    S_["in_context"] = S_.gap & (S_.ctx_nother >= 2) & (S_.ctx_frac >= 0.8)
    S_["in_ctx50"] = S_.gap & (S_.ctx_nother >= 2) & (S_.ctx_frac >= 0.5)
    S_["has_module"] = S_.ctx_nother > 0
    S_["enzyme"] = S_.ko_top.map(lambda k: bool(ko_ec.get(k)))
    S_.to_parquet(D / "genes_function.parquet", index=False)

    # genome x function gaps
    gg = S_[S_.gap].copy()
    gg["tr"] = gg.tier.map(TIER_RANK)
    pair = gg.groupby(["bin_id", "ko_top"]).agg(tr=("tr", "min"), both=("embedding_only", lambda x: bool((~x).any())),
                                                 in_context=("in_context", "any"), in_ctx50=("in_ctx50", "any"),
                                                 has_module=("has_module", "any"), src=("func_source", "first"),
                                                 enzyme=("enzyme", "first"), comp=("completeness", "first")).reset_index()
    inv = {v: k for k, v in TIER_RANK.items()}
    glob["gaps"] = dict(strong_genes=len(S_), gap_genes=int(S_.gap.sum()), gap_ko_genes=int(S_.gap_ko.sum()),
                        genome_function_gaps=len(pair), with_module=int(pair.has_module.sum()), in_context=int(pair.in_context.sum()),
                        in_ctx50=int(pair.in_ctx50.sum()), by_source=pair.src.value_counts().to_dict(),
                        enzyme_gaps=int(pair.enzyme.sum()),
                        by_tier={inv[t]: dict(gaps=int((pair.tr == t).sum()), in_context=int(((pair.tr == t) & pair.in_context).sum()))
                                 for t in sorted(pair.tr.unique())},
                        embedding_only=dict(gaps=int((~pair.both).sum()), in_context=int(((~pair.both) & pair.in_context).sum())),
                        both_lists=dict(gaps=int(pair.both.sum()), in_context=int((pair.both & pair.in_context).sum())),
                        complete_ge90=dict(gaps=int((pair.comp >= 90).sum()), in_context=int(((pair.comp >= 90) & pair.in_context).sum())))
    # completeness control: gap rate by genome completeness
    S_["comp_bin"] = pd.cut(S_.completeness, [0, 70, 80, 90, 95, 101], right=False)
    glob["gap_rate_by_completeness"] = {str(k): dict(n=int(len(g)), gap=round(float(g.gap.mean()), 4),
                                                     in_context=round(float(g.in_context.mean()), 4))
                                        for k, g in S_.groupby("comp_bin", observed=True)}
    # null: KOs shuffled among strong genes of the same order
    rng = np.random.default_rng(20261121)
    null = []
    for _ in range(20):
        # the winning label travels with the KO (the gene's whole function is shuffled)
        sh = S_.ko_top.to_numpy().copy()
        labs = S_.proposed_label_positional_homology.to_numpy().copy()
        for o, idx in S_.groupby("order").indices.items():
            perm = rng.permutation(len(idx))
            sh[idx] = S_.ko_top.to_numpy()[idx][perm]
            labs[idx] = S_.proposed_label_positional_homology.to_numpy()[idx][perm]
        rr = [assess(b_, f, lab) for b_, f, lab in zip(S_.bin_id, sh, labs)]
        gap = np.array([r[5] for r in rr])
        ctx = np.array([r[5] and r[6][1] >= 2 and r[6][0] >= 0.8 for r in rr])
        ctx50 = np.array([r[5] and r[6][1] >= 2 and r[6][0] >= 0.5 for r in rr])
        hm = np.array([r[6][1] > 0 for r in rr])
        pr = pd.DataFrame(dict(b=S_.bin_id.to_numpy(), f=sh, gap=gap, ctx=ctx, ctx50=ctx50, hm=hm))
        pg = pr[pr.gap].groupby(["b", "f"]).agg(ctx=("ctx", "any"), ctx50=("ctx50", "any"), hm=("hm", "any"))
        null.append(dict(gap_genes=int(gap.sum()), genome_function_gaps=len(pg), in_context=int(pg.ctx.sum()),
                         in_ctx50=int(pg.ctx50.sum()), with_module=int(pg.hm.sum())))
    N = pd.DataFrame(null)
    glob["null"] = dict(permutations=20, gap_genes_mean=float(N.gap_genes.mean()),
                        genome_function_gaps_mean=float(N.genome_function_gaps.mean()), in_context_mean=float(N.in_context.mean()),
                        in_context_q95=float(N.in_context.quantile(.95)), with_module_mean=float(N.with_module.mean()),
                        in_ctx50_mean=float(N.in_ctx50.mean()), in_ctx50_q95=float(N.in_ctx50.quantile(.95)),
                        in_ctx50_share_of_module_gaps_mean=float((N.in_ctx50 / N.with_module).mean()),
                        in_context_share_of_module_gaps_mean=float((N.in_context / N.with_module).mean()))
    glob["gaps"]["in_context_share_of_module_gaps"] = round(glob["gaps"]["in_context"] / max(1, glob["gaps"]["with_module"]), 4)
    glob["gaps"]["in_ctx50_share_of_module_gaps"] = round(glob["gaps"]["in_ctx50"] / max(1, glob["gaps"]["with_module"]), 4)

    # leads (KO, order)
    clade_n = bins.groupby("order").size()
    clade_comp = bins.groupby("order").completeness.median()
    bin_order = bins["order"]
    rows = []
    allbins = list(bins.index)
    clade_bins = {o: list(bin_order.index[bin_order == o]) for o in bin_order.unique()}
    pres_cache = {}

    def present_bins(f):
        if f not in pres_cache:
            pres_cache[f] = {b_ for b_ in allbins if not assess(b_, f, "")[4]}
        return pres_cache[f]
    for (f, o), g in S_.groupby(["ko_top", "order"]):
        gb = set(g.bin_id)
        others = [b_ for b_ in clade_bins.get(o, []) if b_ not in gb]
        pb = present_bins(f)
        rate = np.mean([b_ in pb for b_ in others]) if others else np.nan
        gapg = g[g.gap]
        gapb = set(gapg.bin_id)
        rows.append(dict(ko=f, ko_name=koname.get(f, ""), ec=";".join(ko_ec.get(f, [])), enzyme=bool(ko_ec.get(f)), clade=o,
                         phylum=g.phylum.iloc[0], labels=";".join(g.proposed_label_name.value_counts().index[:3]),
                         family_label=g.proposed_label_name.value_counts().index[0], genes=len(g), genomes=len(gb),
                         clade_genomes=int(clade_n.get(o, 0)), tier_T1=int((g.tier == "T1").sum()), tier_T2=int((g.tier == "T2").sum()),
                         tier_T2b=int((g.tier == "T2b").sum()), tier_contradicted=int((g.tier == "contradicted").sum()),
                         tier_rest=int((g.tier == "rest").sum()), support_median=round(float(g.vfrac.median()), 3),
                         ko_share_median=round(float(g.ko_share.median()), 3), gap_genomes=len(gapb),
                         gap_share=round(len(gapb) / len(gb), 3), gap_ko_genomes=g[g.gap_ko].bin_id.nunique(),
                         module_genomes=g[g.has_module].bin_id.nunique(),
                         in_context_genomes=gapg[gapg.in_context].bin_id.nunique(),
                         ctx50_genomes=gapg[(gapg.ctx_nother >= 2) & (gapg.ctx_frac >= 0.5)].bin_id.nunique(),
                         func_source=";".join(f"{k}:{v}" for k, v in g.func_source.value_counts().items()),
                         context_share_of_gap=round(gapg[gapg.has_module].in_context.mean(), 3) if gapg.has_module.any() else np.nan,
                         ctx_module=gapg.ctx_module.mode().iloc[0] if gapg.has_module.any() else "",
                         completeness_median_gap=round(float(gapg.completeness.median()), 1) if len(gapg) else np.nan,
                         clade_completeness_median=round(float(clade_comp.get(o, np.nan)), 1),
                         gap_genomes_ge90=gapg[gapg.completeness >= 90].bin_id.nunique(),
                         clade_rate_function=round(float(rate), 3) if others else np.nan, clade_others=len(others),
                         ilvfmawc_median=round(float(g.ilvfmawc.median()), 3), median_length=int(g.length.median()),
                         embedding_only_share=round(float(g.embedding_only.mean()), 3)))
    LT = pd.DataFrame(rows)
    pred = pd.read_csv("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/reference/kegg/kegg_predicates.tsv", sep="\t", comment="#",
                       header=None, names=["ko", "predicates", "evidence"]).set_index("ko").predicates.fillna("")
    HK = {"ribosomal_protein", "translation", "translation_factor", "transcription", "rna_polymerase", "sigma_factor", "replication",
          "dna_polymerase", "dna_repair", "recombinational_repair", "nucleotide_excision_repair", "base_excision_repair",
          "mismatch_repair", "topoisomerase", "trna_modification", "trna_synthetase", "rna_processing", "chaperone", "heat_shock",
          "hsp70", "hsp60", "small_hsp", "sec_pathway", "tat_pathway", "signal_recognition", "chromosome_partitioning",
          "cell_division", "divisome", "ligase_dna"}
    hkre = re.compile(r"signal peptidase|ribonuclease P|ribosom|tRNA|RNA polymerase|DNA polymerase|primase|elongation factor|"
                      r"initiation factor|preflagellin|flagell|pilus|pilin|replication|exosome|helicase|topoisomerase", re.I)
    LT["housekeeping"] = [bool(set(str(pred.get(k, "")).split(",")) & HK) or bool(hkre.search(n)) for k, n in zip(LT.ko, LT.ko_name)]
    LT["subunit_or_component"] = LT.ko_name.str.contains(r"subunit|component|chain", case=False, regex=True)
    LT["criteria_prefilter"] = (LT.enzyme & ~LT.housekeeping & (LT.ilvfmawc_median < 0.45) & LT.median_length.between(100, 600)
                                & (LT.gap_genomes >= 3) & (LT.tier_T1 >= 2))
    LT = LT.sort_values(["enzyme", "in_context_genomes", "ctx50_genomes", "gap_genomes", "tier_T1", "genes"], ascending=[False] * 6)
    LT.insert(0, "rank", np.arange(1, len(LT) + 1))
    LT.to_csv(H / "screen_leads.tsv", sep="\t", index=False)
    glob["leads"] = dict(total=len(LT), enzyme=int(LT.enzyme.sum()), with_gap=int((LT.gap_genomes > 0).sum()),
                         with_in_context=int((LT.in_context_genomes > 0).sum()))
    json.dump(glob, open(H / "global.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 30)
    print(json.dumps(glob, indent=1, default=str)[:4000])
    print(LT.head(40)[["rank", "ko", "ko_name", "clade", "family_label", "genes", "genomes", "clade_genomes", "tier_T1", "support_median",
                       "gap_genomes", "in_context_genomes", "ctx_module", "clade_rate_function", "ilvfmawc_median", "median_length",
                       "embedding_only_share"]].to_string(index=False))
    for k in ("K01808", "K03465", "K01641"):
        print(k, LT[LT.ko == k][["rank", "clade", "family_label", "genes", "gap_genomes", "in_context_genomes", "tier_T1"]].to_string(index=False))


if __name__ == "__main__":
    main()
