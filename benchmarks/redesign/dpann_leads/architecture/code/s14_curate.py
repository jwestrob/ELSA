#!/usr/bin/env python3
"""Step 14: manual curation of the 60 groups that pass every automated control (candidates_ranked.tsv, pass_all), from the
extra-region hits (weak full-Pfam, CDD, family HHsearch) and the locus prints of s12_inspect.py. Categories:
  JOIN      extra region carries a domain of known function from a different family (lead candidates)
  DUP       the label family twice in one chain
  SAMEFAM   the extra region is the rest of a larger known protein family (the label names a part); extension dark
  KNOWN     architecture of characterized relatives (curated records) or of a housekeeping protein
  DARK      extra region has no hit (weak Pfam i-E <= 1e-3, CDD, HHsearch) in >= 50% of members
  REG       the extra region or the protein is a DNA-binding regulator (HTH)
  SURF      surface / adhesin-like or > 1,000 aa
  MGE       intein, integrase, transposase
Writes candidates_curated.tsv and prints counts per category."""
import pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
C = {
 "HLC90278.1": ("JOIN", "NTP_transf_3 (MobA/LicC-like NTase) + C-terminal APH/choline-kinase-like kinase; LicCA-type; lead L1"),
 "MBN2421503.1": ("JOIN", "NTP_transf_5 (uncharacterized NTase) + C-terminal Thymidylate_kin; lead L2"),
 "HIG96325.1": ("JOIN", "same architecture as MBN2421503.1 (UBA12501); merged into lead L2"),
 "MBS3154103.1": ("DUP", "two cupin domains (MannoseP_isomer x2); 2 genes from L1 in the same locus; lead L3"),
 "JAJZLM010000010.1_19": ("DUP", "V4R twice in one chain, N-terminal 114 aa dark; regulatory ligand-binding domain"),
 "DAHXFS010000039.1_8": ("DUP", "RE_TdeIII/HinfI-like restriction endonuclease twice in one chain (4 genomes)"),
 "MBT6744359.1": ("SAMEFAM", "Fer4_12 label covers the radical SAM core of a KamA-like protein (CDD KamA 53-388); C-terminal 143 aa dark; lead L4"),
 "MBI2650380.1": ("SAMEFAM", "M29 aminopeptidase (AmpS) with ~160-aa N-terminal extension; 13/28 already carry KO K19689; lead L5"),
 "DAHXML010000012.1_17": ("KNOWN", "EndoQ-like PHP endonuclease (CDD PfuEndoQ-like 94/96; characterized EndoQ); 151 GA carriers share it"),
 "DAIDNG010000020.1_18": ("KNOWN", "2x[4Fe-4S] ferredoxin, carrier length; housekeeping"),
 "JALTYM010000035.1_12": ("SURF", "NPCBM/Cep192/COG1470 surface-protein module"),
 "DAHVZR010000003.1_9": ("KNOWN", "AlaX-like (CDD AlaS), shorter than carriers"),
 "OGJ20652.1": ("REG", "HTH + Pol-beta NTase; 1,797 GA carriers share it"),
 "JBGOZN010000002.1_120": ("KNOWN", "DHH-DHHA1 (RecJ-like / COG2404)"),
 "JAKASF010000010.1_8": ("KNOWN", "PEP synthase-like (CDD PRK06241), shorter than carriers"),
 "MDO8642963.1": ("REG", "CarboxypepD_reg + COG2512 regulator"),
 "MCK4997697.1": ("SAMEFAM", "FixC-like flavoprotein; the DAO hit is the FAD-binding motif"),
 "DAHWQL010000010.1_15": ("KNOWN", "HerA ATPase (DUF87 + HerA_C)"),
 "JALTYG010000051.1_5": ("REG", "HTH_24 + COG1510"),
 "MDD5181866.1": ("MGE", "intein-containing EndoQ-like protein"),
 "MBI4152374.1": ("REG", "Lrp/AsnC + N-terminal extension"),
 "MCX6708378.1": ("MGE", "LAGLIDADG + C-terminal HTH"),
 "SBBR01000109.1_15": ("REG", "Lrp/AsnC + HTH"),
 "MBL7050915.1": ("REG", "HTH_17 + dark C-terminal region"),
 "MAE49949.1": ("REG", "NUDIX + N-terminal winged helix (NrtR-like); 198 GA carriers"),
 "MBD3260247.1": ("REG", "HTH_5 + COG1510"),
 "MBI2079504.1": ("KNOWN", "HEAT_2 + PRK06461; 95 GA carriers"),
 "MBR9699191.1": ("DARK", "Hexapep_2 + C-terminal region with sporadic NRPS-like CDD (2/4)"),
 "MBI1978601.1": ("SAMEFAM", "QueC/WbuX amidotransferase family (same ATP-PPase superfamily)"),
 "MEM3215593.1": ("KNOWN", "AlaX-like"),
 "MEM4097557.1": ("DARK", "MazE/AbrB N-terminal + 242-aa C-terminal region without hits (196 genomes)"),
 "MBT7025105.1": ("REG", "HTH_3 + extension"),
 "JAFCFQ010000002.1_3": ("DARK", "AP_endonuc_2 TIM barrel + 379-aa N-terminal region without hits (52 genomes, 28 genera)"),
 "VVB99399.1": ("DARK", "HATPase_c_2 + 103-aa N-terminal region without hits (47 genomes, 35 genera)"),
 "MEK6846899.1": ("KNOWN", "AlgX-like SGNH acetyltransferase; 212 GA carriers"),
 "MEM5791781.1": ("DARK", "V4R + 192-aa C-terminal region (family HHblits to MopR sensor domain only); no DNA-binding or enzyme domain"),
 "WAPD01000098.1_5": ("REG", "HTH_17 + extension"),
 "OYT42662.1": ("DARK", "AP_endonuc_2 + 320-aa C-terminal region without hits (23 genomes)"),
 "MFA4855623.1": ("KNOWN", "class III RNR NrdD (ATP-cone + RNR_III)"),
 "MBN1544362.1": ("REG", "HTH_27 + extension"),
 "MBS3059387.1": ("KNOWN", "B12-binding radical SAM; 332 GA carriers"),
 "MBL7055569.1": ("REG", "four sigma70_r4-like HTH repeats"),
 "CAXBYQ010000001.1_425": ("SURF", "2,382-aa Ig-like adhesin"),
 "MBU4245885.1": ("REG", "HTH_17 + extension"),
 "MBU0586191.1": ("DARK", "SLT + 494-aa N-terminal region (761 aa)"),
 "MBU2441235.1": ("DARK", "APH kinase + 242-aa N-terminal region without hits (7 genomes)"),
 "VVB59871.1": ("REG", "MarR + extension"),
 "HLD89008.1": ("REG", "HTH_24 + coiled-coil extension"),
 "MBN2422620.1": ("MGE", "PDDEXK + InsA zinc ribbon (transposase-like)"),
 "PEYR01000215.1_4": ("DARK", "GNAT (Acetyltransf_9) + 149-aa N-terminal region (6 genomes)"),
 "CAXGWQ010000011.1_9": ("REG", "DUF6293_C + MarR-like winged helix"),
 "DATDXW010000028.1_2": ("KNOWN", "Metallophos_2; 153 GA carriers"),
 "MFA6023114.1": ("MGE", "phage integrase"),
 "JAHFOI010000001.1_65": ("DARK", "cytidine deaminase + 160-aa N-terminal region (4 genomes)"),
 "MBA4405220.1": ("REG", "PadR + extension"),
 "HLD01122.1": ("DARK", "HD phosphohydrolase + 187-aa C-terminal region (3 genomes)"),
 "MBI4140302.1": ("DARK", "SGNH (Lipase_GDSL_2) + 202-aa N-terminal region (3 genomes)"),
 "MBS1266272.1": ("DARK", "GNAT (Acetyltransf_1) + 204-aa N-terminal region (3 genomes)"),
 "MDD5054600.1": ("DARK", "VWA_2 + 337-aa N-terminal region (3 genomes)"),
 "MDO8508639.1": ("DARK", "Amidohydro_2 + 95-aa N-terminal region (3 genomes)"),
}
T = pd.read_csv("candidates_ranked.tsv", sep="\t")
P = T[T.pass_all].copy()
missing = set(P.rep) - set(C)
assert not missing, missing
P["category"] = P.rep.map(lambda r: C[r][0]); P["curation_note"] = P.rep.map(lambda r: C[r][1])
order = {"JOIN": 0, "DUP": 1, "SAMEFAM": 2, "DARK": 3, "KNOWN": 4, "REG": 5, "MGE": 6, "SURF": 7}
P["o"] = P.category.map(order)
P = P.sort_values(["o", "genomes"], ascending=[True, False]).drop(columns="o")
P.to_csv("candidates_curated.tsv", sep="\t", index=False)
print(P.category.value_counts().to_string())
print(P[["rep", "label", "category", "genes", "genomes", "genera"]].head(12).to_string(index=False))
