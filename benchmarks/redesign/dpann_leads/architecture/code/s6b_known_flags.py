#!/usr/bin/env python3
"""Step 6b: known-form decision as the brief states it (carriers show the same full-length architecture at a similar rate:
carrier rate >= 0.5 x labeled rate -> known, dropped). The absolute count is kept as a flag: arch_annotated_in_carriers
= carriers with the architecture >= max(5, 0.5 x group genes) (the architecture already exists among GA-annotated DPANN
proteins, at a lower rate). Rewrites data/known_form.parquet columns known_form / arch_annotated_in_carriers."""
import pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
S = pd.read_parquet("data/known_form.parquet")
S["rate_ratio"] = S.carrier_arch_rate / S.labeled_arch_rate
S["known_form"] = S.rate_ratio >= 0.5
S["arch_annotated_in_carriers"] = S.carriers_fl >= (0.5 * S.genes).clip(lower=5)
S.to_parquet("data/known_form.parquet", index=False)
print("conserved groups", len(S), "| known form (rate rule)", int(S.known_form.sum()), "| pass", int((~S.known_form).sum()),
      "| of which architecture annotated in >= max(5, half as many) carriers", int((~S.known_form & S.arch_annotated_in_carriers).sum()))
