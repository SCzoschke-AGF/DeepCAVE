These immutable fixtures contain only archived configurations, KDE scores, geometry,
and expected selections. Production code does not import or read diagnostic scripts.
Each archive includes the SHA-256 hashes and paths of its original source artifacts.

`cat38.json.gz` preserves all 12,102 configurations from the frozen augmented
CatBoost pool, the numerical ConfigSpace, and the archived equal-width 14/13
selection. It verifies exact configuration identities as well as score and RMS
distance aggregates. No models are fitted for this regression.

`xgb.json.gz` preserves the eligible frozen global/equal-width proposal arms for
studies 30 and 32 and their archived retrospective split selections. Eligibility
was frozen from the existing prospective outcome export; outcome values are not
inputs to the selection routine. These limited proposal arms validate the shared
selection primitive, not the new 1,000-per-bin supply requirement. Supply is tested
separately against synthetic, unrestricted and constrained ConfigSpaces.
