# Bundled Data

This folder contains the input data used by the APDTM comparison.

- `dataset_v8.csv`  
  CASH result table with log features, algorithm/configuration parameters, and
  measured quality values (`fitness`, `precision`, `generalization`,
  `simplicity`). This is used to evaluate APDTM's recommendations and to run
  the CASH-common/CASH-full LOLO comparison.

- `raw.zip`  
  Raw real event logs in XES/XES.GZ format. This is used to extract APDTM's
  APDTM meta-features and to compute APDTM discovery metrics for the added real
  CASH logs.

The scripts first look for these bundled files. If they are missing, they fall
back to the same files in the parent project root.
