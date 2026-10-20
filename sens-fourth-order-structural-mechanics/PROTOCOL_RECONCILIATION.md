# Reproducibility protocol reconciliation

This repository is a public, independently reconstructed reference implementation based on the mathematical method described in the SENS manuscript. The original historical training scripts were not present among the files available for the revision.

There is a protocol mismatch that must be resolved before claiming exact reproduction of every numerical value in the current manuscript:

| Item | Current manuscript text | Public reference configuration |
|---|---|---|
| Hidden-layer width | 128 neurons per layer | 64 neurons per layer |
| Training budget | 10,000 epochs | 1,200 epochs |
| Repeated runs | 10 independent runs | Fixed seeds 17, 29, and 41 |
| Refinement points | N = 25, 50, 100, 200, 400, 800, 1600 | N = 25, 50, 100, 200, 400, 800, 1600 |

The archive's `reproduce_experiments.py` calculates fresh output tables and figures for the settings in `config.json`; it does not hard-code the reported error values. It should not be represented as recovering the exact source of earlier results, nor should its fresh outputs be described as matching the previous tables unless they have been run and the manuscript has been reconciled against them.

For a fully reproducibility-complete journal revision, either recover and deposit the historical scripts and settings that generated the currently reported numerical results, or adopt this archive as the final implementation, execute the full experiments, and update every affected manuscript value, table, and figure to those new outputs.