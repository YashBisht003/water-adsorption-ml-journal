# Dual-Track Manuscript Strategy

## Core Idea

Use both stories, but keep the claims separate:

- **Track A: high-R2 source-specific modeling.** This demonstrates that adsorption RE can be predicted very accurately when the task is interpolation within a lab/source/system.
- **Track B: leakage-aware global validation.** This demonstrates the real generalization boundary across unseen references, adsorbents, and adsorbates.

This is stronger than competing only on a single row-wise R2 because it explains why many papers report near-0.95 while broader models fail under stricter validation.

## Dataset Snapshot

- Clean rows: 563
- Unique references: 189
- Unique adsorbents: 219
- Unique adsorbates: 16
- Minimum rows for source-specific CV in Track A: 20

## Track A: Source-Specific High-R2 Models

| Ref_group                                                                  |   n_rows | feature_set                 |      R2 |    RMSE |     MAE |
|:---------------------------------------------------------------------------|---------:|:----------------------------|--------:|--------:|--------:|
| (Shen et al., 2017b)                                                       |       65 | full_capacity_process_model |  0.9916 |  3.2035 |  2.334  |
| (Gao et al., 2019)                                                         |       48 | full_capacity_process_model |  0.9735 |  2.9718 |  2.1381 |
| (Zama et al., 2017)                                                        |       91 | full_capacity_process_model |  0.9601 |  7.8421 |  3.5562 |
| (Cui et al., 2016a)                                                        |       24 | full_capacity_process_model |  0.8936 |  6.6548 |  4.5576 |
| https://www.sciencedirect.com/science/article/pii/S2213343721006655        |       33 | full_capacity_process_model | -0.0732 |  8.5151 |  6.2749 |
| https://www.sciencedirect.com/science/article/pii/S0045653519308616#bib150 |       26 | full_capacity_process_model | -1.2003 | 24.2753 | 16.9909 |

Interpretation: these results are suitable for a high-accuracy model claim only within a known source/lab/system.

## Track B: Global Leakage-Aware Validation

| split             | feature_set                 | model       |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |
|:------------------|:----------------------------|:------------|----------:|---------:|------------:|-----------:|
| holdout_adsorbent | full_capacity_process_model | xgboost     |    0.5768 |   0.1888 |     16.3797 |    11.1837 |
| holdout_adsorbent | full_capacity_process_model | lightgbm    |    0.5243 |   0.2337 |     17.2686 |    11.526  |
| holdout_adsorbent | screening_safe_numeric_only | xgboost     |    0.4886 |   0.087  |     18.4923 |    13.7132 |
| holdout_reference | screening_safe_numeric_only | extra_trees |    0.1117 |   0.2705 |     26.6838 |    20.4276 |
| holdout_reference | screening_safe_numeric_only | lightgbm    |    0.0214 |   0.2065 |     27.9382 |    21.7024 |
| holdout_reference | full_capacity_process_model | lightgbm    |   -0.014  |   0.2776 |     27.5677 |    20.6788 |
| random_regime     | full_capacity_process_model | xgboost     |    0.7975 |   0.0477 |     13.1606 |     8.3207 |
| random_regime     | full_capacity_process_model | lightgbm    |    0.7953 |   0.0445 |     13.2409 |     8.2771 |
| random_regime     | full_capacity_process_model | extra_trees |    0.7716 |   0.053  |     13.9955 |     9.0524 |

Interpretation: row-wise interpolation and source-held-out generalization are different scientific tasks. The paper should report both.

## Proposed Paper Framing

Suggested title:

> Source-aware and leakage-aware machine learning for adsorption removal efficiency: high-accuracy interpolation versus cross-source generalization

Suggested contribution bullets:

- Compile and clean a heterogeneous heavy-metal adsorption RE dataset.
- Show that source-specific models can reach near-0.95 or higher R2.
- Show that global models degrade under reference/source-held-out validation.
- Separate capacity-including process-monitoring models from pre-experiment screening models.
- Provide a lab-source validation protocol ready for own-lab versus external-lab experiments.

## What We Need From The Two-Lab Data

The current CSV has no explicit `Lab_Source` column. Fill one of these files:

```text
results/dual_track/lab_source_by_reference_template.csv
results/dual_track/data_with_lab_source_template.csv
```

Recommended labels:

```text
Own_Lab
External_Lab
Literature
```

Then run:

```bash
python scripts/run_leakage_aware_benchmark.py --lab-source-column Lab_Source
python scripts/run_dual_track_analysis.py --lab-source-column Lab_Source
```

## Journal Strategy

- Do not claim universal R2 near 0.95 on the heterogeneous dataset.
- Do claim near-0.95 source-specific prediction where supported.
- Make the novel JECE angle the difference between interpolation and external/source generalization.
- If own-lab to external-lab validation is reasonable, promote that as the main result.
- If lab transfer is weak, frame it as a calibrated applicability-domain method rather than a failed model.
