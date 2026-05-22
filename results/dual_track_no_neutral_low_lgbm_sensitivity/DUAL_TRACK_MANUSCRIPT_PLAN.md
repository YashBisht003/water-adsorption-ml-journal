# Dual-Track Manuscript Strategy

## Core Idea

Use both stories, but keep the claims separate:

- **Track A: high-R2 source-specific modeling.** This demonstrates that adsorption RE can be predicted very accurately when the task is interpolation within a lab/source/system.
- **Track B: leakage-aware global validation.** This demonstrates the real generalization boundary across unseen references, adsorbents, and adsorbates.

This is stronger than competing only on a single row-wise R2 because it explains why many papers report near-0.95 while broader models fail under stricter validation.

## Dataset Snapshot

- Clean rows: 563
- Excluded primary-domain regimes: Neutral_LowC0
- Rows before regime exclusion: 563
- Primary modeling rows after exclusion: 537
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
| https://www.sciencedirect.com/science/article/pii/S2213343721006655        |       30 | full_capacity_process_model | -0.5584 | 10.4543 |  6.8807 |
| https://www.sciencedirect.com/science/article/pii/S0045653519308616#bib150 |       25 | full_capacity_process_model | -1.6159 | 26.699  | 18.8601 |

Interpretation: these results are suitable for a high-accuracy model claim only within a known source/lab/system.

## Track B: Global Leakage-Aware Validation

| split             | feature_set                 | model                 |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |
|:------------------|:----------------------------|:----------------------|----------:|---------:|------------:|-----------:|
| holdout_adsorbent | screening_safe_numeric_only | lightgbm_deeper       |    0.5323 |   0.1568 |     19.3503 |    14.5511 |
| holdout_adsorbent | full_capacity_process_model | lightgbm_conservative |    0.5258 |   0.2098 |     19.1854 |    12.9996 |
| holdout_adsorbent | screening_safe_numeric_only | xgboost               |    0.5199 |   0.1644 |     19.6133 |    15.146  |
| holdout_reference | full_capacity_process_model | lightgbm_deeper       |    0.3296 |   0.1393 |     22.8804 |    17.6665 |
| holdout_reference | screening_safe_numeric_only | extra_trees           |    0.3054 |   0.158  |     23.5441 |    18.6703 |
| holdout_reference | full_capacity_process_model | lightgbm              |    0.282  |   0.0736 |     23.7831 |    18.6679 |
| random_regime     | full_capacity_process_model | lightgbm              |    0.8064 |   0.0474 |     12.9356 |     8.0805 |
| random_regime     | full_capacity_process_model | lightgbm_deeper       |    0.8026 |   0.0445 |     13.0817 |     7.7863 |
| random_regime     | full_capacity_process_model | xgboost               |    0.7985 |   0.0453 |     13.2206 |     8.436  |

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
results/dual_track_no_neutral_low_lgbm_sensitivity/lab_source_by_reference_template.csv
results/dual_track_no_neutral_low_lgbm_sensitivity/data_with_lab_source_template.csv
```

Recommended labels:

```text
Own_Lab
External_Lab
Literature
```

Then run:

```bash
python scripts/run_leakage_aware_benchmark.py --data path/to/labelled_data.csv --out-dir results/leakage_aware_benchmark_no_neutral_low --lab-source-column Lab_Source --exclude-regimes Neutral_LowC0
python scripts/run_dual_track_analysis.py --data path/to/labelled_data.csv --out-dir results/dual_track_no_neutral_low --benchmark-dir results/leakage_aware_benchmark_no_neutral_low --lab-source-column Lab_Source --exclude-regimes Neutral_LowC0
```

## Journal Strategy

- Do not claim universal R2 near 0.95 on the heterogeneous dataset.
- Do claim near-0.95 source-specific prediction where supported.
- Make the novel JECE angle the difference between interpolation and external/source generalization.
- If own-lab to external-lab validation is reasonable, promote that as the main result.
- If lab transfer is weak, frame it as a calibrated applicability-domain method rather than a failed model.
