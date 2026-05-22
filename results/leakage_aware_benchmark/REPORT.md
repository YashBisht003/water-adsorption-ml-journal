# Leakage-Aware Adsorption ML Benchmark

## Dataset

- Clean rows: 563 / raw rows: 566
- Unique references: 189
- Unique adsorbents: 219
- Unique adsorbates: 16
- Lab source column used: None

Regime counts:

```text
Acidic_HighC0: 246
Acidic_LowC0: 231
Neutral_HighC0: 60
Neutral_LowC0: 26
```

## Feature Sets

- `full_capacity_process_model`: Includes adsorption capacity; suitable only as process-monitoring/descriptive model.
- `screening_safe_plus_ion`: No adsorption capacity; uses ion descriptors and adsorbent family.
- `screening_safe_plus_adsorbate_onehot`: No adsorption capacity; uses adsorbate identity as a categorical feature.
- `screening_safe_numeric_only`: No adsorption capacity and no ion identity; strictest screening baseline.

## Best Models By Split

| split             | feature_set                          | model       |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |
|:------------------|:-------------------------------------|:------------|----------:|---------:|------------:|-----------:|
| holdout_adsorbent | full_capacity_process_model          | xgboost     |    0.5768 |   0.1888 |     16.3797 |    11.1837 |
| holdout_adsorbent | full_capacity_process_model          | lightgbm    |    0.5243 |   0.2337 |     17.2686 |    11.526  |
| holdout_adsorbent | screening_safe_numeric_only          | xgboost     |    0.4886 |   0.087  |     18.4923 |    13.7132 |
| holdout_adsorbent | screening_safe_numeric_only          | lightgbm    |    0.4472 |   0.1249 |     19.1332 |    13.9633 |
| holdout_adsorbent | full_capacity_process_model          | extra_trees |    0.4092 |   0.4132 |     18.8263 |    12.5915 |
| holdout_reference | screening_safe_numeric_only          | extra_trees |    0.1117 |   0.2705 |     26.6838 |    20.4276 |
| holdout_reference | screening_safe_numeric_only          | lightgbm    |    0.0214 |   0.2065 |     27.9382 |    21.7024 |
| holdout_reference | full_capacity_process_model          | lightgbm    |   -0.014  |   0.2776 |     27.5677 |    20.6788 |
| holdout_reference | screening_safe_numeric_only          | xgboost     |   -0.0223 |   0.2586 |     28.4237 |    21.8563 |
| holdout_reference | screening_safe_plus_adsorbate_onehot | lightgbm    |   -0.0271 |   0.2629 |     28.4095 |    21.7342 |
| random_regime     | full_capacity_process_model          | xgboost     |    0.7975 |   0.0477 |     13.1606 |     8.3207 |
| random_regime     | full_capacity_process_model          | lightgbm    |    0.7953 |   0.0445 |     13.2409 |     8.2771 |
| random_regime     | full_capacity_process_model          | extra_trees |    0.7716 |   0.053  |     13.9955 |     9.0524 |
| random_regime     | screening_safe_plus_ion              | extra_trees |    0.7279 |   0.0662 |     15.2517 |    10.5444 |
| random_regime     | screening_safe_plus_adsorbate_onehot | xgboost     |    0.7278 |   0.0714 |     15.2307 |    10.7636 |

## Capacity-Including Model Summary

| model       | split             |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |   bias_mean |
|:------------|:------------------|----------:|---------:|------------:|-----------:|------------:|
| dummy_mean  | holdout_adsorbent |   -0.101  |   0.0956 |     27.2798 |    23.9322 |     -6.2923 |
| dummy_mean  | holdout_reference |   -0.1182 |   0.1419 |     29.5249 |    25.7608 |      0.7679 |
| dummy_mean  | random_regime     |   -0.0153 |   0.0172 |     29.7453 |    25.3152 |      0.4643 |
| extra_trees | holdout_adsorbent |    0.4092 |   0.4132 |     18.8263 |    12.5915 |     -1.912  |
| extra_trees | holdout_reference |   -0.1544 |   0.3857 |     29.6771 |    23.0093 |      8.1645 |
| extra_trees | random_regime     |    0.7716 |   0.053  |     13.9955 |     9.0524 |      0.4861 |
| lightgbm    | holdout_adsorbent |    0.5243 |   0.2337 |     17.2686 |    11.526  |     -2.3856 |
| lightgbm    | holdout_reference |   -0.014  |   0.2776 |     27.5677 |    20.6788 |      6.8902 |
| lightgbm    | random_regime     |    0.7953 |   0.0445 |     13.2409 |     8.2771 |      0.6043 |
| xgboost     | holdout_adsorbent |    0.5768 |   0.1888 |     16.3797 |    11.1837 |     -2.4902 |
| xgboost     | holdout_reference |   -0.0553 |   0.2302 |     28.3369 |    21.8395 |      5.7137 |
| xgboost     | random_regime     |    0.7975 |   0.0477 |     13.1606 |     8.3207 |      0.1933 |

## Screening Model Summary

| feature_set                          | model       | split             |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |
|:-------------------------------------|:------------|:------------------|----------:|---------:|------------:|-----------:|
| screening_safe_numeric_only          | dummy_mean  | holdout_adsorbent |   -0.101  |   0.0956 |     27.2798 |    23.9322 |
| screening_safe_numeric_only          | dummy_mean  | holdout_reference |   -0.1182 |   0.1419 |     29.5249 |    25.7608 |
| screening_safe_numeric_only          | dummy_mean  | random_regime     |   -0.0153 |   0.0172 |     29.7453 |    25.3152 |
| screening_safe_numeric_only          | extra_trees | holdout_adsorbent |    0.3323 |   0.125  |     21.2578 |    16.2968 |
| screening_safe_numeric_only          | extra_trees | holdout_reference |    0.1117 |   0.2705 |     26.6838 |    20.4276 |
| screening_safe_numeric_only          | extra_trees | random_regime     |    0.3957 |   0.1438 |     22.8073 |    16.6165 |
| screening_safe_numeric_only          | lightgbm    | holdout_adsorbent |    0.4472 |   0.1249 |     19.1332 |    13.9633 |
| screening_safe_numeric_only          | lightgbm    | holdout_reference |    0.0214 |   0.2065 |     27.9382 |    21.7024 |
| screening_safe_numeric_only          | lightgbm    | random_regime     |    0.6061 |   0.0648 |     18.5435 |    13.0231 |
| screening_safe_numeric_only          | xgboost     | holdout_adsorbent |    0.4886 |   0.087  |     18.4923 |    13.7132 |
| screening_safe_numeric_only          | xgboost     | holdout_reference |   -0.0223 |   0.2586 |     28.4237 |    21.8563 |
| screening_safe_numeric_only          | xgboost     | random_regime     |    0.6863 |   0.0623 |     16.4284 |    12.034  |
| screening_safe_plus_adsorbate_onehot | dummy_mean  | holdout_adsorbent |   -0.101  |   0.0956 |     27.2798 |    23.9322 |
| screening_safe_plus_adsorbate_onehot | dummy_mean  | holdout_reference |   -0.1182 |   0.1419 |     29.5249 |    25.7608 |
| screening_safe_plus_adsorbate_onehot | dummy_mean  | random_regime     |   -0.0153 |   0.0172 |     29.7453 |    25.3152 |
| screening_safe_plus_adsorbate_onehot | extra_trees | holdout_adsorbent |    0.3248 |   0.3853 |     20.5187 |    14.3243 |
| screening_safe_plus_adsorbate_onehot | extra_trees | holdout_reference |   -0.2235 |   0.3378 |     30.8289 |    23.7127 |
| screening_safe_plus_adsorbate_onehot | extra_trees | random_regime     |    0.7194 |   0.0527 |     15.541  |    10.7137 |
| screening_safe_plus_adsorbate_onehot | lightgbm    | holdout_adsorbent |    0.3247 |   0.4077 |     20.4965 |    14.6236 |
| screening_safe_plus_adsorbate_onehot | lightgbm    | holdout_reference |   -0.0271 |   0.2629 |     28.4095 |    21.7342 |
| screening_safe_plus_adsorbate_onehot | lightgbm    | random_regime     |    0.7223 |   0.0538 |     15.4398 |    10.571  |
| screening_safe_plus_adsorbate_onehot | xgboost     | holdout_adsorbent |    0.356  |   0.3248 |     20.1698 |    14.4089 |
| screening_safe_plus_adsorbate_onehot | xgboost     | holdout_reference |   -0.1032 |   0.2614 |     29.4427 |    22.2937 |
| screening_safe_plus_adsorbate_onehot | xgboost     | random_regime     |    0.7278 |   0.0714 |     15.2307 |    10.7636 |
| screening_safe_plus_ion              | dummy_mean  | holdout_adsorbent |   -0.101  |   0.0956 |     27.2798 |    23.9322 |
| screening_safe_plus_ion              | dummy_mean  | holdout_reference |   -0.1182 |   0.1419 |     29.5249 |    25.7608 |
| screening_safe_plus_ion              | dummy_mean  | random_regime     |   -0.0153 |   0.0172 |     29.7453 |    25.3152 |
| screening_safe_plus_ion              | extra_trees | holdout_adsorbent |    0.3371 |   0.3526 |     20.3237 |    14.4402 |
| screening_safe_plus_ion              | extra_trees | holdout_reference |   -0.0689 |   0.139  |     29.1274 |    22.2733 |
| screening_safe_plus_ion              | extra_trees | random_regime     |    0.7279 |   0.0662 |     15.2517 |    10.5444 |
| screening_safe_plus_ion              | lightgbm    | holdout_adsorbent |    0.3085 |   0.4076 |     20.7783 |    14.8376 |
| screening_safe_plus_ion              | lightgbm    | holdout_reference |   -0.0322 |   0.2488 |     28.4976 |    21.5659 |
| screening_safe_plus_ion              | lightgbm    | random_regime     |    0.71   |   0.0652 |     15.7542 |    10.908  |
| screening_safe_plus_ion              | xgboost     | holdout_adsorbent |    0.3768 |   0.3077 |     19.8691 |    14.2952 |
| screening_safe_plus_ion              | xgboost     | holdout_reference |   -0.1128 |   0.2502 |     29.5788 |    22.2612 |
| screening_safe_plus_ion              | xgboost     | random_regime     |    0.7278 |   0.0671 |     15.2595 |    10.7963 |

## Leave-One-Adsorbate-Out Summary

| feature_set                          | model       |   n_groups |   R2_mean |   R2_median |   RMSE_mean |   MAE_mean |
|:-------------------------------------|:------------|-----------:|----------:|------------:|------------:|-----------:|
| full_capacity_process_model          | lightgbm    |          7 |   -0.7245 |     -0.479  |     27.2958 |    22.5502 |
| full_capacity_process_model          | extra_trees |          7 |   -0.7891 |     -0.09   |     28.0121 |    23.1106 |
| screening_safe_plus_ion              | extra_trees |          7 |   -0.7958 |     -0.1402 |     27.1017 |    23.186  |
| full_capacity_process_model          | xgboost     |          7 |   -0.8771 |     -0.5205 |     28.5958 |    23.5913 |
| screening_safe_plus_adsorbate_onehot | extra_trees |          7 |   -0.945  |     -0.3353 |     29.2616 |    24.0798 |
| screening_safe_plus_ion              | lightgbm    |          7 |   -1.1434 |     -0.0828 |     29.9681 |    25.386  |
| screening_safe_plus_ion              | xgboost     |          7 |   -1.7189 |     -0.1888 |     32.0315 |    27.1846 |
| screening_safe_plus_adsorbate_onehot | lightgbm    |          7 |   -1.7897 |     -0.2321 |     33.7176 |    28.1549 |
| screening_safe_numeric_only          | extra_trees |          7 |   -1.917  |     -0.2882 |     33.7119 |    28.2019 |
| screening_safe_plus_adsorbate_onehot | xgboost     |          7 |   -1.9916 |     -0.4201 |     34.8093 |    28.8654 |
| screening_safe_numeric_only          | lightgbm    |          7 |   -2.0898 |     -0.3539 |     35.1196 |    29.3532 |
| screening_safe_numeric_only          | xgboost     |          7 |   -2.1766 |     -0.3377 |     35.348  |    29.6726 |

## Interpretation

- `random_regime` estimates row-level interpolation within the merged dataset.
- `holdout_reference` estimates generalization to unseen literature/lab sources; this is the most important reviewer-facing stress test available without explicit lab labels.
- `holdout_adsorbent` estimates generalization to unseen adsorbent names.
- Capacity-including models should not be framed as pre-experiment screening models.
- Screening models are more scientifically defensible, but lower accuracy is expected.

## Missing Lab Split

No usable `Lab_Source` column was found. Add a column with values such as `Own_Lab` and `External_Lab`, then rerun with:

```bash
python scripts/run_leakage_aware_benchmark.py --lab-source-column Lab_Source
```
