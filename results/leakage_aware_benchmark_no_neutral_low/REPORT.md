# Leakage-Aware Adsorption ML Benchmark

## Dataset

- Clean rows: 563 / raw rows: 566
- Unique references: 189
- Unique adsorbents: 219
- Unique adsorbates: 16
- Lab source column used: None
- Excluded regimes before modeling: Neutral_LowC0
- Rows before regime exclusion: 563
- Rows after regime exclusion: 537

Regime counts:

```text
Acidic_HighC0: 246
Acidic_LowC0: 231
Neutral_HighC0: 60
```

## Feature Sets

- `full_capacity_process_model`: Includes adsorption capacity; suitable only as process-monitoring/descriptive model.
- `screening_safe_plus_ion`: No adsorption capacity; uses ion descriptors and adsorbent family.
- `screening_safe_plus_adsorbate_onehot`: No adsorption capacity; uses adsorbate identity as a categorical feature.
- `screening_safe_numeric_only`: No adsorption capacity and no ion identity; strictest screening baseline.

## Best Models By Split

| split             | feature_set                          | model       |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |
|:------------------|:-------------------------------------|:------------|----------:|---------:|------------:|-----------:|
| holdout_adsorbent | screening_safe_numeric_only          | xgboost     |    0.5199 |   0.1644 |     19.6133 |    15.146  |
| holdout_adsorbent | screening_safe_numeric_only          | lightgbm    |    0.5002 |   0.1224 |     20.0972 |    15.1926 |
| holdout_adsorbent | screening_safe_numeric_only          | extra_trees |    0.4741 |   0.1852 |     20.5337 |    16.8631 |
| holdout_adsorbent | full_capacity_process_model          | xgboost     |    0.4385 |   0.2861 |     20.7583 |    13.9363 |
| holdout_adsorbent | full_capacity_process_model          | lightgbm    |    0.4364 |   0.2922 |     20.7584 |    13.7177 |
| holdout_reference | screening_safe_numeric_only          | extra_trees |    0.3054 |   0.158  |     23.5441 |    18.6703 |
| holdout_reference | full_capacity_process_model          | lightgbm    |    0.282  |   0.0736 |     23.7831 |    18.6679 |
| holdout_reference | full_capacity_process_model          | xgboost     |    0.1678 |   0.093  |     25.6616 |    19.5265 |
| holdout_reference | screening_safe_plus_ion              | extra_trees |    0.1307 |   0.1549 |     26.2957 |    20.66   |
| holdout_reference | screening_safe_numeric_only          | xgboost     |    0.0945 |   0.3673 |     26.8764 |    20.8367 |
| random_regime     | full_capacity_process_model          | lightgbm    |    0.8064 |   0.0474 |     12.9356 |     8.0805 |
| random_regime     | full_capacity_process_model          | xgboost     |    0.7985 |   0.0453 |     13.2206 |     8.436  |
| random_regime     | full_capacity_process_model          | extra_trees |    0.773  |   0.0551 |     14.0172 |     9.015  |
| random_regime     | screening_safe_plus_adsorbate_onehot | xgboost     |    0.7283 |   0.0716 |     15.2905 |    10.839  |
| random_regime     | screening_safe_plus_ion              | xgboost     |    0.7265 |   0.0745 |     15.3375 |    10.8817 |

## Capacity-Including Model Summary

| model       | split             |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |   bias_mean |
|:------------|:------------------|----------:|---------:|------------:|-----------:|------------:|
| dummy_mean  | holdout_adsorbent |   -0.0296 |   0.0318 |     28.9773 |    25.859  |     -0.9986 |
| dummy_mean  | holdout_reference |   -0.0925 |   0.0865 |     29.4447 |    25.0407 |      3.4043 |
| dummy_mean  | random_regime     |   -0.014  |   0.0136 |     29.853  |    25.3624 |      0.7708 |
| extra_trees | holdout_adsorbent |    0.3636 |   0.3911 |     21.8578 |    14.9645 |      3.2179 |
| extra_trees | holdout_reference |    0.02   |   0.2867 |     28.0401 |    21.0126 |     10.8895 |
| extra_trees | random_regime     |    0.773  |   0.0551 |     14.0172 |     9.015  |      0.8303 |
| lightgbm    | holdout_adsorbent |    0.4364 |   0.2922 |     20.7584 |    13.7177 |      1.3887 |
| lightgbm    | holdout_reference |    0.282  |   0.0736 |     23.7831 |    18.6679 |      5.4654 |
| lightgbm    | random_regime     |    0.8064 |   0.0474 |     12.9356 |     8.0805 |      0.7392 |
| xgboost     | holdout_adsorbent |    0.4385 |   0.2861 |     20.7583 |    13.9363 |      1.501  |
| xgboost     | holdout_reference |    0.1678 |   0.093  |     25.6616 |    19.5265 |      6.4665 |
| xgboost     | random_regime     |    0.7985 |   0.0453 |     13.2206 |     8.436  |      0.3579 |

## Screening Model Summary

| feature_set                          | model       | split             |   R2_mean |   R2_std |   RMSE_mean |   MAE_mean |
|:-------------------------------------|:------------|:------------------|----------:|---------:|------------:|-----------:|
| screening_safe_numeric_only          | dummy_mean  | holdout_adsorbent |   -0.0296 |   0.0318 |     28.9773 |    25.859  |
| screening_safe_numeric_only          | dummy_mean  | holdout_reference |   -0.0925 |   0.0865 |     29.4447 |    25.0407 |
| screening_safe_numeric_only          | dummy_mean  | random_regime     |   -0.014  |   0.0136 |     29.853  |    25.3624 |
| screening_safe_numeric_only          | extra_trees | holdout_adsorbent |    0.4741 |   0.1852 |     20.5337 |    16.8631 |
| screening_safe_numeric_only          | extra_trees | holdout_reference |    0.3054 |   0.158  |     23.5441 |    18.6703 |
| screening_safe_numeric_only          | extra_trees | random_regime     |    0.3806 |   0.1518 |     23.163  |    16.9098 |
| screening_safe_numeric_only          | lightgbm    | holdout_adsorbent |    0.5002 |   0.1224 |     20.0972 |    15.1926 |
| screening_safe_numeric_only          | lightgbm    | holdout_reference |    0.0068 |   0.4683 |     28.0366 |    21.6373 |
| screening_safe_numeric_only          | lightgbm    | random_regime     |    0.6119 |   0.0775 |     18.47   |    12.9397 |
| screening_safe_numeric_only          | xgboost     | holdout_adsorbent |    0.5199 |   0.1644 |     19.6133 |    15.146  |
| screening_safe_numeric_only          | xgboost     | holdout_reference |    0.0945 |   0.3673 |     26.8764 |    20.8367 |
| screening_safe_numeric_only          | xgboost     | random_regime     |    0.6774 |   0.0653 |     16.7285 |    12.2828 |
| screening_safe_plus_adsorbate_onehot | dummy_mean  | holdout_adsorbent |   -0.0296 |   0.0318 |     28.9773 |    25.859  |
| screening_safe_plus_adsorbate_onehot | dummy_mean  | holdout_reference |   -0.0925 |   0.0865 |     29.4447 |    25.0407 |
| screening_safe_plus_adsorbate_onehot | dummy_mean  | random_regime     |   -0.014  |   0.0136 |     29.853  |    25.3624 |
| screening_safe_plus_adsorbate_onehot | extra_trees | holdout_adsorbent |    0.2888 |   0.3972 |     23.3017 |    16.435  |
| screening_safe_plus_adsorbate_onehot | extra_trees | holdout_reference |   -0.0666 |   0.3354 |     29.2471 |    21.8605 |
| screening_safe_plus_adsorbate_onehot | extra_trees | random_regime     |    0.7197 |   0.0481 |     15.6206 |    10.7542 |
| screening_safe_plus_adsorbate_onehot | lightgbm    | holdout_adsorbent |    0.3278 |   0.4067 |     22.5125 |    16.2803 |
| screening_safe_plus_adsorbate_onehot | lightgbm    | holdout_reference |   -0.1002 |   0.5653 |     29.5444 |    22.2449 |
| screening_safe_plus_adsorbate_onehot | lightgbm    | random_regime     |    0.7193 |   0.0604 |     15.5791 |    10.7512 |
| screening_safe_plus_adsorbate_onehot | xgboost     | holdout_adsorbent |    0.3646 |   0.3491 |     22.0416 |    16.049  |
| screening_safe_plus_adsorbate_onehot | xgboost     | holdout_reference |   -0.0296 |   0.3513 |     28.6931 |    21.8531 |
| screening_safe_plus_adsorbate_onehot | xgboost     | random_regime     |    0.7283 |   0.0716 |     15.2905 |    10.839  |
| screening_safe_plus_ion              | dummy_mean  | holdout_adsorbent |   -0.0296 |   0.0318 |     28.9773 |    25.859  |
| screening_safe_plus_ion              | dummy_mean  | holdout_reference |   -0.0925 |   0.0865 |     29.4447 |    25.0407 |
| screening_safe_plus_ion              | dummy_mean  | random_regime     |   -0.014  |   0.0136 |     29.853  |    25.3624 |
| screening_safe_plus_ion              | extra_trees | holdout_adsorbent |    0.3821 |   0.3275 |     21.7983 |    15.911  |
| screening_safe_plus_ion              | extra_trees | holdout_reference |    0.1307 |   0.1549 |     26.2957 |    20.66   |
| screening_safe_plus_ion              | extra_trees | random_regime     |    0.7218 |   0.0674 |     15.493  |    10.6645 |
| screening_safe_plus_ion              | lightgbm    | holdout_adsorbent |    0.3234 |   0.3745 |     22.7171 |    16.2957 |
| screening_safe_plus_ion              | lightgbm    | holdout_reference |   -0.0183 |   0.3825 |     28.4909 |    21.5892 |
| screening_safe_plus_ion              | lightgbm    | random_regime     |    0.7093 |   0.0607 |     15.8744 |    10.9123 |
| screening_safe_plus_ion              | xgboost     | holdout_adsorbent |    0.3635 |   0.3441 |     22.059  |    16.2711 |
| screening_safe_plus_ion              | xgboost     | holdout_reference |   -0.0055 |   0.3352 |     28.3407 |    21.6004 |
| screening_safe_plus_ion              | xgboost     | random_regime     |    0.7265 |   0.0745 |     15.3375 |    10.8817 |

## Leave-One-Adsorbate-Out Summary

| feature_set                          | model       |   n_groups |   R2_mean |   R2_median |   RMSE_mean |   MAE_mean |
|:-------------------------------------|:------------|-----------:|----------:|------------:|------------:|-----------:|
| full_capacity_process_model          | lightgbm    |          7 |   -0.9465 |     -0.2797 |     27.4476 |    23.1823 |
| screening_safe_plus_ion              | extra_trees |          7 |   -1.0732 |     -0.1396 |     27.8311 |    24.111  |
| full_capacity_process_model          | extra_trees |          7 |   -1.123  |     -0.3738 |     29.4093 |    24.5299 |
| full_capacity_process_model          | xgboost     |          7 |   -1.1303 |     -0.4098 |     29.0102 |    23.756  |
| screening_safe_plus_adsorbate_onehot | extra_trees |          7 |   -1.3013 |     -0.4108 |     30.3038 |    25.2211 |
| screening_safe_plus_ion              | lightgbm    |          7 |   -1.782  |     -0.4671 |     32.204  |    27.3395 |
| screening_safe_plus_ion              | xgboost     |          7 |   -2.0172 |     -0.3617 |     32.8956 |    28.0254 |
| screening_safe_numeric_only          | extra_trees |          7 |   -2.1596 |     -0.2974 |     34.0768 |    28.743  |
| screening_safe_plus_adsorbate_onehot | xgboost     |          7 |   -2.1901 |     -0.6556 |     34.9455 |    29.239  |
| screening_safe_plus_adsorbate_onehot | lightgbm    |          7 |   -2.2117 |     -0.6064 |     35.479  |    29.6609 |
| screening_safe_numeric_only          | lightgbm    |          7 |   -2.3598 |     -0.5155 |     35.884  |    29.7887 |
| screening_safe_numeric_only          | xgboost     |          7 |   -2.3699 |     -0.4365 |     35.6968 |    30.0237 |

## Interpretation

- `random_regime` estimates row-level interpolation within the merged dataset.
- `holdout_reference` estimates generalization to unseen literature/lab sources; this is the most important reviewer-facing stress test available without explicit lab labels.
- `holdout_adsorbent` estimates generalization to unseen adsorbent names.
- Capacity-including models should not be framed as pre-experiment screening models.
- Screening models are more scientifically defensible, but lower accuracy is expected.
- Regime exclusions define the primary applicability domain and must be reported explicitly.

## Reproducibility Command

```bash
python scripts/run_leakage_aware_benchmark.py --out-dir results/leakage_aware_benchmark_no_neutral_low --seeds 5 --models dummy_mean xgboost extra_trees lightgbm --exclude-regimes Neutral_LowC0
```

## Missing Lab Split

No usable `Lab_Source` column was found. Add a column with values such as `Own_Lab` and `External_Lab`, then rerun with:

```bash
python scripts/run_leakage_aware_benchmark.py --data path/to/labelled_data.csv --lab-source-column Lab_Source --exclude-regimes Neutral_LowC0
```
