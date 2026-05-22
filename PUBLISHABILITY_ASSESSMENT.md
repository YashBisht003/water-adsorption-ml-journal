# Water Adsorption ML Journal Assessment

Date: 2026-05-22

## Current Project Location

The water-treatment ML work has been separated from the humanoid repository:

```text
/home/ub/yash_projects/water_adsorption_ml_journal/
```

The original transferred archive is preserved in:

```text
/home/ub/yash_projects/water_adsorption_ml_journal/watersusmit_source/
```

The extracted working files are in:

```text
/home/ub/yash_projects/water_adsorption_ml_journal/waterresearchpaper/
```

## Current Scientific Story

The current work predicts adsorption removal efficiency, `RE (%)`, for water/wastewater heavy-metal removal from a heterogeneous literature-compiled dataset.

The current manuscript direction is:

- robust cleaning of literature adsorption data,
- physics-inspired features such as site density, capacity/loading ratio, contact-time transforms, mixing index, and acidity strength,
- regime-aware modeling by pH and initial concentration,
- gradient-boosting models with SHAP-style interpretation.

## Dataset Audit

Using `waterresearchpaper/data_2023(1).csv`:

- Raw rows: 566
- Clean rows after numeric parsing and `0 <= RE <= 100`: 563
- Unique references: 192
- Unique adsorbents: 240
- Unique adsorbates: 27
- Median initial concentration: 50 ppm

Regime counts:

```text
Acidic_HighC0      246
Acidic_LowC0       231
Neutral_HighC0      60
Neutral_LowC0       26
```

This means `Neutral_LowC0` is underpowered for strong regime-wise claims.

## Current Saved Results

Saved prediction files show:

```text
analysis_plots/test_predictions.csv
R2   = 0.8093
RMSE = 13.4398
MAE  = 9.0604

ionic_feature_results/test_predictions_with_ionic_features.csv
R2   = 0.8532
RMSE = 11.7891
MAE  = 7.9590
```

These are good row-wise holdout metrics, but they should not yet be treated as final journal-grade generalization evidence.

## Main Methodological Risk

`Adsorption capacity (mg/g)` and capacity-derived terms are high-risk predictors for `RE (%)`.

Reason: in many adsorption experiments, adsorption capacity and removal efficiency are both derived from the same concentration change. Even when the formula does not exactly reconstruct `RE` in this dataset, capacity remains outcome-adjacent and dominates feature importance:

```text
Cap_Load_Ratio                 408
Surface area(m2/g)             352
Adsorption capacity(mg/g)      296
Site_Density                   246
```

For journal submission, report both:

- a full model with capacity included, framed as a descriptive/process-monitoring model,
- a stricter pre-experiment screening model without capacity or capacity-derived features.

## Generalization Benchmark

Latest reproducible benchmark:

```bash
python scripts/run_leakage_aware_benchmark.py --seeds 5 --models dummy_mean xgboost extra_trees
```

Outputs:

```text
results/leakage_aware_benchmark/REPORT.md
results/leakage_aware_benchmark/benchmark_summary.csv
results/leakage_aware_benchmark/leave_one_adsorbate.csv
```

Earlier sanity checks used XGBoost under repeated splits. The updated script
adds a dummy baseline, ExtraTrees, four feature sets, and leave-one-adsorbate
stress tests.

Mean over 5 random seeds in the latest run:

```text
Feature set                       Split              Model      Mean R2   Mean RMSE
full_capacity_process_model       random_regime      XGBoost      0.798      13.16
screening_safe_plus_ion           random_regime      XGBoost      0.728      15.26
screening_safe_numeric_only       random_regime      XGBoost      0.686      16.43

full_capacity_process_model       holdout_adsorbent  XGBoost      0.577      16.38
screening_safe_numeric_only       holdout_adsorbent  XGBoost      0.489      18.49

screening_safe_numeric_only       holdout_reference  ExtraTrees   0.112      26.68
screening_safe_numeric_only       holdout_reference  XGBoost     -0.022      28.42
full_capacity_process_model       holdout_reference  XGBoost     -0.055      28.34
```

Interpretation:

- Random row-level prediction is workable.
- Generalization to unseen literature references is currently weak.
- Generalization to unseen adsorbent names is moderate, but still unstable.
- Leave-one-adsorbate-out performance is poor for many metals, so the current dataset should not claim reliable new-metal extrapolation.
- The current CSV has no explicit `Lab_Source` column. The script is ready to run leave-lab/source-out validation once that column is added.

## Publishability Judgment

The current version is not ready for a strong Journal of Environmental Chemical Engineering submission if framed mainly as "we achieved high R2."

It could become publishable if reframed and strengthened as:

```text
Honest, regime-aware, leakage-aware machine learning for adsorption removal efficiency:
what can be predicted from heterogeneous literature data, what fails under external-source validation,
and how lab-calibrated descriptors improve reliability.
```

## Why Other Papers Report R2 Near 0.95

I ran an additional diagnostic to check whether high R2 appears when the
prediction task is narrowed to one data source/reference:

```text
results/leakage_aware_benchmark/high_r2_diagnostic.csv
```

Within-reference 5-fold results:

```text
Reference/source          Rows   R2
Shen et al. (2017b)        65    0.991
Gao et al. (2019)          48    0.977
Zama et al. (2017)         91    0.951
Cui et al. (2016a)         24    0.871
```

This explains the gap:

- Source-specific interpolation can easily reach R2 around 0.95.
- Heterogeneous multi-source prediction is much harder.
- Leave-source/reference-out validation is the honest test for a general model.

Therefore, a high-R2 paper is possible if the scope is narrowed to a specific
lab, adsorbent family, or adsorbate/adsorbent system. For a broader JECE paper,
the stronger contribution is not a single inflated R2, but a source-aware
framework showing when high accuracy is and is not valid.

## Dual-Track Plan

We can do both tracks, as separate claims:

```text
Track A: high-R2 source-specific/interpolation models
Track B: leakage-aware global/source-held-out validation
```

I generated the dual-track package with:

```bash
python scripts/run_dual_track_analysis.py
```

Outputs:

```text
results/dual_track/DUAL_TRACK_MANUSCRIPT_PLAN.md
results/dual_track/track_a_within_source_cv.csv
results/dual_track/track_b_global_summary_for_dual_track.csv
results/dual_track/track_a_within_source_r2.png
results/dual_track/track_b_global_split_r2.png
results/dual_track/lab_source_by_reference_template.csv
results/dual_track/data_with_lab_source_template.csv
```

Track A examples:

```text
Shen et al. (2017b): R2 = 0.992, RMSE = 3.20
Gao et al. (2019):   R2 = 0.974, RMSE = 2.97
Zama et al. (2017):  R2 = 0.960, RMSE = 7.84
```

Track B examples:

```text
Random row/regime split, full process model:      R2 = 0.798
Holdout adsorbent, full process model:            R2 = 0.577
Holdout reference/source, best screening model:   R2 = 0.112
```

This gives the manuscript a balanced structure:

- report high accuracy where it is scientifically valid,
- show the limits of global transfer,
- then position own-lab/external-lab validation as the decisive contribution.

## Highest-Value Improvements

1. Add a source/lab column.

   The user noted that data comes from two labs: one internal and one external. The dataset currently has `Ref.` but no explicit `Lab_Source` column. For publication, add a clean provenance field such as `Own_Lab`, `External_Lab`, or `Literature`, then report leave-one-lab-out validation.

2. Separate two prediction tasks.

   Task A: process-monitoring prediction with adsorption capacity included.

   Task B: pre-experiment screening prediction with adsorption capacity removed.

3. Use stricter validation.

   Include random split, group split by reference, group split by adsorbent, and leave-one-adsorbate-out for metals with enough rows.

4. Improve chemical descriptors.

   Replace mostly one-hot adsorbate encoding with charge, ionic radius, hydrated radius, electronegativity, hydration energy, softness/hardness, and likely aqueous species at pH.

5. Add uncertainty and applicability-domain analysis.

   Use conformal prediction or quantile regression, and flag predictions outside the training domain.

6. Add mechanistic plots beyond SHAP bar charts.

   Include SHAP dependence/PDP/ALE plots for pH, concentration, dose, contact time, and ion descriptors. Link each trend to adsorption chemistry.

7. Consider new experiments.

   The strongest publishable addition would be external validation using fresh own-lab experiments not used in training, especially for the weak regimes: neutral/low concentration and alkaline pH.
