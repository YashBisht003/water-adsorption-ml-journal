# Manuscript Blueprint

Working title:

**Source-aware and leakage-aware machine learning for adsorption removal efficiency: high-accuracy interpolation versus cross-source generalization**

Target journal:

**Journal of Environmental Chemical Engineering** or a related water-treatment / environmental-data-science journal.

## One-Sentence Thesis

High removal-efficiency prediction accuracy is achievable within known adsorption study sources, but heterogeneous multi-source adsorption datasets show much weaker cross-source generalization; therefore, adsorption ML papers should separate source-specific interpolation, process-monitoring prediction, pre-experiment screening, and external-source validation.

## Paper Strategy

Use a dual-track structure:

```text
Track A: high-R2 source-specific modeling
Track B: leakage-aware global validation
```

This lets us report high R2 where it is scientifically valid while also making the paper more rigorous than many ML-for-adsorption studies that report only random-split performance.

## Core Claims

1. Source-specific adsorption RE models can reach near-0.95 or higher R2.
2. Global row-wise validation on heterogeneous data gives moderate-to-good R2.
3. Reference/source-held-out validation is much harder and exposes weak external generalization.
4. Adsorption capacity must be treated carefully because it is outcome-adjacent to removal efficiency.
5. A useful adsorption ML framework should report both high-accuracy interpolation and external/source transfer limits.

## Current Results To Report

### Track A: Source-Specific High-R2 Results

Use `results/dual_track/track_a_within_source_cv.csv`.

| Source | Rows | Feature set | R2 | RMSE | MAE |
| --- | ---: | --- | ---: | ---: | ---: |
| Shen et al. (2017b) | 65 | Full process model | 0.992 | 3.20 | 2.33 |
| Gao et al. (2019) | 48 | Full process model | 0.974 | 2.97 | 2.14 |
| Zama et al. (2017) | 91 | Full process model | 0.960 | 7.84 | 3.56 |
| Cui et al. (2016a) | 24 | Full process model | 0.894 | 6.65 | 4.56 |

Important wording:

```text
These are source-specific/interpolation results, not universal model results.
```

### Track B: Global Generalization Results

Use `results/dual_track/track_b_global_summary_for_dual_track.csv`.

| Validation | Feature/model | R2 | RMSE | Interpretation |
| --- | --- | ---: | ---: | --- |
| Random row/regime split | Full capacity XGBoost | 0.798 | 13.16 | In-dataset interpolation |
| Random row/regime split | Safer no-capacity + ion features | 0.728 | 15.26 | Screening-style model |
| Holdout adsorbent | Full capacity XGBoost | 0.577 | 16.38 | Moderate unseen-adsorbent transfer |
| Holdout adsorbent | Safer no-capacity numeric model | 0.489 | 18.49 | Moderate safer transfer |
| Holdout reference/source | Best screening model | 0.112 | 26.68 | Weak unseen-source transfer |

Important wording:

```text
The performance gap between random row-wise validation and reference-held-out validation indicates that adsorption RE prediction is strongly source-dependent.
```

## Existing Work And Limitations

### What Existing Work Often Does

Many adsorption ML papers:

- use random train/test splits or k-fold CV over rows,
- focus on one adsorbent family, one pollutant family, or one experimental source,
- report high test R2 values, often around 0.95-0.99,
- use SHAP/PDP or feature importance to interpret pH, dose, initial concentration, surface area, or contact time,
- do not always test leave-source-out, leave-adsorbent-out, or leave-contaminant-out generalization.

### Examples To Discuss

1. Hafsa et al., Water 2020

   Reported high R2 values around 0.96-0.99 for heavy-metal adsorption modeling. Their abstract states that wet-experiment data were supplemented with synthetic data and evaluated using ten-fold cross-validation. This supports the point that high R2 is possible under within-dataset validation, but it does not answer the harder question of unseen-source transfer.

2. Wang et al., Journal of Water Process Engineering 2024

   Reported XGBoost test R2 around 0.99 for heavy-metal removal by biochar and used SHAP/PDP-style interpretability. This is close to the style reviewers may expect. We can position our Track A as comparable high-accuracy interpolation, while Track B adds a validation-stress-test layer.

3. Patel et al., ACS ES&T Water 2026

   This is a very useful comparison because it explicitly separates random 80/20 validation and leave-one-compound-out validation. They reported R2 around 0.93 for the random split but around 0.30 for leave-one-PFAS-out validation. This supports our central claim: performance depends strongly on the validation task.

## Our Gap Statement

Use language like:

```text
Previous adsorption ML studies have demonstrated that high removal-efficiency prediction accuracy is achievable, particularly under row-wise cross-validation or within restricted adsorbent-pollutant systems. However, for heterogeneous literature-compiled adsorption data, the distinction between source-specific interpolation and cross-source generalization is often underreported. This distinction is critical because adsorption measurements are affected by laboratory protocols, adsorbent preparation routes, pollutant speciation, and reported variables that may be outcome-adjacent to removal efficiency.
```

## Proposed Contributions

1. A curated and cleaned heterogeneous adsorption RE dataset covering heavy-metal adsorption observations.
2. A dual-track validation framework separating source-specific interpolation from cross-source generalization.
3. Demonstration that high R2 is reproducible within individual sources, matching the range commonly reported in adsorption ML papers.
4. Demonstration that reference/source-held-out validation is much harder, revealing the limits of universal RE prediction.
5. A leakage-aware feature analysis distinguishing process-monitoring models from pre-experiment screening models.
6. A lab-source validation template for own-lab versus external-lab testing.

## Manuscript Structure

### 1. Abstract

Structure:

- Problem: ML is increasingly used for adsorption RE prediction, but validation protocols often blur interpolation and generalization.
- Method: heterogeneous literature dataset, physics-informed features, XGBoost/ExtraTrees, dual-track validation.
- Result: source-specific models reach R2 up to 0.992; random global split reaches R2 around 0.798; reference-held-out validation falls to around 0.112.
- Conclusion: high R2 is valid for source-specific prediction, but universal claims require source-held-out validation and applicability-domain analysis.

### 2. Introduction

Key paragraphs:

1. Heavy-metal contamination and adsorption relevance.
2. ML for adsorption prediction and optimization.
3. Existing high-R2 results and their appeal.
4. Limitation: row-wise validation can overstate deployability on heterogeneous literature datasets.
5. Our solution: source-aware, leakage-aware, dual-track validation.

### 3. Related Work

Organize by theme:

- Adsorption ML with high random-split accuracy.
- Interpretability in adsorption ML using SHAP/PDP.
- Generalization and validation concerns in environmental ML.
- Need for source/lab-aware and leakage-aware validation.

### 4. Dataset And Preprocessing

Include:

- 566 raw rows, 563 cleaned rows.
- 189 unique references after normalization in the analysis pipeline.
- 219 unique adsorbents, 16 normalized adsorbates.
- Regime counts: acidic/high C0, acidic/low C0, neutral/high C0, neutral/low C0.
- Numeric cleaning rules.
- Adsorbate normalization.
- Adsorbent-family classification.

Be transparent:

- `Neutral_LowC0` is underrepresented.
- There is no explicit lab-source label yet; `Ref_group` is currently used as a source proxy.

### 5. Feature Engineering

Separate feature families:

- Operational variables: concentration, dose, pH, temperature, RPM, contact time.
- Adsorbent variables: surface area, adsorbent family.
- Process-monitoring variables: adsorption capacity and capacity-derived terms.
- Screening-safe physics features: site density, log time, mixing index, driving force, acidity strength.
- Ion descriptors: charge, ionic radius, hydrated radius, ionic potential, hydration ratio.

Critical discussion:

```text
Adsorption capacity is included only in process-monitoring models because it may not be known before an adsorption experiment and may be mathematically related to RE.
```

### 6. Modeling And Validation

Models:

- XGBoost as main model.
- ExtraTrees as robustness comparator.
- Dummy mean baseline.

Validation protocols:

- Track A: within-source 5-fold CV for sources with at least 20 rows.
- Track B1: random row/regime split.
- Track B2: holdout adsorbent.
- Track B3: holdout reference/source.
- Optional later: holdout lab source after `Lab_Source` labels are added.

### 7. Results

Recommended order:

1. Dataset overview table.
2. Track A high-R2 source-specific results.
3. Track B global validation comparison.
4. Capacity/leakage ablation.
5. Leave-one-adsorbate or holdout-adsorbent stress test.
6. Interpretation plots or SHAP/PDP if we add them.

### 8. Discussion

Core discussion points:

- Why high R2 appears in source-specific models.
- Why source-held-out validation is harder.
- Why this does not invalidate ML, but clarifies its deployment boundary.
- How own-lab/external-lab validation can turn this from a benchmarking paper into a stronger applied paper.
- Why capacity-inclusive and capacity-free models answer different questions.

### 9. Limitations

State clearly:

- Current lab source labels are missing.
- Some regimes are underrepresented.
- Ion descriptor table is preliminary and should be cited/cleaned.
- Adsorbent descriptors such as pHpzc, pore volume, pore size, zeta potential, FTIR groups, and CEC are not yet complete.
- Cross-source performance is currently weak.

### 10. Conclusion

Message:

```text
Adsorption ML should not be judged by a single R2. High-accuracy interpolation and external-source generalization are different tasks. Reporting both makes adsorption ML more reproducible, deployable, and chemically honest.
```

## Figures And Tables

### Main Figures

1. Workflow figure

   Dataset cleaning -> feature families -> Track A source-specific CV -> Track B global validation -> source-aware interpretation.

2. Track A source-specific R2 bar plot

   Existing file:

   ```text
   results/dual_track/track_a_within_source_r2.png
   ```

3. Track B validation comparison plot

   Existing file:

   ```text
   results/dual_track/track_b_global_split_r2.png
   ```

4. Optional: parity plot for best source-specific model.

5. Optional: SHAP/PDP figure for pH, concentration, dose, contact time, and ion descriptors.

### Main Tables

1. Dataset summary and regimes.
2. Feature sets and whether each is process-monitoring or screening-safe.
3. Track A source-specific model results.
4. Track B validation results.
5. Comparison with existing adsorption ML studies and their validation protocols.

## First Writing Sprint

Write in this order:

1. Results section first, because our core evidence is already clear.
2. Methods/validation section second, to lock down terminology.
3. Introduction third, built around the validation gap.
4. Discussion fourth, around interpolation versus generalization.
5. Abstract last.

## Immediate Next Tasks

1. Add/fill `Lab_Source` using one of:

   ```text
   results/dual_track/lab_source_by_reference_template.csv
   results/dual_track/data_with_lab_source_template.csv
   ```

2. Decide whether the main manuscript uses:

   ```text
   Journal version A: dual-track validation paper
   Journal version B: own-lab/external-lab validation paper
   ```

3. Add a cited ion descriptor table.
4. Generate SHAP/PDP or ALE interpretation plots.
5. Draft the Results section from the tables above.

