# Source-Aware Adsorption ML Journal Project

This repository contains the manuscript, analysis scripts, figures, and benchmark
outputs for a source-aware machine-learning study of heavy-metal adsorption
removal efficiency.

## Current Manuscript Framing

The study separates two prediction tasks:

- Source-specific interpolation, where high removal-efficiency prediction
  accuracy is possible within known experimental sources.
- Leakage-aware global validation, where performance is tested under random
  row-wise, adsorbent-held-out, and reference-held-out splits.

The primary modelling domain excludes the sparse `Neutral_LowC0` regime because
it contains only 26 cleaned observations. The audited dataset contains 563 valid
records, and the restricted primary modelling domain contains 537 records.

## Main Restricted-Domain Results

- Source-specific Track A models reached `R2` up to 0.992.
- Random row-wise full-capacity LightGBM reached `R2 = 0.806`.
- Adsorbent-held-out capacity-free XGBoost reached `R2 = 0.520`.
- Reference-held-out capacity-free ExtraTrees reached `R2 = 0.305`.

These results are intentionally framed as an applicability-domain and validation
study, not as a universal high-`R2` adsorption predictor.

## Reproduce The Main Analysis

```bash
python scripts/run_leakage_aware_benchmark.py \
  --out-dir results/leakage_aware_benchmark_no_neutral_low \
  --seeds 5 \
  --models dummy_mean xgboost extra_trees lightgbm \
  --exclude-regimes Neutral_LowC0

python scripts/run_dual_track_analysis.py \
  --out-dir results/dual_track_no_neutral_low \
  --benchmark-dir results/leakage_aware_benchmark_no_neutral_low \
  --exclude-regimes Neutral_LowC0
```

## Compile The JECE Bundle

```bash
cd paper_draft/jece_latex_bundle
tectonic main.tex
tectonic cover_letter.tex
```

The compiled manuscript PDF and LaTeX sources are in
`paper_draft/jece_latex_bundle/`.

## Submission Readiness Notes

Before journal submission, add the verified corresponding-author email, deposit
the curated data/code in a public repository if submission is public, and add
explicit own-lab/external-lab labels if those labels are available.

