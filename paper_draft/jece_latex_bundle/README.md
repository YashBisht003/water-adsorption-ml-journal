# JECE LaTeX Bundle

This bundle contains a Journal of Environmental Chemical Engineering submission
package formatted with the Elsevier `elsarticle` class.

## Analysis Domain

The raw dataset contains 566 records. After deterministic cleaning, 563 records
remain. The neutral/low-concentration regime (`Neutral_LowC0`) contains only 26
records and is treated as outside the primary applicability domain. The main
modeling and manuscript results therefore use the remaining 537 records:
`Acidic_HighC0`, `Acidic_LowC0`, and `Neutral_HighC0`.

This exclusion is intentional and reported in the manuscript rather than hidden:
the full cleaned dataset is audited, while the restricted-domain results are
used for the primary ML claims.

## Files

- `main.tex`: manuscript source.
- `main.pdf`: compiled manuscript PDF.
- `cover_letter.tex`: cover letter source.
- `cover_letter.pdf`: compiled cover letter.
- `highlights.tex`: separate highlights file for Elsevier submission.
- `graphical_abstract.png`: graphical abstract file.
- `figure_1_workflow.png`: workflow figure used in the manuscript.
- `track_a_within_source_r2.png`: Track A source-specific R2 figure.
- `track_b_global_split_r2.png`: Track B validation comparison figure.
- `track_a_within_source_cv.csv`: source-specific model results.
- `track_b_global_summary_for_dual_track.csv`: global validation results.

The figure and table files are also retained in `figures/` and `tables/` for
project organization, but the root-level copies are the submission-friendly
versions referenced by `main.tex`.

## Compile

On this machine:

```bash
/home/ub/.local/bin/tectonic main.tex
/home/ub/.local/bin/tectonic cover_letter.tex
```

If using a standard TeX Live installation:

```bash
latexmk -pdf main.tex
```

## Reproduce Restricted-Domain Results

From the project root:

```bash
python scripts/run_leakage_aware_benchmark.py --out-dir results/leakage_aware_benchmark_no_neutral_low --seeds 5 --models dummy_mean xgboost extra_trees lightgbm --exclude-regimes Neutral_LowC0
python scripts/run_dual_track_analysis.py --out-dir results/dual_track_no_neutral_low --benchmark-dir results/leakage_aware_benchmark_no_neutral_low --exclude-regimes Neutral_LowC0
```

## Submission Notes

The manuscript uses Elsevier review format, 12 pt Times-style text/math, line
numbers, graphical abstract, and JECE-compliant highlights.

Before final online submission, add the corresponding-author email for Yash
Bisht in the submission system and, if desired, in `main.tex`. Also confirm the
funding statement. If explicit own-lab/external-lab labels become available,
rerun the `Lab_Source` validation and update the limitation section.
