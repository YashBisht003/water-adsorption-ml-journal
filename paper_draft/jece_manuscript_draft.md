# Source-aware and leakage-aware machine learning for adsorption removal efficiency: high-accuracy interpolation versus cross-source generalization

**Manuscript type:** Research article

**Target journal:** Journal of Environmental Chemical Engineering

**Authors:** [Author names to be inserted]

**Affiliations:** [Affiliations to be inserted]

**Corresponding author:** [Name, email]

## Highlights

- Source-specific adsorption models reached removal-efficiency prediction R2 up to 0.992.
- Random row-wise validation overstated performance relative to source-held-out testing.
- Reference-held-out validation reduced the best observed R2 to approximately 0.11.
- Capacity-inclusive and capacity-free models answer different adsorption-design questions.
- A dual-track validation protocol is proposed for adsorption machine-learning studies.

## Graphical abstract

Suggested graphical abstract:

```text
Heterogeneous adsorption dataset
        |
        v
Feature families: operating conditions, adsorbent descriptors, ion descriptors, capacity terms
        |
        +--> Track A: source-specific CV --> high-R2 interpolation
        |
        +--> Track B: random / adsorbent-held-out / reference-held-out validation --> generalization boundary
        |
        v
Guidance: process-monitoring model vs pre-experiment screening model
```

## Abstract

Machine learning is increasingly used to predict adsorption removal efficiency for water and wastewater treatment, and many studies report high coefficient-of-determination values under row-wise train-test splits or cross-validation. However, adsorption datasets compiled from different laboratories and literature sources are heterogeneous in adsorbent preparation, pollutant speciation, operating protocols, and reported variables. This study develops a source-aware and leakage-aware machine-learning framework for predicting heavy-metal adsorption removal efficiency. A literature-compiled dataset containing 566 observations was cleaned to 563 valid records covering 189 normalized source references, 219 adsorbents, and 16 normalized adsorbate classes. Four feature sets were evaluated to separate capacity-inclusive process-monitoring models from capacity-free pre-experiment screening models. Two validation tracks were used. Track A quantified source-specific interpolation using within-source cross-validation, where XGBoost models reached R2 values of 0.992, 0.974, and 0.960 for three major sources. Track B evaluated broader generalization using repeated random row-wise, adsorbent-held-out, and reference-held-out splits. The full capacity-inclusive XGBoost model achieved R2 = 0.798 under random row-wise validation and R2 = 0.577 under adsorbent-held-out validation, but reference-held-out validation reduced the best screening-model performance to R2 = 0.112. These results show that high adsorption-prediction accuracy is valid for in-source interpolation, but universal claims require source-held-out validation and careful feature definition. The proposed dual-track protocol provides a practical route for reporting both high-accuracy adsorption modeling and its external generalization limits.

## Keywords

Adsorption; heavy metals; removal efficiency; machine learning; XGBoost; data leakage; source-aware validation; wastewater treatment

## 1. Introduction

Heavy-metal contamination remains a persistent water and wastewater treatment challenge because metals such as Pb, Cd, Cr, Ni, Cu, As, Hg, and Zn are toxic, non-biodegradable, and prone to accumulation in environmental and biological systems. Adsorption is widely studied for heavy-metal removal because it can be operated under mild conditions, adapted to low-cost or waste-derived adsorbents, and optimized through pH, dose, contact time, temperature, and adsorbent surface chemistry. In parallel, the rapid growth of experimental adsorption datasets has encouraged the use of machine learning to predict removal efficiency and guide adsorbent selection.

Recent adsorption and water-treatment machine-learning studies demonstrate that high predictive accuracy is possible. Hafsa et al. modeled heavy-metal adsorption efficiencies using several machine-learning algorithms and reported R2 values in the 0.96-0.99 range under ten-fold cross-validation, with wet experimental data supplemented by synthetic interpolation [1]. Wang et al. used interpretable machine learning for heavy-metal removal by biochar and reported XGBoost test performance near R2 = 0.99, together with SHAP and partial-dependence analyses to relate removal efficiency to biochar and operating variables [2]. Lu et al. predicted heavy-metal removal by chitosan-based flocculants using random forests and reported R2 = 0.9354, identifying solution pH and flocculant molecular weight as important drivers [4]. These studies are valuable because they show that nonlinear tree-based models can learn strong structure-property-process relationships in adsorption and flocculation systems.

However, high row-wise validation performance does not necessarily imply that a model generalizes to unseen laboratories, adsorbents, pollutants, or source studies. In machine-learning-based science, data leakage and overly permissive validation can produce optimistic estimates of deployable performance [5]. This concern is especially relevant for heterogeneous adsorption datasets because different sources may use different adsorbent synthesis routes, measurement protocols, concentration ranges, pollutant species, and reporting conventions. Moreover, some reported variables are outcome-adjacent. For example, adsorption capacity and removal efficiency are both derived from concentration changes in many batch adsorption experiments:

```text
RE (%) = 100 * (C0 - Ce) / C0
qe = (C0 - Ce) V / m
```

where C0 and Ce are initial and equilibrium concentrations, V is solution volume, and m is adsorbent mass. Capacity can therefore be useful for process monitoring, but it should be treated carefully in pre-experiment screening models.

The distinction between interpolation and generalization is beginning to receive more attention. In PFAS adsorption modeling, Patel et al. reported strong performance under an 80/20 random split but substantially weaker performance under leave-one-compound-out validation, highlighting that compound-wise generalization is a distinct and more difficult task [3]. A similar distinction is needed for adsorption removal-efficiency prediction across heterogeneous heavy-metal datasets.

This study addresses that gap by developing a dual-track validation framework for adsorption removal-efficiency prediction. Track A evaluates source-specific interpolation and asks whether high R2 values comparable to the adsorption-ML literature can be reproduced within individual sources. Track B evaluates global generalization under random row-wise, adsorbent-held-out, and reference-held-out validation. The study further separates capacity-inclusive process-monitoring models from capacity-free screening models. The objective is not only to maximize R2, but to identify when high R2 is scientifically valid, when it is validation-dependent, and how adsorption ML studies should report performance for practical use.

## 2. Materials and methods

### 2.1 Dataset

The dataset was compiled from reported heavy-metal adsorption experiments. Each record corresponds to an adsorption condition with a reported removal efficiency. The raw dataset contained 566 rows and 13 columns. After numeric parsing, missing-value filtering, and constraining removal efficiency to the physically meaningful range 0-100%, 563 records remained. The cleaned analysis dataset included 189 normalized source references, 219 adsorbents, and 16 normalized adsorbate classes.

The target variable was removal efficiency, RE (%). Input variables included adsorbent name, adsorbate identity, surface area, adsorption capacity, initial concentration, contact time, adsorbent dose, agitation speed, initial pH, temperature, and reference/source identifier. The source identifier, denoted `Ref_group`, was used as a proxy for source or laboratory origin because the present dataset does not yet contain an explicit `Lab_Source` column.

### 2.2 Numeric cleaning and source normalization

Literature-derived adsorption tables often contain nonuniform numeric formatting. Numeric entries were cleaned using deterministic rules: decimal commas were converted to decimal points, uncertainty and approximate symbols were removed, multiple values in one cell were reduced to the first numeric value, and nonnumeric text was parsed using regular expressions. Rows with missing values in required numeric variables were removed. Adsorbate names were normalized to reduce duplicates such as elemental and valence-state aliases. Adsorbent names were also grouped into broad families, including carbon-based, mineral/oxide, polymer, biomass, and composite/other categories.

The cleaned data were assigned to four pH-concentration regimes using pH = 7 and the median initial concentration, C0 = 50 ppm:

| Regime | Count |
| --- | ---: |
| Acidic_HighC0 | 246 |
| Acidic_LowC0 | 231 |
| Neutral_HighC0 | 60 |
| Neutral_LowC0 | 26 |

The small size of `Neutral_LowC0` was treated as a limitation for regime-wise claims.

### 2.3 Feature engineering

Four feature sets were constructed to distinguish scientific use cases.

**Full capacity process model.** This feature set included operating variables, surface area, adsorption capacity, capacity-derived ratios, ion descriptors, and adsorbent/adsorbate categories. It is suitable for descriptive modeling or process monitoring when capacity is known.

**Screening-safe plus ion model.** This feature set excluded adsorption capacity and capacity-derived terms. It retained operating variables, surface area, physics-inspired transformations, ion descriptors, and adsorbent family.

**Screening-safe plus adsorbate one-hot model.** This feature set excluded adsorption capacity and used adsorbate identity as a categorical variable.

**Screening-safe numeric-only model.** This was the strictest feature set, excluding adsorption capacity, capacity-derived variables, and adsorbate identity.

Physics-inspired variables included site density, logarithmic contact time, mixing index, driving-force proxy, and acidity strength:

```text
Site_Density = dose * surface area / C0
LogTime = log(1 + contact time)
Mixing_Index = RPM * LogTime
Driving_Force = C0 / dose
Acidity_Strength = |pH - 7|
```

Ion descriptors included charge, ionic radius, hydrated radius, ionic potential, and hydration ratio. These descriptors were used to reduce reliance on one-hot pollutant labels, although the ion descriptor table should be expanded and cited in future versions.

### 2.4 Models

XGBoost was selected as the primary model because gradient-boosted trees are effective for nonlinear tabular datasets and are widely used in adsorption ML studies [6,9]. ExtraTrees and a mean dummy regressor were included as comparator models. Random-forest-style ensembles are common baselines for environmental adsorption prediction and provide robust nonlinear performance [7]. The present draft reports XGBoost and ExtraTrees because they capture the main performance trends while keeping the analysis reproducible and computationally light.

### 2.5 Validation protocols

Two validation tracks were used.

**Track A: source-specific interpolation.** For each source/reference with at least 20 records, source-specific 5-fold cross-validation was performed. This tests whether high R2 values are achievable within a known experimental source, which is similar to the validation setting of many high-performance adsorption ML studies.

**Track B: global generalization.** Three global validation protocols were applied:

1. Random row-wise split stratified by pH-concentration regime.
2. Group split holding out adsorbent names.
3. Group split holding out entire references/sources.

Each Track B split was repeated over five random seeds. Metrics included R2, RMSE, MAE, and bias. Reference-held-out validation was treated as the most conservative source-transfer test available without explicit `Lab_Source` labels.

## 3. Results

### 3.1 Source-specific models reproduce high-R2 adsorption prediction

Track A showed that high removal-efficiency prediction accuracy is achievable when models are trained and evaluated within individual sources. The full capacity process model achieved R2 = 0.9916 for Shen et al. (2017b), R2 = 0.9735 for Gao et al. (2019), and R2 = 0.9601 for Zama et al. (2017). These results are in the range commonly reported in adsorption ML studies under row-wise or within-domain validation [1,2,4].

| Source | Rows | Feature set | R2 | RMSE | MAE |
| --- | ---: | --- | ---: | ---: | ---: |
| Shen et al. (2017b) | 65 | Full capacity process model | 0.9916 | 3.20 | 2.33 |
| Gao et al. (2019) | 48 | Full capacity process model | 0.9735 | 2.97 | 2.14 |
| Zama et al. (2017) | 91 | Full capacity process model | 0.9601 | 7.84 | 3.56 |
| Cui et al. (2016a) | 24 | Full capacity process model | 0.8936 | 6.65 | 4.56 |

Capacity-free source-specific models also performed well in some cases. For example, the source-specific model for Zama et al. (2017) achieved R2 = 0.9250 using the screening-safe ion feature set, while Shen et al. (2017b) achieved R2 values above 0.91 using capacity-free screening feature sets. This indicates that high source-specific performance is not solely a capacity-leakage artifact, although capacity-inclusive models generally gave the strongest interpolation performance.

### 3.2 Global row-wise validation gives moderate-to-good performance

Under repeated random row-wise splits, the full capacity process model achieved R2 = 0.7975 and RMSE = 13.16. The screening-safe plus ion model achieved R2 = 0.7278 and RMSE = 15.26, while the strict numeric-only screening model achieved R2 = 0.6863 and RMSE = 16.43. Thus, even without adsorption capacity, the model retained useful predictive ability under in-dataset interpolation.

| Validation | Feature/model | R2 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Random row/regime split | Full capacity XGBoost | 0.7975 | 13.16 | 8.32 |
| Random row/regime split | Screening-safe plus ion XGBoost | 0.7278 | 15.26 | 10.80 |
| Random row/regime split | Screening-safe numeric-only XGBoost | 0.6863 | 16.43 | 12.03 |

These results support the use of ML for in-domain adsorption prediction. However, random row-wise validation is not sufficient to claim deployment on unseen sources.

### 3.3 Generalization weakens under adsorbent- and reference-held-out validation

The validation protocol strongly affected performance. When entire adsorbents were held out, the full capacity process model achieved R2 = 0.5768 and RMSE = 16.38. The safer numeric-only screening model achieved R2 = 0.4886 and RMSE = 18.49. This indicates moderate but imperfect transfer to unseen adsorbent names.

Reference-held-out validation was much harder. The best reference-held-out result was obtained by the screening-safe numeric-only ExtraTrees model, with R2 = 0.1117 and RMSE = 26.68. XGBoost achieved R2 values near or below zero under reference-held-out validation, including R2 = -0.0553 for the full capacity process model and R2 = -0.0223 for the screening-safe numeric-only model.

| Validation | Feature/model | R2 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Holdout adsorbent | Full capacity XGBoost | 0.5768 | 16.38 | 11.18 |
| Holdout adsorbent | Screening-safe numeric-only XGBoost | 0.4886 | 18.49 | 13.71 |
| Holdout reference/source | Screening-safe numeric-only ExtraTrees | 0.1117 | 26.68 | 20.43 |
| Holdout reference/source | Screening-safe numeric-only XGBoost | -0.0223 | 28.42 | 21.86 |
| Holdout reference/source | Full capacity XGBoost | -0.0553 | 28.34 | 21.84 |

These results show that removal-efficiency prediction is strongly source-dependent. The same data and model family can produce high R2 under source-specific interpolation, moderate R2 under random global validation, and weak R2 under source-held-out validation.

### 3.4 Capacity-inclusive and capacity-free models answer different questions

The full capacity process model consistently performed best for random row-wise validation and several source-specific models. This is expected because adsorption capacity carries strong information about adsorption performance. However, capacity can be outcome-adjacent to RE and may not be available before an experiment. Therefore, capacity-inclusive models are best framed as process-monitoring or descriptive models. Capacity-free models are more appropriate for pre-experiment screening and adsorbent-design guidance, even though their performance is lower.

## 4. Discussion

### 4.1 Why near-0.95 R2 is both achievable and limited

The Track A results explain why many adsorption ML studies report R2 values near or above 0.95. Within a single source, adsorbent family, pollutant system, or experimental protocol, the response surface can be learned accurately by tree-based models. In this setting, high R2 is not surprising and is not necessarily invalid. The present results reproduce that behavior, with source-specific R2 values of 0.960-0.992 for three major sources.

However, Track B shows that this performance should not be interpreted as universal adsorption prediction. When sources are held out, the model must extrapolate across experimental protocols, adsorbent synthesis methods, pollutant distributions, and reporting conventions. Under that validation setting, performance fell sharply. This finding is consistent with recent adsorption-related work showing that random-split performance can be substantially higher than leave-compound-out performance [3].

### 4.2 Relation to existing adsorption ML literature

The proposed framework complements rather than contradicts previous high-accuracy adsorption ML studies. Hafsa et al. demonstrated that tree-based algorithms can achieve high removal-efficiency prediction accuracy for heavy-metal adsorption datasets under repeated cross-validation [1]. Wang et al. showed that XGBoost, SHAP, and PDP analyses can identify interpretable drivers of heavy-metal removal by biochar [2]. Lu et al. showed that random forests can predict heavy-metal removal by chitosan-based flocculants with R2 above 0.93 [4]. These studies establish that ML is useful for adsorption modeling.

The present study adds a validation-layer contribution: the same adsorption ML workflow should explicitly state whether it is predicting within a known source or generalizing to unseen sources. This distinction is critical because high row-wise R2 can be overoptimistic when the deployment target is a new laboratory, adsorbent synthesis route, pollutant matrix, or literature source. The broader ML literature has identified leakage and validation mismatch as major reproducibility risks [5], and adsorption datasets have several mechanisms by which such mismatch can occur.

### 4.3 Practical implications

For practical adsorption screening, a single universal RE model should be used cautiously. A more reliable workflow is:

1. Use source-specific or lab-calibrated models for high-accuracy optimization within a known experimental system.
2. Use capacity-free screening models for preliminary adsorbent and operating-condition selection.
3. Use source-held-out validation to estimate external deployment risk.
4. Report applicability-domain or uncertainty information when making predictions for new adsorbents, pollutants, or laboratories.

This framing makes high-R2 models useful without overstating their generality.

### 4.4 Limitations

This draft has several limitations. First, the current dataset lacks an explicit `Lab_Source` column, so bibliographic reference was used as a source proxy. The next version should annotate each record as `Own_Lab`, `External_Lab`, or `Literature` and repeat the validation with leave-lab-out splits. Second, the ion descriptor table is preliminary and should be replaced by a fully cited descriptor database including electronegativity, hydration energy, softness/hardness, and likely aqueous speciation. Third, key adsorbent descriptors such as pHpzc, pore volume, pore size, zeta potential, FTIR-derived functional groups, and cation-exchange capacity are incomplete. Fourth, the neutral/low-concentration regime is underrepresented. Finally, reference-held-out performance is currently weak, so the model should not be presented as a universal RE predictor.

## 5. Conclusions

This study developed a source-aware and leakage-aware framework for adsorption removal-efficiency prediction. The results show that high R2 values are achievable within individual adsorption sources, with source-specific XGBoost models reaching R2 up to 0.992. However, global performance depends strongly on the validation protocol: random row-wise validation reached R2 = 0.798 for the full capacity process model, while reference-held-out validation reduced the best observed R2 to approximately 0.112. These findings show that high-accuracy interpolation and cross-source generalization are different scientific tasks. Future adsorption ML studies should report both, separate capacity-inclusive process-monitoring models from capacity-free screening models, and include source- or lab-held-out validation when deployment beyond the training source is claimed.

## CRediT authorship contribution statement

[To be completed after author order is finalized.]

Suggested draft:

- [Author 1]: Conceptualization, Methodology, Software, Formal analysis, Visualization, Writing - original draft.
- [Author 2]: Data curation, Investigation, Writing - review and editing.
- [Author 3]: Supervision, Project administration, Writing - review and editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The cleaned dataset, analysis scripts, and generated benchmark tables will be made available in a public repository upon publication. The present working files include the analysis scripts `scripts/run_leakage_aware_benchmark.py` and `scripts/run_dual_track_analysis.py`, together with benchmark outputs under `results/`.

## Acknowledgements

[Funding and laboratory support to be inserted.]

## References

[1] N. Hafsa, S. Rushd, M. Al-Yaari, M. Rahman, A generalized method for modeling the adsorption of heavy metals with machine learning algorithms, Water 12 (2020) 3490. https://doi.org/10.3390/w12123490.

[2] C. Wang, Y. Zhao, Y. Gao, H. Chen, X. Li, B. Zhou, D. Fan, Z. Fang, J. Liu, Interpretable machine learning for predicting heavy metal removal and optimizing biochar characteristics, Journal of Water Process Engineering 68 (2024) 106484. https://doi.org/10.1016/j.jwpe.2024.106484.

[3] H.V. Patel, J. Green, H. Park, S. Luster-Teasley Pass, R. Zhao, A hybrid response surface methodology and machine learning framework for quantifying effects of physicochemical parameters on PFAS distribution, ACS ES&T Water (2026) Advance online publication. https://doi.org/10.1021/acsestwater.5c01162.

[4] C. Lu, Z. Xu, B. Dong, Y. Zhang, M. Wang, Y. Zeng, C. Zhang, Machine learning for the prediction of heavy metal removal by chitosan-based flocculants, Carbohydrate Polymers 285 (2022) 119240. https://doi.org/10.1016/j.carbpol.2022.119240.

[5] S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (2023) 100804. https://doi.org/10.1016/j.patter.2023.100804.

[6] T. Chen, C. Guestrin, XGBoost: A scalable tree boosting system, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ACM, San Francisco, CA, USA, 2016, pp. 785-794. https://doi.org/10.1145/2939672.2939785.

[7] L. Breiman, Random forests, Machine Learning 45 (2001) 5-32. https://doi.org/10.1023/A:1010933404324.

[8] S.M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Advances in Neural Information Processing Systems 30, 2017, pp. 4765-4774.

[9] J.H. Friedman, Greedy function approximation: A gradient boosting machine, Annals of Statistics 29 (2001) 1189-1232. https://doi.org/10.1214/aos/1013203451.

