# Effects of Empirical Formulation Development on Abrams-Guided Machine Learning for Concrete Compressive Strength Prediction

## Overview

This repository contains the computational workflow supporting a study of how the development of Abrams-based empirical formulations affects concrete compressive strength prediction and the performance of guided residual machine learning.

Three formulations are evaluated, namely Classical Abrams, Abrams with curing age, and a multivariable Abrams-type formulation. The age-incorporated and multivariable formulations are subsequently used as empirical priors in residual learning. The study examines whether improvements in standalone empirical prediction consistently lead to better guided models and how the results change after experimental observations are incorporated into the development domain.

The manuscript reports conventional multiple linear regression (MLR) as the direct machine learning baseline alongside the empirical formulations and two families of guided residual models. Other learning algorithms are compared under the Abrams-guided and multivariable-guided configurations.

| Component | Description |
| --- | --- |
| Literature dataset | 937 observations from 33 published studies |
| Experimental datasets | Exp1 and Exp2 |
| Combined experimental dataset | Exp12, comprising Exp1 and Exp2 |
| Prediction target | Concrete compressive strength (MPa) |
| Empirical formulations | Classical Abrams, Abrams with curing age, Multivariable Abrams-type model |
| Guided learning | Abrams-guided ML and Multivariable-Abrams-guided ML |
| Interpretation | SHAP, PDP, ICE, and model-response analysis |
| Experimental integration | Literature + Exp1, Literature + Exp2, and Literature + Exp12 retraining |

---

## Research Questions

The study addresses two research questions.

1. How does the development of Abrams-based empirical formulations influence standalone predictive performance and the effectiveness of Abrams-guided machine learning compared with conventional machine learning?
2. How does the predictive performance of Abrams-guided machine learning change following the separate and combined integration of experimental datasets into the literature-based development domain?

The first question is examined using the literature-only benchmark. The second is examined using separate and combined literature-experimental retraining scenarios. Model interpretation provides supporting evidence for both questions.

---

## Research Framework

The workflow comprises six stages.

| Stage | Description |
| --- | --- |
| Data processing | Data cleaning, harmonization, feature preparation, and quality control |
| Modeling | Empirical formulations, conventional MLR, and guided residual learning |
| Evaluation | Primary and complementary prediction metrics |
| Interpretation | SHAP, PDP, ICE, and model-response analyses |
| Retraining | Separate Exp1 and Exp2 integration and combined Exp12 integration |
| Application | Parametric response analysis and preliminary mixture screening |

---

## Dataset

### Literature Dataset

The initial benchmark uses **937 observations collected from 33 published experimental studies**. Each observation represents a concrete mixture and its measured compressive strength at a specified curing age.

| Parameter | Description |
| --- | --- |
| Source | 33 published studies |
| Observations | 937 |
| Main variables | Curing age, cement, water, W/B, supplementary cementitious materials (SCM), fine aggregate, coarse aggregate, superplasticizer, and fiber |
| Target | Concrete compressive strength (MPa) |

Measurements for the same mixture at different curing ages are retained as separate observations because they correspond to different strength responses.

### Experimental Datasets

Two experimental datasets are incorporated into the literature-based development domain, first separately and then together.

| Dataset | Observations | Role |
| --- | ---: | --- |
| Exp1 | 131 | Separate integration with the literature dataset |
| Exp2 | 397 | Separate integration with the literature dataset |
| Exp12 | 528 | Combined Exp1 and Exp2 integration |

| Development domain | Total observations |
| --- | ---: |
| Literature + Exp1 | 1,068 |
| Literature + Exp2 | 1,334 |
| Literature + Exp12 | 1,465 |

**These are integrated retraining analyses, not independent external validation.** Experimental observations are incorporated into model development before the expanded dataset is divided into training and testing subsets.

---

## Modeling Framework

### Abrams-Based Empirical Formulations

| Formulation | Information represented |
| --- | --- |
| Classical Abrams | Water-to-binder ratio (W/B) |
| Abrams with curing age | W/B and curing age |
| Multivariable Abrams-type model | W/B, curing age, and broader mixture composition |

The three formulations are compared as standalone empirical predictors. The latter two are also evaluated as empirical priors for machine learning.

### Guided Residual Learning

For each observation, the empirical model first estimates compressive strength. The machine learning algorithm then estimates the remaining error:

```text
Residual = Measured strength - Empirical prediction
Final prediction = Empirical prediction + Predicted residual
```

The two guided schemes use the same residual-learning concept but different empirical formulations:

- **Abrams-guided ML:** uses the Abrams formulation incorporating curing age.
- **Multivariable-Abrams-guided ML:** uses the multivariable Abrams-type formulation.

This comparison evaluates whether a more accurate standalone empirical formulation also provides a more effective prior for residual learning. The reported comparisons should be interpreted in the context of their respective evaluation subsets.

---

## Machine Learning Models

The modeling workflow includes the following learning algorithms.

| Algorithm | Learning approach |
| --- | --- |
| MLR | Multiple linear regression |
| SVR | Support vector regression |
| KNN | K-nearest neighbors |
| RF | Random forest |
| GBR | Gradient boosting regression |
| XGBoost | Extreme gradient boosting |
| CatBoost | Gradient boosting |
| DNN | Deep neural network |
| AutoGluon | Automated machine learning |

In the manuscript's main literature-only comparison, **MLR** is reported as the conventional direct ML baseline. The other algorithms are reported in the two guided configurations, using model names such as `Abrams-SVR`, `Multivariable-Abrams-SVR`, `Abrams-XGBoost`, and `Multivariable-Abrams-XGBoost`. This distinction avoids treating guided models as conventional ML-only results.

---

## Experimental Design

### Literature-Only Benchmark

The initial analysis uses 937 literature observations and an 80:20 training-testing split. It evaluates:

1. Standalone Classical Abrams, Abrams with curing age, and Multivariable Abrams-type formulations.
2. Conventional MLR as the direct baseline reported in the manuscript.
3. Abrams-guided residual learning.
4. Multivariable-Abrams-guided residual learning.

The objective is to compare the three empirical formulations and evaluate how the selection of an empirical prior affects guided prediction across learning algorithms.

### Literature-Experimental Retraining

The second analysis retrains models using three expanded development domains:

```text
Literature + Exp1
Literature + Exp2
Literature + Exp12
```

Each integrated dataset is divided into training and testing subsets for the corresponding retraining analysis. Results show **within-domain adaptation following experimental data integration**. Comparisons across these domains are not independent source-held-out external validation.

---

## Model Evaluation

The following complementary metrics are reported.

| Metric | Interpretation |
| --- | --- |
| R² | Coefficient of determination |
| RMSE | Root mean squared error, emphasizing larger absolute errors |
| MAE | Mean absolute error |
| MBE | Mean bias error |
| MAPE | Mean absolute percentage error |
| RRMSE | Relative root mean squared error |
| A20 | Fraction of predictions within ±20% of the measured strength |

Model comparisons consider multiple metrics rather than relying on R² alone.

### Main Reported Findings

**Literature-only benchmark.** Standalone test R² increased from 0.5060 for Classical Abrams to 0.8271 for Abrams with curing age and 0.8847 for the multivariable formulation. The two leading guided configurations were **Abrams-AutoGluon**, which achieved test R² = 0.9731 and RMSE = 4.79 MPa, and **Multivariable-Abrams-AutoGluon**, which achieved R² = 0.9719 and RMSE = 4.89 MPa and showed small advantages in some complementary error metrics.

**Combined experimental integration.** After Exp12 retraining, **Abrams-XGBoost** achieved test R² = 0.9830, RMSE = 3.71 MPa, and MAE = 2.45 MPa. **Multivariable-Abrams-XGBoost** followed closely, with R² = 0.9820 and RMSE = 3.82 MPa.

The central finding is that **better standalone empirical prediction does not necessarily lead to better guided residual learning**. The preferred guided configuration depends on the empirical formulation, learning algorithm, and development data domain.

---

## Model Interpretation

Interpretation is used to examine the material relevance of the learned relationships and how these relationships change with empirical formulation and experimental data integration. The workflow includes:

- SHAP global feature importance and summary analysis
- SHAP dependence analysis
- Partial dependence plots (PDP)
- Individual conditional expectation (ICE) analysis
- Parametric model-response analysis

Particular attention is given to W/B and curing age. Their learned response patterns are interpreted alongside established concrete strength mechanisms involving capillary porosity, cement hydration, and calcium silicate hydrate (C-S-H) formation. SHAP values describe model attribution, **not direct causal or microstructural measurements**.

---

## Environment Setup

### Requirements

- Python 3.11
- Conda
- Windows or Linux
- GPU optional for computationally intensive DNN or AutoML training

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate abrams-ml
```

If required, install additional dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package | Purpose |
| --- | --- |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Scikit-learn | Classical machine learning |
| XGBoost | Gradient boosting |
| CatBoost | Gradient boosting |
| PyTorch | Deep neural networks |
| AutoGluon | Automated machine learning |
| SHAP | Model interpretation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |

Package versions and installation details should follow the files supplied in the repository.

---

## Reproducibility Workflow

The project organizes its scripts by research stage:

```text
scripts/
|
|-- 01_data_processing/
|   `-- Data cleaning, harmonization, and dataset preparation
|
|-- 02_modeling/
|   `-- Empirical formulations, conventional MLR, and guided residual models
|
|-- 03_evaluation/
|   `-- Performance metrics and comparative evaluation
|
|-- 04_interpretation/
|   `-- SHAP, PDP, ICE, and response interpretation
|
|-- 05_retraining/
|   `-- Literature-Exp1, Literature-Exp2, and Literature-Exp12 retraining
|
`-- 06_application/
    `-- Parametric response analysis and preliminary mixture screening
```

The interpretation and screening outputs are supporting analyses. The study's two central questions concern empirical formulation development and the effect of experimental data integration on guided prediction.
