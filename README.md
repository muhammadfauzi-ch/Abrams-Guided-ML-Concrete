# An Abrams-Guided Residual Learning Framework for Concrete Compressive Strength Prediction Across Multi-Source Datasets

## Overview

This repository provides the reproducible computational workflow for an empirical-prior-guided machine learning framework developed to predict concrete compressive strength across multi-source datasets.

Rather than using Abrams-type formulations only as standalone benchmarks, the framework uses them as empirical priors in residual learning. The study examines how **empirical prior complexity, learner behavior, and data-domain coverage** interact in determining predictive performance and model response.

The workflow compares conventional empirical formulations, direct machine learning, and two guided residual-learning schemes based on an age-incorporated Abrams prior and a multivariable Abrams-type prior.

| Component | Description |
| --- | --- |
| Primary dataset | 937 observations compiled from 33 published studies |
| Experimental datasets | Exp1 and Exp2 |
| Pooled experimental dataset | Exp12, combining Exp1 and Exp2 |
| Target | Concrete compressive strength (MPa) |
| Empirical hierarchy | Classical Abrams, Abrams with curing age, Multivariable Abrams-type model |
| Data-driven models | Direct ML and guided residual ML |
| Interpretation | SHAP, PDP, and ICE analyses |
| Domain-expansion analysis | Literature + Exp1, Literature + Exp2, and Literature + Exp12 retraining |

---

# Research Framework

The computational workflow consists of six main stages:

| Stage | Description |
| --- | --- |
| Data processing | Cleaning, harmonization, feature preparation, and quality control |
| Modeling | Empirical models, direct ML, and guided residual learning |
| Evaluation | Performance metrics and comparative assessment |
| Interpretation | SHAP, PDP, ICE, and response analysis |
| Retraining | Literature-experimental integration using Exp1, Exp2, and Exp12 |
| Application | Parametric response analysis and model-guided mixture screening |

The framework is designed to answer three main questions:

1. How much compressive-strength variation can be represented by progressively richer Abrams-type empirical formulations?
2. Does increasing empirical prior complexity consistently improve residual machine learning?
3. How does the behavior of the guided models change when the development domain is expanded with experimental data?

---

# Dataset

## Literature Dataset

The primary dataset contains **937 observations from 33 published experimental studies**. Each observation represents a concrete mixture evaluated at a particular curing age.

| Parameter | Description |
| --- | --- |
| Source | 33 published experimental studies |
| Number of observations | 937 |
| Main input variables | Age, cement, water, W/B, SCM, fine aggregate, coarse aggregate, superplasticizer, and fiber |
| Target variable | Concrete compressive strength (MPa) |

Observations reported for the same mixture at different curing ages were retained separately because each age corresponds to a different compressive-strength response.

## Experimental Datasets

Two independent experimental datasets were used to examine model adaptation when the development domain was expanded beyond the literature data.

| Dataset | Records | Role |
| --- | ---: | --- |
| Exp1 | 131 | Separate literature-experimental integration |
| Exp2 | 397 | Separate literature-experimental integration |
| Exp12 | 528 | Pooled Exp1 + Exp2 integration |

The corresponding expanded domains contain:

| Modeling domain | Total observations |
| --- | ---: |
| Literature + Exp1 | 1,068 |
| Literature + Exp2 | 1,334 |
| Literature + Exp12 | 1,465 |

These scenarios represent **integrated retraining analyses**, not independent external validation, because the experimental observations are incorporated into model development before evaluation on the held-out portion of each expanded domain.

---

# Modeling Framework

## Empirical Model Hierarchy

Three empirical formulations are evaluated progressively:

| Model | Main information represented |
| --- | --- |
| Classical Abrams | W/B |
| Abrams with curing age | W/B + curing age |
| Multivariable Abrams-type model | W/B, age, and broader mixture composition |

The hierarchy is used to determine how the empirical representation changes as progressively more information is introduced before machine learning.

## Guided Residual Learning

The age-incorporated Abrams model and the Multivariable Abrams-type model are subsequently used as empirical priors.

For each observation, the residual learning target is defined conceptually as:

```text
Residual = Measured strength - Empirical prediction
```

The machine learning model predicts this remaining residual, and the final compressive-strength prediction is reconstructed as:

```text
Final prediction = Empirical prediction + Predicted residual
```

Using the same learners with two empirical priors enables the effect of **prior complexity** to be evaluated without changing the residual-learning structure.

---

# Machine Learning Models

The following learners are evaluated:

| Model | Category |
| --- | --- |
| MLR | Linear regression |
| SVR | Kernel-based regression |
| KNN | Instance-based learning |
| RF | Bagging tree ensemble |
| GBR | Gradient boosting |
| XGBoost | Extreme gradient boosting |
| CatBoost | Gradient boosting |
| DNN | Deep neural network |
| AutoGluon | Automated machine learning |

The same learning families are used where applicable for direct ML and empirical-prior-guided residual learning to enable consistent comparison.

---

# Experimental Design

## Literature-Only Benchmark

The 937 literature observations are divided into training and testing subsets using a fixed 80:20 partition.

This stage compares:

```text
Empirical models
        ↓
Direct machine learning
        ↓
Abrams-guided residual learning
        ↓
Multivariable-guided residual learning
```

The objective is to determine how predictive performance changes as empirical information is introduced before machine learning.

## Literature-Experimental Retraining

The second stage evaluates model adaptation after experimental observations are incorporated into the development domain.

Three expanded domains are considered:

```text
Literature + Exp1
Literature + Exp2
Literature + Exp12
```

Each expanded dataset is repartitioned into training and held-out testing subsets before the model families are refitted.

This analysis evaluates whether the benefit of empirical guidance and residual learning changes as the available data domain becomes broader.

---

# Model Evaluation

Model performance is evaluated using complementary metrics:

| Metric | Purpose |
| --- | --- |
| R² | Explained variance |
| RMSE | Absolute prediction error with greater penalty for large errors |
| MAE | Mean absolute prediction error |
| MBE | Directional prediction bias |
| MAPE | Relative prediction error |
| RRMSE | Error normalized by mean compressive strength |
| A20 | Fraction of predictions within ±20% of measured strength |

---

# Model Interpretation

Interpretability analyses are used to examine not only which variables influence the predictions, but also how the learned response changes with empirical prior complexity and data-domain expansion.

The workflow includes:

- SHAP global feature importance
- SHAP summary analysis
- SHAP dependence analysis
- Partial dependence plots
- Individual conditional expectation analysis
- Parametric response analysis

Particular attention is given to curing age and W/B because they provide a connection between the learned model behavior and established concrete-strength relationships.

---

# Environment Setup

## Requirements

- Python 3.11
- Conda
- Windows or Linux
- GPU optional for computationally intensive DNN or AutoML training

Create the environment using:

```bash
conda env create -f environment.yml
conda activate abrams-ml
```

If required:

```bash
pip install -r requirements.txt
```

---

# Core Dependencies

| Package | Purpose |
| --- | --- |
| NumPy | Numerical computation |
| Pandas | Dataset manipulation |
| Scikit-learn | Classical machine learning |
| XGBoost | Gradient boosting |
| CatBoost | Gradient boosting |
| PyTorch | Deep neural network |
| AutoGluon | Automated machine learning |
| SHAP | Model interpretation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |

Exact package versions are provided in `environment.yml` and/or `requirements.txt`.

---

# Reproducibility Workflow

The repository follows the existing script structure:

```text
scripts/
│
├── 01_data_processing/
│   └── Data cleaning, harmonization, and dataset preparation
│
├── 02_modeling/
│   └── Empirical models, direct ML, and guided residual-learning models
│
├── 03_evaluation/
│   └── Performance metrics and comparative model assessment
│
├── 04_interpretation/
│   └── SHAP, PDP, ICE, and model-response interpretation
│
├── 05_retraining/
│   └── Literature-Exp1, Literature-Exp2, and Literature-Exp12 retraining
│
└── 06_application/
    └── Parametric response analysis and model-guided mixture screening
