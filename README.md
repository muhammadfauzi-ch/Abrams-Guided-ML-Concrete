# Abrams-Guided Machine Learning Framework for Predicting Concrete Compressive Strength Across Multi-Source Datasets

## Overview

This repository provides a reproducible machine learning workflow integrating empirical concrete knowledge with data-driven modeling for concrete compressive strength prediction.

The framework investigates whether incorporating Abrams-based empirical relationships as prior knowledge can improve prediction accuracy, robustness, and interpretability compared with conventional machine learning approaches.

The workflow includes:

| Component               | Description                                        |
| ----------------------- | -------------------------------------------------- |
| Dataset                 | Multi-source concrete compressive strength dataset |
| Literature dataset      | 937 concrete mixtures from 33 studies              |
| Experimental validation | Exp1 and Exp2 independent datasets                 |
| Target                  | Concrete compressive strength (MPa)                |
| Modeling strategy       | Classical empirical models, ML, physics-guided ML  |
| Interpretation          | SHAP, PDP, ICE                                     |

---

# Research Framework

The framework consists of five computational stages:

| Stage              | Description                                         |
| ------------------ | --------------------------------------------------- |
| Data processing    | Dataset cleaning, harmonization, and integration    |
| Empirical modeling | Classical Abrams and multivariable empirical models |
| Machine learning   | Conventional ML and guided ML development           |
| Validation         | External experimental-literature retraining     |
| Interpretation     | Model explanation and feature contribution analysis |

---

# Dataset

## Literature Dataset

The primary dataset consists of:

| Parameter         | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| Source            | 33 published experimental studies                                       |
| Number of records | 937 concrete mixtures                                                   |
| Input variables   | Cement, water, SCM, aggregates, admixture, fiber, W/B ratio, curing age |
| Target variable   | Concrete compressive strength (MPa)                                     |

## Experimental Dataset

Independent experimental datasets:

| Dataset     | Purpose                                       |
| ----------- | --------------------------------------------- |
| Exp1        | External validation                           |
| Exp2        | External validation                           |
| Exp1 + Exp2 | Literature–experimental integrated retraining |

---

# Modeling Framework

## Empirical Models

| Model                         | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| Classical Abrams              | Traditional water–cement strength relationship |
| Abrams + Age                  | Age-enhanced empirical formulation             |
| Multivariable empirical model | Extended formulation using mixture variables   |

---

## Machine Learning Models

The following algorithms are evaluated:

| Model     | Category                  |
| --------- | ------------------------- |
| MLR       | Linear regression         |
| SVR       | Kernel-based regression   |
| KNN       | Instance-based learning   |
| RF        | Tree ensemble             |
| GBR       | Gradient boosting         |
| XGBoost   | Extreme gradient boosting |
| CatBoost  | Ordered boosting          |
| DNN       | Neural network            |
| AutoGluon | Automated ML ensemble     |

---

# Environment Setup

## Requirements

* Python 3.11
* Conda environment
* Windows/Linux
* NVIDIA GPU recommended for DNN and AutoGluon training

---

## Create Environment

```bash
conda env create -f environment.yml

conda activate abrams-ml
```

Install additional packages:

```bash
pip install -r requirements.txt
```

---

# Package Versions

| Package      | Purpose                       |
| ------------ | ----------------------------- |
| Python       | Programming language          |
| NumPy        | Numerical computation         |
| Pandas       | Dataset manipulation          |
| Scikit-learn | Classical machine learning    |
| XGBoost      | Gradient boosting model       |
| CatBoost     | Gradient boosting model       |
| PyTorch      | Deep neural network framework |
| AutoGluon    | Automated machine learning    |
| SHAP         | Model interpretation          |
| Matplotlib   | Visualization                 |
| Seaborn      | Statistical visualization     |

---

# Reproducibility Workflow

Run scripts sequentially:

## 1. Data Processing

```bash
scripts/01_data_processing/
```

Purpose:

* Dataset cleaning
* Feature harmonization
* Dataset preparation

---

## 2. Model Development

```bash
scripts/02_modeling/
```

Purpose:

* Train empirical models
* Train ML models
* Develop Abrams-guided models

---

## 3. Validation

```bash
scripts/03_validation/
```

Purpose:

* External validation
* Literature–experimental integration

---

## 4. Evaluation

```bash
scripts/04_evaluation/
```

Purpose:

* Performance metrics
* Prediction analysis

---

## 5. Interpretation

```bash
scripts/05_interpretation/
```

Purpose:

* SHAP analysis
* PDP analysis
* ICE analysis

---

# Citation

If you use this repository, please cite:

Fauzi, M.; Potiyaraj, P.; Prasittisopin, L.

*"An Abrams Guided Comparative Machine Learning Framework for Predicting Concrete Compressive Strength Across Multi Source Datasets."*

---
