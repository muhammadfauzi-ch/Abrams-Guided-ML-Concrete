# Dataset Description

This directory contains the datasets used for developing and evaluating the Abrams-guided physics-informed machine learning framework for concrete compressive strength prediction.

## Dataset Organization

### Raw Dataset

The `raw` folder contains the original datasets collected from literature and experimental validation studies.

Files:

- `literature_raw.xlsx`
  
  Literature-based concrete mixture dataset collected from published studies.

- `experimental_exp1_raw.xlsx`
  
  External experimental dataset used for independent validation (Exp1).

- `experimental_exp2_raw.xlsx`
  
  External experimental dataset used for independent validation (Exp2).

---

### Processed Dataset

The `processed` folder contains harmonized datasets after preprocessing, quality control, feature engineering, and schema alignment.

Files:

- `literature_train_ready.csv`

  Final literature dataset prepared for machine learning model development.

- `finaldataexp1.csv`

  Harmonized Exp1 dataset compatible with the literature dataset structure.

- `finaldataexp2.csv`

  Harmonized Exp2 dataset compatible with the literature dataset structure.

- `finaldataexp12.csv`

  Combined experimental dataset (Exp1 + Exp2) used for integrated validation and retraining analysis.

---

## Target Variable

The prediction target is:

- Concrete compressive strength (MPa)

## Main Input Features

The datasets include concrete mixture design parameters, including:

- Cement content
- Water content
- Water-to-binder ratio
- Supplementary cementitious materials
- Fine aggregate
- Coarse aggregate
- Chemical admixture
- Curing age

Additional physics-guided features derived from Abrams' law are generated during model development.

## Data Usage

The processed datasets are directly used for:

1. Physics-based baseline modelling
2. Machine learning benchmark modelling
3. Abrams-guided machine learning development
4. External validation
5. Literature-experimental integrated retraining