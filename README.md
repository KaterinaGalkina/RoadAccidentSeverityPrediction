# Road Accident Severity Prediction

Binary classification project predicting the severity of road accidents in France, built as part of the M1 Informatique curriculum at Paris Dauphine-PSL (2026).

**Authors:** Nerdjes Ahdad & Ekaterina Galkina

---

## Overview

Using official French road accident data (2019-2024) from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-2005-a-2024/), we trained machine learning models to predict whether a road accident victim will be severely injured or not.

The dataset covers **621,000+ accident records** across four relational tables: accident circumstances, location, vehicles, and people involved.

**Best result: Random Forest + undersampling -> F1 = 0.58, AUC = 0.88 on the test set.**

---

## Repository Structure

```
├── data/                        # Raw data by year (2019-2024), 4 CSV tables per year
│   ├── 2019/
│   │   ├── caract-2019.csv
│   │   ├── lieux-2019.csv
│   │   ├── usagers-2019.csv
│   │   └── vehicules-2019.csv
│   └── ...
├── data_clean/                  # Cleaned and merged tables (output of preprocessing notebooks)
├── models/                      # Saved models from hyperparameter tuning (GridSearchCV)
│   ├── GradientBoosting-hpt-os-*/
│   ├── RandomForest-hpt-us-*/
│   └── ...
├── results/                     # Test set evaluation: metrics.json + confusion matrix + ROC curve
├── caract.ipynb                 # Preprocessing & EDA — accident circumstances table
├── lieux.ipynb                  # Preprocessing & EDA — location table
├── usagers.ipynb                # Preprocessing & EDA — people table
├── vehicules.ipynb              # Preprocessing & EDA — vehicles table
├── all_features.ipynb           # Final join of all tables into one dataset
├── main.py                      # Full pipeline: hyperparameter tuning + model evaluation
└── report_NerdjesAhdad_EkaterinaGalkina.pdf   # Full project report
```

---

## Problem Statement

The target variable `grav_bin` is a binary label derived from the original 4-class severity score:

| Original label | Meaning | Binary label |
|---|---|---|
| 1 - Unharmed | No injury | 0 - Not severe |
| 4 - Minor injury | Light injury | 0 - Not severe |
| 3 - Hospitalized | Serious injury | 1 - Severe |
| 2 - Killed | Fatal | 1 - Severe |

The dataset is **imbalanced**: ~82% non-severe, ~18% severe. We use F1-score on the minority class as the main evaluation metric.

---

## Methodology

### Data & Preprocessing

Each year's data comes as 4 separate CSV files. Preprocessing is handled in individual notebooks (one per table) before being joined into a final dataset in `all_features.ipynb`.

Key preprocessing steps:
- Missing value handling (imputation or row removal depending on feature and rate)
- Outlier removal (e.g. ages above 100, invalid speed limits)
- Geographic filtering to metropolitan France (lat 41-51°, lon −5-10°)
- Cyclical encoding of time features: hour, day, month, weekday → sin/cos pairs
- One-hot encoding of safety equipment (seatbelt, helmet, gloves, etc.)
- Speed limit (`vma`) imputation using median per road category (`catr`)

### Models

Three classifiers were compared, each tuned with GridSearchCV (5-fold stratified cross-validation):

| Model | Key hyperparameters tuned |
|---|---|
| Logistic Regression | `C`, `penalty` (L1/L2), `class_weight` |
| Random Forest | `n_estimators`, `max_depth`, `class_weight` |
| Gradient Boosting | `n_estimators`, `learning_rate` |

Each model was evaluated under three sampling strategies: no resampling, undersampling, and oversampling.

### Results

| Model | Sampling | F1 (severe) | AUC |
|---|---|---|---|
| Logistic Regression | oversampling | 0.55 | - |
| Gradient Boosting | oversampling | 0.57 | - |
| **Random Forest** | **undersampling** | **0.58** | **0.88** |

The best model (Random Forest + undersampling) correctly identifies **83% of severe accidents** (recall), with a precision of 0.45 — meaning roughly half of predicted severe cases are false alarms, which is expected given class imbalance.

### Most informative features

Consistent across models: seatbelt use (`ceinture`), geographic coordinates (`lat`, `long`), vehicle category (`catv`), victim age, and obstacle type (`obs`, `obsm`).

---

## How to Run

### Option 1 - Docker (recommended)

A ready-to-use Docker image is available on Docker Hub, with all dependencies pre-installed:

```bash
docker pull basiliskk/road-accident-ml:v1
docker run -p 8888:8888 -v $(pwd):/app basiliskk/road-accident-ml:v1
```

Then open `http://localhost:8888` in your browser.

-> [Docker Hub — basiliskk/road-accident-ml](https://hub.docker.com/r/basiliskk/road-accident-ml)

### Option 2 - Local install

```bash
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 \
            imbalanced-learn==0.12.4 matplotlib==3.9.2 seaborn==0.13.2 \
            jupyter==1.1.1
jupyter notebook
```

### Running the full pipeline

Open the notebooks in this order:
1. `usagers.ipynb`
2. `caract.ipynb`
3. `lieux.ipynb`
4. `vehicules.ipynb`
5. `all_features.ipynb`
6. `main.py` (runs hyperparameter tuning + test set evaluation)

---

## Data Source

**Bases de données annuelles des accidents corporels de la circulation routière - Années 2005 à 2024**
Published by the French Ministry of the Interior, collected by law enforcement at accident scenes.
[https://www.data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-2005-a-2024/)
