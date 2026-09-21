# Freight Rate Prediction Pipeline

An end-to-end machine-learning workflow for predicting freight rates on **future loads**, with a strong focus on realistic validation, leakage prevention, and reproducible feature engineering.

## Problem

The training data covers **Jan–Oct 2025**, while the target validation period covers **Nov–Dec 2025**. Because the real task is forecasting forward in time, a random train/test split would overstate model performance.

I therefore used a **time-based holdout**:

- Train: Jan–Aug 2025
- Holdout: Sep–Oct 2025
- Final predictions: Nov–Dec 2025

This makes validation reflect the actual deployment scenario.

## Key EDA Findings

- **48,000** training rows
- Distance is the dominant signal, with a **0.91 correlation** to posted rate
- Equipment affects rate per mile: Reefer > Flatbed > Dry Van
- Seasonal movement appears in rate-per-mile across the year
- **8 cities and 736 lanes** in validation do not appear in training
- Weight contained missing and negative values that required cleaning
- Some latitude/longitude pairs were unreliable, so the supplied distance feature was used directly

## Modeling Approach

The pipeline performs:

1. Data cleaning and missing-value handling
2. Cyclical date features for day-of-year and day-of-week
3. Smoothed lane target encoding using rate-per-mile
4. Equipment one-hot encoding
5. XGBoost regression on a **log-transformed target**
6. Evaluation against a naive rate-per-mile baseline
7. Refit on the full training dataset before generating final predictions

## Holdout Results

| Model | MAE ($) | RMSE ($) | MAPE |
| --- | ---: | ---: | ---: |
| Naive rate-per-mile | 290.60 | 709.45 | 12.13% |
| **XGBoost (log target)** | **145.79** | **645.82** | **6.98%** |

The XGBoost pipeline reduced holdout MAPE from **12.13% to 6.98%**.

## Feature Pipeline

The model uses:

- distance
- weight
- market index
- quote signal
- cyclical day-of-year features
- cyclical day-of-week features
- days since Jan 1, 2025
- smoothed lane target encoding
- equipment type indicators

Training-derived statistics are learned only from the training split and then reused on the holdout to avoid leakage.

## Repository Structure

```text
src/features.py       Feature engineering + preprocessing pipeline
train.py              Time split, baseline, model training, evaluation, final refit
predict.py            Loads trained artifacts and generates predictions
score.py              Validation/scoring utility
data/                 Assessment datasets and prediction templates
models/               Serialized pipeline/model artifacts
what_i_got_from_data_eda.md
                      EDA findings and modeling rationale
```

## Run

```bash
python -m pip install -r requirements.txt
python train.py
python predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Why this project matters

The main challenge was not simply fitting a regressor. It was designing the evaluation and feature pipeline around the **actual business setting: predicting unseen future freight rates** while handling unseen lanes, imperfect source data, and seasonality.
