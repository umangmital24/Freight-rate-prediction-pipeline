import pandas as pd
from src.features import FreightFeaturePipeline

df = pd.read_csv("data/train_test.csv")
df["date"] = pd.to_datetime(df["date"])

holdout_start = pd.to_datetime("2025-09-01")
train_raw = df[df["date"] < holdout_start]
holdout_raw = df[df["date"] >= holdout_start]

pipeline = FreightFeaturePipeline()
pipeline.fit(train_raw)
train = pipeline.transform(train_raw)
holdout = pipeline.transform(holdout_raw)

# naive baseline: predict using average rate-per-mile x distance
naive_pred = holdout["distance"] * pipeline.global_rpm_
actual = holdout["posted_rate"]

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

mae = mean_absolute_error(actual, naive_pred)
rmse = mean_squared_error(actual, naive_pred) ** 0.5
mape = mean_absolute_percentage_error(actual, naive_pred) * 100

print("Naive baseline:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")


import numpy as np
from xgboost import XGBRegressor

feat_cols = pipeline.feature_columns()
X_train = train[feat_cols]
y_train = np.log(train["posted_rate"])   # log-transform the target

X_holdout = holdout[feat_cols]
y_holdout_actual = holdout["posted_rate"]

model = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=0,
)
model.fit(X_train, y_train)

pred_log = model.predict(X_holdout)
pred = np.exp(pred_log)   # convert back from log-space

mae = mean_absolute_error(y_holdout_actual, pred)
rmse = mean_squared_error(y_holdout_actual, pred) ** 0.5
mape = mean_absolute_percentage_error(y_holdout_actual, pred) * 100

print("\nXGBoost model:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")



import joblib
import json
import os

os.makedirs("models", exist_ok=True)

print("\nRefitting on full dataset...")
final_pipeline = FreightFeaturePipeline()
final_pipeline.fit(df)
full = final_pipeline.transform(df)

X_full = full[feat_cols]
y_full = np.log(full["posted_rate"])

final_model = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=0,
)
final_model.fit(X_full, y_full)

joblib.dump(final_pipeline, "models/pipeline.joblib")
joblib.dump(final_model, "models/model.joblib")
json.dump(feat_cols, open("models/feature_columns.json", "w"))

print("Saved final pipeline and model to models/")