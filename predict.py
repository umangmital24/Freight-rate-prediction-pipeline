import json
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load("models/pipeline.joblib")
model = joblib.load("models/model.joblib")
feat_cols = json.load(open("models/feature_columns.json"))

val_raw = pd.read_csv("data/validation.csv")
val = pipeline.transform(val_raw)



pred_log = model.predict(val[feat_cols])
pred = np.exp(pred_log)

template = pd.read_csv("data/validation_predictions_template.csv")
template["predicted_rate"] = np.round(pred, 2)
template.to_csv("validation_predictions.csv", index=False)

print("Saved validation_predictions.csv")
print(template.head())
print(template["predicted_rate"].describe())


dec_raw = pd.read_csv("data/december_chart_inputs.csv")
dec_features = dec_raw.drop(columns=["predicted_rate"])
dec = pipeline.transform(dec_features)

dec_pred_log = model.predict(dec[feat_cols])
dec_pred = np.exp(dec_pred_log)

dec_raw["predicted_rate"] = np.round(dec_pred, 2)
dec_raw.to_csv("data/december_chart_inputs.csv", index=False)

print("Saved december_chart_inputs.csv")
print(dec_raw[["date", "predicted_rate"]])