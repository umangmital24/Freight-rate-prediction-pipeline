import pandas as pd
from src.features import FreightFeaturePipeline

df = pd.read_csv("data/train_test.csv")
df["date"] = pd.to_datetime(df["date"])

# same time-based split we decided on earlier
holdout_start = pd.to_datetime("2025-09-01")
train_raw = df[df["date"] < holdout_start]
holdout_raw = df[df["date"] >= holdout_start]

pipeline = FreightFeaturePipeline()
pipeline.fit(train_raw)          # learn medians from TRAIN only
train = pipeline.transform(train_raw)
holdout = pipeline.transform(holdout_raw)  # apply the SAME medians to holdout

print("Train missing after transform:", train[["weight","market_index"]].isna().sum().sum())
print("Holdout missing after transform:", holdout[["weight","market_index"]].isna().sum().sum())
print("Median learned from train:", pipeline.weight_median_)
print(train[["day_of_year_sin","day_of_year_cos","day_of_week_sin","day_of_week_cos","days_since_origin"]].head())


val = pd.read_csv("data/validation.csv")
val_transformed = pipeline.transform(val)
print("Any missing lane_te in validation?", val_transformed["lane_te"].isna().sum())
print(val_transformed[["pickup","delivery","lane_te"]].head())

print(val_transformed[["equipment","equipment_Dry_Van","equipment_Reefer","equipment_Flatbed"]].head())


feat_cols = pipeline.feature_columns()
print(train[feat_cols].isna().sum())