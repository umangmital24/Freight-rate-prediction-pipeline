import numpy as np
import pandas as pd

class FreightFeaturePipeline:
    def clean(self, df):
        df = df.copy()
        df["weight"] = df["weight"].abs()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def fit(self, df):
        df = self.clean(df)
        self.weight_median_ = df["weight"].median()
        self.market_index_median_ = df["market_index"].median()
        self.quote_signal_median_ = df["quote_signal"].median()
        self.fit_lane_encoding(df)
        return self

    def transform(self, df):
        df = self.clean(df)

        if "market_index" not in df.columns:
            df["market_index"] = np.nan
        if "quote_signal" not in df.columns:
            df["quote_signal"] = np.nan

        df["weight"] = df["weight"].fillna(self.weight_median_)
        df["market_index"] = df["market_index"].fillna(self.market_index_median_)
        df["quote_signal"] = df["quote_signal"].fillna(self.quote_signal_median_)
        df = self.add_date_features(df)
        df = self.apply_lane_encoding(df)
        df = self.apply_equipment_encoding(df)
        return df
    def add_date_features(self, df):
        doy = df["date"].dt.dayofyear
        df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

        dow = df["date"].dt.dayofweek
        df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

        origin = pd.Timestamp("2025-01-01")
        df["days_since_origin"] = (df["date"] - origin).dt.days.astype(float)

        return df
    def fit_lane_encoding(self, df):
        rpm = df["posted_rate"] / df["distance"]
        self.global_rpm_ = rpm.mean()
        work = df.assign(rpm=rpm)

        agg = work.groupby(["pickup", "delivery"])["rpm"].agg(["mean", "count"]).reset_index()
        smoothing = 20
        agg["lane_te"] = (agg["mean"] * agg["count"] + self.global_rpm_ * smoothing) / (agg["count"] + smoothing)
        self.lane_stats_ = agg[["pickup", "delivery", "lane_te"]]   


    def apply_lane_encoding(self, df):
        df = df.merge(self.lane_stats_, on=["pickup", "delivery"], how="left")
        df["lane_te"] = df["lane_te"].fillna(self.global_rpm_)
        return df 

    def apply_equipment_encoding(self, df):
        for level in ["Dry Van", "Reefer", "Flatbed"]:
            col_name = "equipment_" + level.replace(" ", "_")
            df[col_name] = (df["equipment"] == level).astype(int)
        return df

    def feature_columns(self):
        return [
            "distance", "weight", "market_index", "quote_signal",
            "day_of_year_sin", "day_of_year_cos",
            "day_of_week_sin", "day_of_week_cos",
            "days_since_origin", "lane_te",
            "equipment_Dry_Van", "equipment_Reefer", "equipment_Flatbed",
        ]