## EDA Findings
- 48,000 training rows, Jan 1 – Oct 31 2025. Validation is Nov 1 – Dec 31 2025 (entirely future data).
- distance correlates 0.91 with posted_rate — dominant driver.
- weight: 300 missing, 292 negative (data-entry bug, sign flipped)
- market_index: 374 missing
- Equipment type changes rate/mile: Reefer $2.38 > Flatbed $2.29 > Dry Van $2.12
- Rate/mile has seasonality: ~$2.10 in Jan rising to ~$2.33 in June
- 8 cities and 736 lanes in validation.csv never appear in training data

## Data Cleaning
- weight: fixed 292 negative values with .abs(), filled 300 missing with training-median (31496)
- market_index: filled 374 missing with training-median (1.0558)
- lat/lon coordinates unreliable for some city pairs (e.g. Shreveport-New Orleans, ratio 9.6x vs typical 1.15-1.2x)
  -> decided to use the given `distance` column directly, not derive distance from coordinates


  ## Model Comparison (holdout: Sep-Oct 2025)
| Model              | MAE ($) | RMSE ($) | MAPE  |
|---------------------|--------:|---------:|------:|
| Naive rate-per-mile  |  290.60 |  709.45  | 12.13% |
| XGBoost (log target) |  145.79 |  645.82  |  6.98% |


validation.csv covers Nov 1 to Dec 31, 2025, entirely after your training data ends (Oct 31). This is the core insight of the whole assignment.

Here's why that matters: If you took train_test.csv and randomly shuffled rows into train/test buckets (the usual default), your test set would contain dates scattered across Jan–Oct, mixed in with your training dates. The model would basically be interpolating between dates it's already seen nearby. But your real job is to predict Nov–Dec — dates the model has never seen, months into the future. A random split would make you think your model is more accurate than it really is, because it's never actually tested on "predict data I haven't seen yet."

The fix: split your training data itself by time — train on the early months, test/validate on the most recent months you do have. That mimics the real task (forecasting forward) instead of faking it.