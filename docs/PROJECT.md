# Mini-project explanation

## Problem definition

Households need an estimate of upcoming energy use and its cost. Given consecutive daily electricity readings, predict consumption for the next 7â€“30 days and convert that prediction into an estimated bill using an editable rate and fixed charge.

## Dataset and preprocessing

The Kaggle/UCI Individual Household Electric Power Consumption dataset records minute-level measurements for one household. `Global_active_power` is in kW. Summing the minute measurements and dividing by 60 gives daily kWh.

The script parses date/time, marks invalid power values as missing and removes duplicate timestamps. Days with at least 90% valid readings are normalized for missing coverage. Other days use the preceding seven-day median. This assumes the available minutes represent the day and the previous week reasonably represents a missing day; both assumptions can introduce error. Imputed days supply history but are excluded as training/evaluation targets.

Features are usage 1, 7 and 14 days ago, seven- and 28-day averages, seven-day standard deviation, and cyclical weekday/month values. Every usage feature is computed strictly from the past. Linear Regression uses standardized features; tree models do not require scaling.

## Development, evaluation and optimization

Two models are compared: Linear Regression and Random Forest Regressor. A previous-day prediction provides a simple baseline. Random Forest uses bagging: each tree trains on a bootstrap sample (sampling rows with replacement), and the forest averages tree predictions. `bootstrap=True` makes this explicit; `max_features="sqrt"` also selects random feature subsets at each split. The time-ordered data is divided into training, validation and testing periods. Model parameters are tuned only within training using four time-series folds. Validation MAE determines the retained settings and winning model. The test period does not determine selection.

MAE describes average absolute error in kWh; RMSE penalizes large errors more strongly; RÂ² compares prediction errors with a constant mean predictor and can be negative. The dashboard shows real generated values; a complex model is not assumed to outperform the baseline.

Next-day evaluation has actual earlier readings available. For the 30-day evaluation, each predicted value is fed into the following day's features. The report includes complete, non-imputed 30-day windows in the held-out test period. These metrics better represent a multi-day forecast. After evaluation, the selected model is fitted on all observed dates for the interactive demo.

## Application and bill calculation

Express serves the frontend and exposes two endpoints: one for dataset/model information and one for predictions. The prediction endpoint validates the inputs and invokes Python's saved model. The bill is `total predicted kWh Ã— rate + one fixed charge`, rounded to two decimals. No model is trained to predict a bill because the dataset contains no bill labels.

## Results and conclusion

See `models/report.json` and the dashboard for actual model scores, chosen settings and forecast results. Discuss the baseline comparison, whether tuning improved validation, and the gap between one-day and 30-day results. The demonstration supports all seven PR12 stages, while remaining small enough to explain in a practical examination.

The principal limitations are one-household historical data, missing weather/occupancy information, recursive prediction error and a simplified tariff. A more representative multi-household dataset and verified utility tariffs would be needed before applying it to real billing.

## Recorded run

Processed 2,075,259 readings into 1,440 daily records, with 22 low-coverage days filled. Selected **Random Forest**.

| Model | Test MAE (kWh) | Test RMSE (kWh) | Test R2 |
| --- | --- | --- | --- |
| Linear Regression | 4.199 | 5.466 | 0.441 |
| Random Forest | 3.834 | 5.198 | 0.494 |
| Previous-day baseline | 4.564 | 6.525 | 0.203 |

Selected model validation MAE changed from 4.524 to 4.467 kWh after tuning. Recursive 30-day test MAE was 4.394 kWh across 5 complete windows. This is daily error within those forecasts, not monthly bill error.
