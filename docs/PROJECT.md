# Mini-project explanation

## Problem definition

Households need an estimate of upcoming energy use and its cost. Given consecutive daily electricity readings, predict consumption for the next 7â€“30 days and convert that prediction into an estimated bill using an editable rate and fixed charge.

## Dataset and preprocessing

The project uses one dataset, `data/daily.csv`, with 1,440 daily records and three columns: `date`, `kwh` and `imputed`. Both training and the dashboard read this same CSV. Its values were derived from Kaggle/UCI Individual Household Electric Power Consumption; it is an attributed daily adaptation, not a separate original Kaggle dataset. No raw TXT or ZIP is needed.

During original CSV preparation, minute-level kW readings were summed and divided by 60 to give daily kWh. Days with at least 90% valid readings were normalized for coverage; other days used the preceding seven-day median. These assumptions can introduce error. The training script now reads the CSV directly, sorts dates, removes duplicate dates and handles additional invalid/missing readings. Imputed days supply history but are excluded as training/evaluation targets.

Features are usage 1, 7 and 14 days ago, seven- and 28-day averages, seven-day standard deviation, and cyclical weekday/month values. Every usage feature is computed strictly from the past. Linear Regression uses standardized features; tree models do not require scaling.

## Development, evaluation and optimization

Two models are compared: Linear Regression and Random Forest Regressor. A previous-day prediction provides a simple baseline. Random Forest uses bagging: each tree trains on a bootstrap sample (sampling rows with replacement), and the forest averages tree predictions. `bootstrap=True` makes this explicit; `max_features="sqrt"` also selects random feature subsets at each split. The time-ordered data is divided into training, validation and testing periods. Model parameters are tuned only within training using four time-series folds. Validation MAE determines the retained settings and winning model. The test period does not determine selection.

MAE describes average absolute error in kWh; MSE is mean squared error in kWh², and RMSE is its square root. R² compares prediction errors with a constant mean predictor and can be negative. The dashboard shows real generated values; a complex model is not assumed to outperform the baseline.

`ml/plot_metrics.py` uses Matplotlib to plot test MSE and R² for both models. A separate graph compares accuracy, precision and recall for derived high-usage detection. A day is high usage when its kWh is at least the median observed training consumption; the same training-only threshold labels actual test values and regression predictions. Accuracy measures correct high/normal labels; precision measures how many predicted-high days were actually high; recall measures how many actual-high days were detected. These are secondary classification scores, not regression accuracy. No additional classifier is trained. Undefined precision/recall values are set to zero.

Next-day evaluation has actual earlier readings available. For the 30-day evaluation, each predicted value is fed into the following day's features. The report includes complete, non-imputed 30-day windows in the held-out test period. These metrics better represent a multi-day forecast. After evaluation, the selected model is fitted on all observed dates for the interactive demo.

## Application and bill calculation

Express serves the frontend and exposes two endpoints: one for dataset/model information and one for predictions. The prediction endpoint validates the inputs and invokes Python's saved model. The bill is `total predicted kWh Ã— rate + one fixed charge`, rounded to two decimals. No model is trained to predict a bill because the dataset contains no bill labels.

## Results and conclusion

See `models/report.json` and the dashboard for actual model scores, chosen settings and forecast results. Discuss the baseline comparison, whether tuning improved validation, and the gap between one-day and 30-day results. The demonstration supports all seven PR12 stages, while remaining small enough to explain in a practical examination.

The principal limitations are one-household historical data, missing weather/occupancy information, recursive prediction error and a simplified tariff. A more representative multi-household dataset and verified utility tariffs would be needed before applying it to real billing.

## Recorded run

Trained directly from 1,440 CSV daily records, including 22 flagged filled days excluded as targets. Selected **Random Forest**.

| Model | Test MAE (kWh) | Test RMSE (kWh) | Test R2 |
| --- | --- | --- | --- |
| Linear Regression | 4.199 | 5.466 | 0.441 |
| Random Forest | 3.834 | 5.198 | 0.494 |
| Previous-day baseline | 4.564 | 6.525 | 0.203 |

Selected model validation MAE changed from 4.524 to 4.467 kWh after tuning. Recursive 30-day test MAE was 4.394 kWh across 5 complete windows. This is daily error within those forecasts, not monthly bill error.

## Matplotlib model comparison

| Model | Test MSE (kWh squared) | Test R2 | High-usage accuracy | High-usage precision | High-usage recall |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | 29.883 | 0.441 | 76.73% | 68.18% | 63.38% |
| Random Forest | 27.022 | 0.494 | 77.23% | 68.12% | 66.20% |

High usage is defined using the training median (25.305633 kWh). Random Forest has lower MSE and higher R2 on this held-out test, so it performs better for consumption regression here. It also has slightly higher high-usage accuracy and recall; Linear Regression has slightly higher precision. These results are specific to this dataset and split.
