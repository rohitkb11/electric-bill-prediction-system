# Electric Bill Prediction System

A simple AI/ML mini-project that forecasts daily household electricity consumption and estimates the bill. **Express** serves the API and dashboard; **Python/scikit-learn** trains and runs the model. The frontend is plain HTML, CSS and JavaScript.

## What it does

- Opens with real sample readings and a working forecast.
- Forecasts the next 7, 14 or 30 days from at least 28 daily kWh readings.
- Shows historical and predicted usage on a chart.
- Estimates the bill: **predicted kWh × per-unit rate + fixed charge**.
- Accepts your own readings, date and tariff values.
- Shows model comparisons, evaluation metrics and tuning results.
- Plots MSE and R² using Matplotlib, plus separate accuracy/precision/recall graphs for derived high-usage detection.

No database, login, frontend build tool or external prediction service is needed.

## Run locally

Install Node.js 20+ and Python **3.12 or 3.13**. The trained model and processed daily data are included, so downloading and training are optional for the first demo.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm install
npm start
```

Open **http://localhost:3000**. On Windows, use `npm.cmd` if PowerShell blocks `npm.ps1`.

On macOS/Linux, replace the install command with `.venv/bin/python -m pip install -r requirements.txt`. Express automatically detects `.venv`; otherwise set `PYTHON_PATH` to your Python executable. `PORT` is optional. `.env.example` documents these settings; no `.env` loader is needed.

## Reproduce the ML work

```powershell
.\.venv\Scripts\python.exe ml/download.py
.\.venv\Scripts\python.exe ml/train.py
npm test
```

Source: [Kaggle Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set), originally [UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). The download script uses Kaggle's public download endpoint. If Kaggle requires sign-in, download manually and place `household_power_consumption.txt` in `data/`.

1. Convert minute-level kW readings into daily kWh. Correct small gaps; fill low-coverage days using the preceding seven-day median.
2. Build past-only usage features (lags and rolling statistics) and calendar features.
3. Split chronologically into 70% training, 15% validation and 15% testing. Exclude imputed targets from training and metrics.
4. Compare Linear Regression and Random Forest. Tune Random Forest using Grid Search with four time-series folds. Retain untuned settings if validation MAE worsens. Random Forest uses bootstrap samples (bagging), random feature subsets and averaged tree predictions.
5. Select by validation MAE, then report held-out MAE, RMSE and R² plus a previous-day baseline. Separately evaluate recursive 30-day forecasts.
6. Refit the selected model on all observed dates for the demo. Save `models/consumption.joblib` and `models/report.json`.

Training generates two Matplotlib graphs in `public/plots/`, displayed on the dashboard. Regenerate them from the saved report without retraining:

```powershell
.\.venv\Scripts\python.exe ml/plot_metrics.py
```

**Metric meanings:** MSE (kWh², lower is better) and R² (higher is better) compare consumption regression. Accuracy, precision and recall apply to a separate high-usage label derived from each regressor's predictions: **high = daily kWh >= the median observed training consumption**. This fixed training-only threshold is applied to actual and predicted test readings. These scores measure high-usage detection, not accuracy of predicted kWh or bills. Precision and recall refer to the high-usage positive class; an undefined score is reported as zero.

## Small project structure

```text
public/       Dashboard: index.html, style.css, app.js
server/       Express API and bill calculation
ml/           Download, preprocessing, training and prediction
data/         Processed daily data and source attribution
models/       Trained model and actual evaluation report
tests/        Input validation, billing and API integration checks
docs/         Mini-project explanation and implementation phases
```

Flow: **Browser → Express → Python model → Express bill calculation → Browser**. Express invokes the local Python script using JSON over stdin/stdout; Python is not a separate API server.

## API

`GET /api/data` returns sample daily history and the training report.

`POST /api/predict` accepts:

```json
{
  "history": [12, 13, 11, 14, 12, 15, 13, 12, 14, 13, 11, 15, 12, 14, 13, 12, 15, 14, 12, 13, 11, 14, 12, 13, 15, 12, 14, 13],
  "lastDate": "2010-11-25",
  "days": 30,
  "rate": 7,
  "fixedCharge": 100
}
```

It returns daily predictions, total kWh, and the energy charge, fixed charge and total bill. Invalid inputs return HTTP 400. The single-user demo permits one prediction at a time and returns 503 while busy.

## Syllabus coverage

| PR12 requirement | Implementation |
| --- | --- |
| Problem definition | Forecast household electricity usage and estimate its cost |
| Dataset collection | Reproducible Kaggle download with source attribution |
| Data preprocessing | Missing values, daily aggregation, feature engineering and chronological split |
| Model development | Two regression models and a persistence baseline |
| Performance evaluation | MAE, RMSE, R² and recursive forecast evaluation |
| Model optimization | Grid Search and time-series cross-validation; before/after metrics |
| Result analysis and documentation | Dashboard results, generated report and project write-up |

## Limits

The data is from **one French household, 2006–2010**; this is a historical forecasting demonstration, not a model validated across Indian homes. Entering your readings does not retrain it. Consumption is predicted by ML; the bill is calculated from your tariff. The default rupee rate is illustrative and does not include slabs, taxes, subsidies or provider-specific rules. Fixed charges apply once to the selected period. Longer forecasts accumulate error, and real-world savings or billing accuracy are not guaranteed.

The raw 2-million-row dataset, local environment and dependencies are excluded from Git. Credit: Georges Hebrail and Alice Berard, EDF R&D; see `data/README.md` for source details.
