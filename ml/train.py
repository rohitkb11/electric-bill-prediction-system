"""Reproducible preprocessing, temporal evaluation, tuning and final model export."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES, forecast, supervised

ROOT = Path(__file__).resolve().parents[1]

def metrics(actual, predicted):
    return {"mae": round(float(mean_absolute_error(actual, predicted)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            "r2": round(float(r2_score(actual, predicted)), 4)}

def preprocess():
    source = ROOT / "data/household_power_consumption.txt"
    if not source.exists():
        raise SystemExit("Raw Kaggle file missing. Run python ml/download.py first.")
    raw = pd.read_csv(source, sep=";", na_values="?", low_memory=False,
                      usecols=["Date", "Time", "Global_active_power"])
    raw.index = pd.to_datetime(raw.Date + " " + raw.Time, format="%d/%m/%Y %H:%M:%S")
    power = pd.to_numeric(raw.Global_active_power, errors="coerce").sort_index()
    power = power[~power.index.duplicated(keep="first")]
    power = power.where(power >= 0)
    counts = power.resample("D").count()
    # kW sampled once each minute -> kWh = sum(kW) / 60.
    # Correct small daily gaps for observed coverage; do not invent long gaps.
    energy = power.resample("D").sum(min_count=1) / 60 * (1440 / counts.replace(0, np.nan))
    energy = energy.where(counts >= 1296)  # require at least 90% daily coverage
    energy = energy.loc[energy.first_valid_index():energy.last_valid_index()]
    imputed = energy.isna()
    for i in range(len(energy)):
        if pd.isna(energy.iloc[i]):
            energy.iloc[i] = energy.iloc[max(0, i - 7):i].median()
    daily = pd.DataFrame({"kwh": energy.round(6), "imputed": imputed})
    daily.index.name = "date"
    daily.to_csv(ROOT / "data/daily.csv")
    return daily, {"raw_rows": len(raw), "missing_power_readings": int(power.isna().sum()),
                   "daily_rows": len(daily), "imputed_days": int(imputed.sum()),
                   "minimum_daily_coverage": 0.9, "start": str(daily.index[0].date()),
                   "end": str(daily.index[-1].date()),
                   "mean_daily_kwh": round(float(daily.kwh.mean()), 3),
                   "min_daily_kwh": round(float(daily.kwh.min()), 3),
                   "max_daily_kwh": round(float(daily.kwh.max()), 3)}

def main():
    (ROOT / "models").mkdir(exist_ok=True)
    daily, stats = preprocess()
    X, y = supervised(daily)
    n = len(X)
    a, b = int(n * .7), int(n * .85)
    observed = ~daily.imputed.reindex(X.index)
    train = observed.iloc[:a]
    val = observed.iloc[a:b]
    test = observed.iloc[b:]
    Xtrain, ytrain = X.iloc[:a].loc[train], y.iloc[:a].loc[train]
    Xval, yval = X.iloc[a:b].loc[val], y.iloc[a:b].loc[val]
    Xtest, ytest = X.iloc[b:].loc[test], y.iloc[b:].loc[test]
    candidates = {
        "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), {}),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
                          {"max_depth": [3, 5, 8], "min_samples_leaf": [3, 8]}),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42),
                              {"n_estimators": [50, 100], "max_depth": [1, 2], "learning_rate": [.03, .1]})
    }
    cv = TimeSeriesSplit(n_splits=4)
    results, trained = [], {}
    for name, (base, grid) in candidates.items():
        print("Training and evaluating:", name, flush=True)
        base.fit(Xtrain, ytrain)
        before = metrics(yval, np.maximum(0, base.predict(Xval)))
        if grid:
            search = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
            search.fit(Xtrain, ytrain)
            tuned, params = search.best_estimator_, search.best_params_
        else:
            tuned, params = base, {}
        after = metrics(yval, np.maximum(0, tuned.predict(Xval)))
        # Keep the untuned model if tuning worsens validation MAE.
        selected = tuned if after["mae"] <= before["mae"] else base
        trained[name] = selected
        results.append({"name": name, "validation_before": before, "validation_after": after,
                        "tuned_params": params, "using_tuned": after["mae"] <= before["mae"],
                        "validation_selected": metrics(yval, np.maximum(0, selected.predict(Xval)))})
    winner = min(results, key=lambda r: r["validation_selected"]["mae"])["name"]
    # Freeze selection using validation only, then evaluate on untouched test dates.
    for result in results:
        model = clone(trained[result["name"]])
        mask = observed.iloc[:b]
        model.fit(X.iloc[:b].loc[mask], y.iloc[:b].loc[mask])
        result["test"] = metrics(ytest, np.maximum(0, model.predict(Xtest)))
    evaluation_model = clone(trained[winner])
    mask = observed.iloc[:b]
    evaluation_model.fit(X.iloc[:b].loc[mask], y.iloc[:b].loc[mask])
    horizon_actual, horizon_pred, windows = [], [], []
    for start in range(b + 28, len(daily) - 29, 30):
        window = daily.iloc[start:start + 30]
        if window.imputed.any():
            continue
        predicted = forecast(evaluation_model, daily.kwh.iloc[:start], daily.index[start - 1], 30)
        horizon_actual.extend(window.kwh.tolist())
        horizon_pred.extend(p["kwh"] for p in predicted)
        windows.append({"start": str(window.index[0].date()), "actual_kwh": round(float(window.kwh.sum()), 3),
                        "predicted_kwh": round(sum(p["kwh"] for p in predicted), 3)})
    final = clone(trained[winner])
    final.fit(X.loc[observed], y.loc[observed])
    joblib.dump(final, ROOT / "models/consumption.joblib", compress=3)
    report = {
        "dataset": "Kaggle / UCI Household Electric Power Consumption",
        "source_url": "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set",
        "statistics": stats, "features": FEATURES, "selected_model": winner,
        "split": {"train_end": str(X.index[a-1].date()), "validation_end": str(X.index[b-1].date()),
                  "test_start": str(X.index[b].date()), "test_end": str(X.index[-1].date()),
                  "observed_train_rows": len(Xtrain), "observed_validation_rows": len(Xval),
                  "observed_test_rows": len(Xtest)},
        "baseline": {"name": "Previous day", "test": metrics(ytest, Xtest.lag_1)},
        "models": results,
        "recursive_30_day_test": {"metrics": metrics(horizon_actual, horizon_pred) if windows else None,
                                  "windows": windows},
        "evaluation_note": "One-day test uses actual past readings. 30-day test uses recursive forecasts. Selection uses validation MAE only; production model is subsequently refit on all observed dates.",
        "preprocessing_note": "Daily kWh from minute kW readings. Days with >=90% coverage are normalized to 1440 minutes; other days use previous 7-day median. Imputed targets are excluded from fitting and metrics. Features use past values only.",
        "limitations": ["One historical French household; performance may differ for other homes.",
                        "No weather or occupancy inputs; long forecasts accumulate error.",
                        "Bills use editable illustrative tariffs, not verified utility rates."]
    }
    (ROOT / "models/report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved model and report. Selected:", winner, flush=True)

if __name__ == "__main__":
    main()
