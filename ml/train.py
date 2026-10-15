"""Reproducible preprocessing, temporal evaluation, tuning and final model export."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES, forecast, supervised
from plot_metrics import plot_comparison

ROOT = Path(__file__).resolve().parents[1]

def metrics(actual, predicted):
    return {"mae": round(float(mean_absolute_error(actual, predicted)), 4),
            "mse": round(float(mean_squared_error(actual, predicted)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            "r2": round(float(r2_score(actual, predicted)), 4)}

def preprocess():
    source = ROOT / "data/daily.csv"
    if not source.exists():
        raise SystemExit("Dataset missing: data/daily.csv")
    daily = pd.read_csv(source, parse_dates=["date"], index_col="date").sort_index()
    daily = daily[~daily.index.duplicated(keep="first")].asfreq("D")
    energy = pd.to_numeric(daily.kwh, errors="coerce")
    energy = energy.where(energy >= 0)
    energy = energy.loc[energy.first_valid_index():energy.last_valid_index()]
    # Retain the source's filled-day flags and mark any newly missing readings.
    imputed = daily.imputed.reindex(energy.index).fillna(False).astype(bool) | energy.isna()
    for i in range(len(energy)):
        if pd.isna(energy.iloc[i]):
            energy.iloc[i] = energy.iloc[max(0, i - 7):i].median()
    daily = pd.DataFrame({"kwh": energy.round(6), "imputed": imputed})
    daily.index.name = "date"
    return daily, {"daily_rows": len(daily), "imputed_days": int(imputed.sum()),
                   "start": str(daily.index[0].date()),
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
    # Define the positive class using training data only, before evaluating test data.
    high_usage_threshold = float(ytrain.median())
    candidates = {
        "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), {}),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=5, bootstrap=True,
                                               max_features="sqrt", random_state=42, n_jobs=1),
                          {"max_depth": [3, 5, 8], "min_samples_leaf": [3, 8]})
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
        predictions = np.maximum(0, model.predict(Xtest))
        result["test"] = metrics(ytest, predictions)
        actual_high = ytest >= high_usage_threshold
        predicted_high = predictions >= high_usage_threshold
        result["high_usage_test"] = {
            "accuracy": round(float(accuracy_score(actual_high, predicted_high)), 4),
            "precision": round(float(precision_score(actual_high, predicted_high, zero_division=0)), 4),
            "recall": round(float(recall_score(actual_high, predicted_high, zero_division=0)), 4)
        }
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
        "dataset": "daily.csv: daily household consumption derived from Kaggle / UCI",
        "source_url": "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set",
        "statistics": stats, "features": FEATURES, "selected_model": winner,
        "split": {"train_end": str(X.index[a-1].date()), "validation_end": str(X.index[b-1].date()),
                  "test_start": str(X.index[b].date()), "test_end": str(X.index[-1].date()),
                  "observed_train_rows": len(Xtrain), "observed_validation_rows": len(Xval),
                  "observed_test_rows": len(Xtest)},
        "baseline": {"name": "Previous day", "test": metrics(ytest, Xtest.lag_1)},
        "models": results,
        "high_usage_classification": {
            "threshold_kwh": high_usage_threshold,
            "definition": "High usage = daily kWh >= median observed training consumption. Actual and predicted kWh use the same fixed threshold.",
            "note": "Secondary classification derived from regression predictions; accuracy, precision and recall do not measure regression accuracy.",
            "test_positive_days": int((ytest >= high_usage_threshold).sum()),
            "test_total_days": len(ytest)
        },
        "recursive_30_day_test": {"metrics": metrics(horizon_actual, horizon_pred) if windows else None,
                                  "windows": windows},
        "evaluation_note": "One-day test uses actual past readings. 30-day test uses recursive forecasts. Selection uses validation MAE only; production model is subsequently refit on all observed dates.",
        "preprocessing_note": "Read data/daily.csv, sort dates, remove duplicate dates and handle invalid/missing kWh using the preceding 7-day median. The imputed column identifies filled days, which are excluded as fitting/evaluation targets. Features use past values only. Original Kaggle data was aggregated to daily kWh before creating this CSV.",
        "limitations": ["One historical French household; performance may differ for other homes.",
                        "No weather or occupancy inputs; long forecasts accumulate error.",
                        "Bills use editable illustrative tariffs, not verified utility rates."]
    }
    (ROOT / "models/report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plot_comparison(report)
    print("Saved model and report. Selected:", winner, flush=True)

if __name__ == "__main__":
    main()
