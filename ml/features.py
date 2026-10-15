"""Shared, past-only features for training and recursive forecasting."""
import numpy as np
import pandas as pd

FEATURES = ["lag_1", "lag_7", "lag_14", "mean_7", "mean_28", "std_7",
            "weekday_sin", "weekday_cos", "month_sin", "month_cos"]

def feature_row(history, date):
    values = np.asarray(history, dtype=float)
    if len(values) < 28:
        raise ValueError("At least 28 consecutive daily readings are required.")
    return [values[-1], values[-7], values[-14], values[-7:].mean(),
            values[-28:].mean(), values[-7:].std(),
            np.sin(2 * np.pi * date.dayofweek / 7), np.cos(2 * np.pi * date.dayofweek / 7),
            np.sin(2 * np.pi * date.month / 12), np.cos(2 * np.pi * date.month / 12)]

def supervised(daily):
    rows = [feature_row(daily.kwh.iloc[:i], daily.index[i]) for i in range(28, len(daily))]
    return pd.DataFrame(rows, columns=FEATURES, index=daily.index[28:]), daily.kwh.iloc[28:]

def forecast(model, history, last_date, days):
    values = list(history)
    predictions = []
    for offset in range(1, days + 1):
        date = pd.Timestamp(last_date) + pd.Timedelta(days=offset)
        row = pd.DataFrame([feature_row(values, date)], columns=FEATURES)
        kwh = max(0.0, float(model.predict(row)[0]))
        values.append(kwh)
        predictions.append({"date": date.strftime("%Y-%m-%d"), "kwh": round(kwh, 3)})
    return predictions
