"""Compare household-profile regressors without changing the current app model.

Usage: python ml/compare_households.py path/to/household_electricity.csv
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Household_Size", "Rooms", "AC_Units", "Fan_Count", "Light_Count",
           "Refrigerator", "TV_Count", "Washing_Machine", "Other_Appliances"]
CATEGORICAL = ["Location", "Area_Type"]
TARGET = "Monthly_Usage_kWh"
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.csv).drop_duplicates()
    X, y = data[NUMERIC + CATEGORICAL], data[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=42)
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                      OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CATEGORICAL)
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, regressor in [("Linear Regression", LinearRegression()),
                            ("Random Forest", RandomForestRegressor(n_estimators=200,
                             bootstrap=True, random_state=42, n_jobs=1))]:
        print("Training:", name, flush=True)
        pipeline = Pipeline([("preprocess", preprocessing), ("model", regressor)])
        before = -cross_val_score(pipeline, Xtrain, ytrain, cv=cv,
                                 scoring="neg_mean_absolute_error").mean()
        params = {}
        if name == "Random Forest":
            search = GridSearchCV(pipeline, {"model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 3], "model__max_features": [1.0, "sqrt"]},
                cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
            search.fit(Xtrain, ytrain)
            pipeline, params = search.best_estimator_, search.best_params_
            after = -search.best_score_
        else:
            pipeline.fit(Xtrain, ytrain)
            after = before
        prediction = pipeline.predict(Xtest)
        mse = float(mean_squared_error(ytest, prediction))
        results.append({"name": name, "cv_mae_before": float(before), "cv_mae_after": float(after),
            "parameters": params, "test": {"mae": float(mean_absolute_error(ytest, prediction)),
            "mse": mse, "rmse": float(np.sqrt(mse)), "r2": float(r2_score(ytest, prediction))}})
    report = {"source": "https://www.kaggle.com/datasets/armanahmedtaief/electricity-bill-prediction",
        "target": TARGET, "features": NUMERIC + CATEGORICAL, "rows": len(data),
        "training_rows": len(Xtrain), "test_rows": len(Xtest), "random_state": 42,
        "evaluation": "80/20 household split; 5-fold CV and tuning on training only. Preprocessing is fitted inside each CV fold. Same untouched test households for both models.",
        "excluded": [column for column in data.columns if column not in NUMERIC + CATEGORICAL + [TARGET]],
        "selected_by_training_cv": min(results, key=lambda row: row["cv_mae_after"])["name"],
        "models": results}
    folder = ROOT / "output/household-comparison"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ["mse", "r2"], ["Test MSE (kWh²) · lower is better", "Test R² · higher is better"]):
        values = [row["test"][metric] for row in results]
        bars = ax.bar([r["name"] for r in results], values, color=["#8aab8d", "#21614f"], width=.5)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(min(0, min(values)), max(values) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Household-profile prediction · 800 train / 200 test households")
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
