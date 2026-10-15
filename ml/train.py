"""Train Linear Regression and cover all seven PR12 mini-project stages."""
import json
import joblib
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from preprocess import ROOT, DATASET, TARGET, NUMERIC, CATEGORICAL, FEATURES, LABELS, LIMITS, load_dataset, make_model
from plot_metrics import plot_comparison


def evaluate(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {"mae": round(float(mean_absolute_error(actual, predicted)), 4),
            "mse": round(mse, 4), "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2_score(actual, predicted)), 4)}


def main():
    # 1. Problem: estimate monthly household kWh and calculate its cost.
    # 2. Collection: one included CSV from the attributed Kaggle source.
    # 3. Preprocessing: split before fitting imputers, scalers and one-hot encoding.
    data, statistics = load_dataset()
    X, y = data[FEATURES], data[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Development: ordinary least-squares Linear Regression baseline.
    base = make_model()
    base_cv = float(-cross_val_score(base, Xtrain, ytrain, cv=cv,
                                    scoring="neg_mean_absolute_error").mean())
    base.fit(Xtrain, ytrain)

    # 5/6. Optimize on training folds only; test data never chooses parameters.
    # Try an intercept and nonnegative coefficients: Linear Regression has few parameters.
    search = GridSearchCV(make_model(), {"model__fit_intercept": [True, False],
                         "model__positive": [False, True]}, cv=cv,
                         scoring="neg_mean_absolute_error", n_jobs=1)
    search.fit(Xtrain, ytrain)
    tuned_cv = float(-search.best_score_)
    use_tuned = tuned_cv < base_cv - 1e-9
    selected = search.best_estimator_ if use_tuned else base
    threshold = float(ytrain.median())
    results, predictions = [], {}
    for name, model, cv_mae in [("Base Linear Regression", base, base_cv),
                               ("Tuned Linear Regression", search.best_estimator_, tuned_cv)]:
        predicted = np.maximum(0, model.predict(Xtest))
        predictions[name] = predicted
        actual_high, predicted_high = ytest >= threshold, predicted >= threshold
        results.append({"name": name, "cv_mae": round(cv_mae, 4),
            "parameters": {"fit_intercept": model.named_steps["model"].fit_intercept,
                           "positive": model.named_steps["model"].positive},
            "test": evaluate(ytest, predicted),
            "high_usage_test": {
                "accuracy": round(float(accuracy_score(actual_high, predicted_high)), 4),
                "precision": round(float(precision_score(actual_high, predicted_high, zero_division=0)), 4),
                "recall": round(float(recall_score(actual_high, predicted_high, zero_division=0)), 4)}})

    # 7. Save the evaluated training-only model, report and plots.
    # The exported model keeps test households held out; no refitting on all rows.
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    joblib.dump(selected, folder / "consumption.joblib", compress=3)
    schema = [{"name": column, "label": label, "type": "number", "min": limits[0],
               "max": limits[1]} for column, label, limits in zip(NUMERIC, LABELS, LIMITS)]
    for column, label in zip(CATEGORICAL, ["Location", "Area type"]):
        schema.append({"name": column, "label": label, "type": "select",
                       "options": sorted(Xtrain[column].dropna().unique().tolist())})
    sample = {column: int(Xtrain.iloc[0][column]) if column in NUMERIC else str(Xtrain.iloc[0][column])
              for column in FEATURES}
    selected_name = "Tuned Linear Regression" if use_tuned else "Base Linear Regression"
    report = {
        "problem_definition": "Estimate monthly household electricity consumption from location, family size, rooms and appliances. Calculate a bill using a user-entered tariff.",
        "dataset": DATASET.name,
        "source_url": "https://www.kaggle.com/datasets/armanahmedtaief/electricity-bill-prediction",
        "statistics": statistics, "target": TARGET, "features": FEATURES,
        "excluded_columns": [column for column in data.columns if column not in FEATURES + [TARGET]],
        "split": {"training_rows": len(Xtrain), "test_rows": len(Xtest), "random_state": 42,
                  "cross_validation_folds": 5},
        "selected_model": "Linear Regression", "selected_variant": selected_name,
        "optimization": {"method": "GridSearchCV on training data only",
            "base_cv_mae": round(base_cv, 4), "best_cv_mae": round(tuned_cv, 4),
            "best_parameters": search.best_params_, "improved": use_tuned,
            "candidates": [{"parameters": params, "cv_mae": round(float(-score), 4)}
                           for params, score in zip(search.cv_results_["params"],
                                                    search.cv_results_["mean_test_score"])],
            "conclusion": "Tuned settings reduced cross-validation MAE." if use_tuned else
                          "Tuning did not reduce cross-validation MAE. Retained baseline settings."},
        "models": results,
        "high_usage_classification": {"threshold_kwh": threshold,
            "definition": "High usage = monthly kWh >= training median. Secondary classification from regression predictions, not regression accuracy."},
        "input_schema": schema, "sample_input": sample,
        "preprocessing_note": "Remove duplicates; coerce invalid numeric values to missing. Impute numeric features with training medians, standardize them, and one-hot encode location/area type. All fitted preprocessing stays inside the cross-validation pipeline.",
        "limitations": ["Bangladesh locations; collection method is undocumented by the publisher.",
            "This estimates monthly consumption for a household profile; it is not a dated time-series forecast.",
            "The bill uses an illustrative flat BDT tariff, not verified provider slabs/taxes."]}
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    selected_predictions = predictions[selected_name]
    plot_comparison(report, ytest.to_numpy(), selected_predictions)
    examples = [{"actual_kwh": round(float(actual), 3), "predicted_kwh": round(float(predicted), 3)}
                for actual, predicted in zip(ytest, selected_predictions)]
    (folder / "test_predictions.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    summary = ["# Linear Regression training results", "",
        f"Dataset: {DATASET.name}; {len(data)} households. Split: {len(Xtrain)} training / {len(Xtest)} test (seed 42).",
        "", "| Variant | Training CV MAE | Test MAE | Test MSE | Test RMSE | Test R2 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row in results:
        score = row["test"]
        summary.append(f"| {row['name']} | {row['cv_mae']:.4f} | {score['mae']:.4f} | {score['mse']:.4f} | {score['rmse']:.4f} | {score['r2']:.4f} |")
    summary += ["", report["optimization"]["conclusion"],
        "", "Best parameters: " + json.dumps(search.best_params_),
        "", "MAE/RMSE are monthly kWh; MSE is kWh squared. R2 is variance explained, not classification accuracy.",
        "", "Tuning uses five-fold training-only CV. Test households are never used to fit or select parameters.",
        "", "The saved model is fitted on 800 training households, matching the reported test evaluation.",
        "", "See models/report.json for all four tuning candidates, high-usage scores and dataset cleaning counts.",
        "", "The collection method is undocumented by the publisher; results describe this dataset and split, not validated performance across real homes."]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_name, "optimization": report["optimization"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
