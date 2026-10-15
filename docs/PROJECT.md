# AI/ML Mini Project: Household Electricity and Bill Prediction

## 1. Problem Definition

Estimate a household's monthly electricity consumption from details a user can provide: location, area type, family size, rooms and appliances. Calculate an estimated monthly bill from predicted kWh and a user-entered tariff. This is supervised regression, not time-series forecasting or direct prediction of the dataset's bill column.

## 2. Dataset Collection

Use the single included `data/household_electricity.csv`, downloaded from Kaggle's Electricity Bill Prediction dataset by Arman Ahmed Taief. It contains 1,000 Bangladesh household records and 19 columns. The listed license is CC0. The original file was renamed without altering its column structure. The publisher does not document the data collection/generation method.

Use 11 input features: Location, Area_Type, Household_Size, Rooms, AC_Units, Fan_Count, Light_Count, Refrigerator, TV_Count, Washing_Machine and Other_Appliances. The target is Monthly_Usage_kWh. Refrigerator and washing machine are presence flags; the other appliance values are counts.

Do not use Customer_ID, bill amounts or the current consumption target as inputs. Previous usage, temperature, meter type and tariff category are also omitted because the intended form uses household profile details only. This avoids requiring unavailable history and keeps the workflow simple.

## 3. Data Preprocessing

`ml/preprocess.py` reads the CSV, removes exact duplicates, coerces invalid numeric values to missing and trims categorical strings. Rows with invalid targets are removed; features with missing values are retained for imputation.

Split into 800 training and 200 test households with random seed 42. There is no date-based forecasting task here. Inside each model pipeline, numeric missing values are filled with training medians and standardized. Categorical missing values use the training mode, then location/area are one-hot encoded with one reference category dropped. The complete pipeline is refitted inside every training CV fold. Test rows never fit preprocessing or choose hyperparameters.

The supplied dataset currently has no missing feature values, invalid targets or exact duplicate records; the cleaning steps remain for reproducibility if those issues arise.

## 4. Model Development

Use ordinary least-squares Linear Regression. It estimates monthly kWh as an intercept plus a weighted sum of the transformed inputs. This model was chosen after an earlier household-data comparison showed lower CV and test error than Random Forest. The active pipeline now contains only Linear Regression.

The saved `models/consumption.joblib` includes both preprocessing and the regressor. `ml/predict.py` loads it and accepts the same feature names used during training. Express validates the form inputs and calls this script. Negative extrapolated consumption is clipped to zero consistently during evaluation and prediction.

## 5. Performance Evaluation

Evaluate monthly kWh on the same 200 held-out test households for base and tuned variants:

- MAE: average absolute error in kWh.
- MSE: mean squared error in kWh squared.
- RMSE: square root of MSE, in kWh; penalizes larger errors.
- R2: variation explained compared with a constant predictor; not a percentage of correct predictions.

Matplotlib generates base/tuned MSE and R2 graphs, an actual-versus-predicted scatter plot, and a separate high-usage classification graph. The high-usage threshold is the median observed training consumption, applied to actual and predicted monthly values. Accuracy, precision and recall describe those derived high/normal labels; they are not regression accuracy. Undefined precision/recall is reported as zero.

## 6. Model Optimization

Use GridSearchCV with five shuffled training folds and MAE scoring. Evaluate all four combinations of `fit_intercept` (True/False) and `positive` (True/False). The latter optionally constrains fitted coefficients to be nonnegative; it is not assumed to help. Preprocessing is inside the searched pipeline.

Keep the baseline unless the best training CV MAE is strictly lower. For the supplied data, the search chooses the original settings: `fit_intercept=True`, `positive=False`. CV MAE remains 21.7772 kWh. This is a completed optimization experiment with no observed improvement, not an artificial performance gain.

The saved model stays fitted on the 800 training households, matching its held-out evaluation. Training does not refit the selected model on the test set.

## 7. Result Analysis and Documentation

See `docs/RESULTS.md` for actual baseline/tuned results, automatically updated by training. Full results and all tuning candidates are in `models/report.json`. The 200 actual/predicted pairs are in `models/test_predictions.json`. Graphs are in `public/plots/` and are displayed on the dashboard.

The current result is MAE 21.5988 kWh, MSE 724.3373, RMSE 26.9135 kWh and R2 0.9402. Baseline and tuned metrics are equal because tuning selected the same settings. Results apply to this dataset and split, not independently validated household billing accuracy.

The bill formula is `predicted monthly kWh * rate + monthly fixed charge`, in BDT. It is separate from the consumption model and does not implement provider-specific slabs, taxes or subsidies. Bangladesh locations and the undocumented collection method limit generalization. More representative measured data would be needed for real deployment.

## Reproduce

```powershell
.\.venv\Scripts\python.exe ml/train.py
npm.cmd test
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model.py
npm.cmd start
```
