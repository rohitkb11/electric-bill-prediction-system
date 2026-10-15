# Linear Regression training results

Dataset: household_electricity.csv; 1000 households. Split: 800 training / 200 test (seed 42).

| Variant | Training CV MAE | Test MAE | Test MSE | Test RMSE | Test R2 |
| --- | --- | --- | --- | --- | --- |
| Base Linear Regression | 21.7772 | 21.5988 | 724.3373 | 26.9135 | 0.9402 |
| Tuned Linear Regression | 21.7772 | 21.5988 | 724.3373 | 26.9135 | 0.9402 |

Tuning did not reduce cross-validation MAE. Retained baseline settings.

Best parameters: {"model__fit_intercept": true, "model__positive": false}

MAE/RMSE are monthly kWh; MSE is kWh squared. R2 is variance explained, not classification accuracy.

Tuning uses five-fold training-only CV. Test households are never used to fit or select parameters.

The saved model is fitted on 800 training households, matching the reported test evaluation.

See models/report.json for all four tuning candidates, high-usage scores and dataset cleaning counts.

The collection method is undocumented by the publisher; results describe this dataset and split, not validated performance across real homes.
