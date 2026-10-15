# Electric Bill Prediction System

A simple AI/ML mini-project that estimates **monthly electricity consumption** from location, family size, rooms and appliances, then calculates a bill. The model is **Linear Regression**, with an Express backend and plain HTML/CSS/JavaScript frontend.

## Run

Use Node.js 20+ and Python 3.12 or 3.13. One CSV, the trained model and evaluation graphs are included.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm install
npm start
```

Open http://localhost:3000. Use `npm.cmd` if PowerShell blocks `npm.ps1`.
On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.
Express detects `.venv` automatically; `PYTHON_PATH` and `PORT` are optional environment variables.

## Train again

```powershell
.\.venv\Scripts\python.exe ml/train.py
npm test
```

Training saves the complete preprocessing/model pipeline in `models/consumption.joblib`, evaluation and tuning details in `models/report.json`, test prediction pairs in `models/test_predictions.json`, three Matplotlib graphs in `public/plots/`, and an updated result summary in `docs/RESULTS.md`.

Regenerate graphs without retraining:

```powershell
.\.venv\Scripts\python.exe ml/plot_metrics.py
```

## Dataset and inputs

The only dataset is **`data/household_electricity.csv`**, containing 1,000 records downloaded from [Kaggle: Electricity Bill Prediction](https://www.kaggle.com/datasets/armanahmedtaief/electricity-bill-prediction) by Arman Ahmed Taief. The listed license is CC0. Locations are in Bangladesh; the publisher does not document whether records are measured or synthetic.

Inputs: location, area type, family members, rooms, ACs, fans, lights, refrigerator presence, TV count, washing machine presence and other appliance count. The target is **`Monthly_Usage_kWh`**.

Customer ID, current/previous bill amounts, previous usage and other unused fields are excluded. The model cannot see the consumption target as an input. Bills are calculated using an editable flat **BDT/kWh** rate and monthly fixed charge. Dataset bill labels are not used for training.

## PR12 requirements

| Requirement | Implementation |
| --- | --- |
| Problem definition | Estimate household monthly kWh and its cost from household details |
| Dataset collection | One included, attributed Kaggle CSV |
| Data preprocessing | Duplicate removal, numeric cleaning, training-fitted imputation, scaling and one-hot encoding |
| Model development | Ordinary least-squares Linear Regression baseline |
| Performance evaluation | 800/200 household split; MAE, MSE, RMSE and R2 |
| Model optimization | Five-fold GridSearchCV for intercept and nonnegative-coefficient settings |
| Analysis and documentation | Base/tuned metrics, actual-versus-predicted plot, tuning conclusion and project write-up |

All preprocessing is inside the model pipeline and is refitted inside each CV fold. Test data never chooses parameters. The saved model remains fitted on the 800 training households, so its evaluation corresponds to the reported test results. Tuning retains the baseline if no CV improvement occurs.

Accuracy, precision and recall are **secondary high-usage classification scores**, derived by comparing actual/predicted monthly kWh with the median training consumption. They are not regression accuracy; R2 is not a percentage of correct predictions.

## Files

- `ml/preprocess.py`: dataset cleaning, input columns and preprocessing pipeline.
- `ml/train.py`: baseline training, optimization, evaluation and export.
- `ml/predict.py`: saved-pipeline prediction for Express.
- `ml/plot_metrics.py`: Matplotlib evaluation graphs.
- `server/`: Express API and simple bill formula.
- `public/`: household form, results and graphs.
- `docs/PROJECT.md`: all seven syllabus stages explained.
- `docs/RESULTS.md`: actual results, updated by training.

`GET /api/data` returns model results, input schema and a sample household. `POST /api/predict` accepts `{ household: {...}, rate: 8, fixedCharge: 100 }` and returns monthly kWh and a bill breakdown. Required household fields are listed in the returned schema. The single-user demo handles one prediction at a time.

## Limitations

This is a monthly estimate for a household profile, not a dated usage forecast. Results describe this dataset and split; real households may differ. Inputs outside the training distribution can be unreliable. The flat tariff is illustrative and omits provider-specific slabs, taxes and subsidies.
