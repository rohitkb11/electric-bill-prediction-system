# Dataset provenance

Source: [Kaggle: Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set), originally the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

The source measures one household in Sceaux, France, from December 2006 to November 2010. It is not an Indian household dataset and does not contain electricity bills or tariff labels. The UI uses editable illustrative rupee tariffs to demonstrate bill estimation.

Citation: Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included `daily.csv` adapts the source through aggregation, coverage normalization and past-only missing-day filling; no endorsement by the dataset creators is implied.

## One CSV dataset

Both training and the dashboard read **`data/daily.csv`** directly. It contains 1,440 rows and three columns: `date` (YYYY-MM-DD), `kwh` (daily consumption), and `imputed` (True for a filled value). The original minute-level file and ZIP archive are not needed.

Before this CSV was created, minute-level kW readings were summed and divided by 60 to obtain daily kWh. Days with at least 90% valid readings were normalized for coverage; the other 22 days were filled using the preceding seven-day median and flagged. This preparation can introduce error. Filled days provide history but are excluded as fitting/evaluation targets.

Run `python ml/train.py` in your Python environment to train directly from the CSV. The script sorts dates, removes duplicate dates and handles any additional invalid/missing readings. It does not create another dataset or overwrite the CSV.
