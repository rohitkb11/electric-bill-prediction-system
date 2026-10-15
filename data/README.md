# Dataset provenance

Source: [Kaggle: Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set), originally the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

The source measures one household in Sceaux, France, from December 2006 to November 2010. It is not an Indian household dataset and does not contain electricity bills or tariff labels. The UI uses editable illustrative rupee tariffs to demonstrate bill estimation.

Citation: Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included `daily.csv` adapts the source through aggregation, coverage normalization and past-only missing-day filling; no endorsement by the dataset creators is implied.

Run `python ml/download.py` from your Python environment to obtain the raw file from Kaggle. It is ignored by Git because it is large. `daily.csv` is the processed daily kWh series produced by training; missing days and low-coverage days are handled as explained in the training report. Credit: Georges Hebrail and Alice Berard, EDF R&D. See the linked source for licensing and attribution.
