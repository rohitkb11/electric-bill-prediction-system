# Dataset provenance

Source: [Kaggle: Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set), originally the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

The source measures one household in Sceaux, France, from December 2006 to November 2010. It is not an Indian household dataset and does not contain electricity bills or tariff labels. The UI uses editable illustrative rupee tariffs to demonstrate bill estimation.

Run `npm run download:data` to obtain the raw file from Kaggle. It is ignored by Git because it is large. `daily.csv` is the processed daily kWh series produced by training; missing days and low-coverage days are handled as explained in the training report. Credit: Georges Hebrail and Alice Berard, EDF R&D. See the linked source for licensing and attribution.
