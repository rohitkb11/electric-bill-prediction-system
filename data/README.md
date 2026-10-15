# Dataset collection and provenance

The project uses one file: **household_electricity.csv**.

- Source: https://www.kaggle.com/datasets/armanahmedtaief/electricity-bill-prediction
- Publisher: Arman Ahmed Taief.
- Original filename: Electricity Bill Prediction.csv (renamed locally; original columns retained).
- Size: 1,000 household records and 19 columns.
- Listed license: CC0 / Public Domain.
- Geography: Bangladesh. Recorded bill amounts are BDT.
- Collection method: not documented by the publisher. Do not describe it as verified real survey data.

## Features used

Location, Area_Type, Household_Size, Rooms, AC_Units, Fan_Count, Light_Count, Refrigerator, TV_Count, Washing_Machine and Other_Appliances.

Refrigerator and Washing_Machine are 0/1 presence flags, not counts. Other appliance fields are counts. Target: Monthly_Usage_kWh.

## Columns excluded from inputs

Customer_ID, Monthly_Usage_kWh (the target), Previous_Month_Usage_kWh, Temperature_C, Meter_Type, Tariff_Category, Previous_Bill_BDT and Electricity_Bill_BDT.

The selected model uses only the household details entered in the form. Previous usage and bills are unnecessary for that workflow. Temperature/meter/tariff fields are also omitted to keep the inputs simple. Bill amounts are calculated from predicted kWh and a user-entered rate, not learned from bill labels.

The CSV is included so training requires no download or ZIP extraction. To collect it again, download the named CSV from the source and save it with the local filename above.
