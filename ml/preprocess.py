"""Load the single CSV and share preprocessing between training and prediction."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/household_electricity.csv"
TARGET = "Monthly_Usage_kWh"
NUMERIC = ["Household_Size", "Rooms", "AC_Units", "Fan_Count", "Light_Count",
           "Refrigerator", "TV_Count", "Washing_Machine", "Other_Appliances"]
CATEGORICAL = ["Location", "Area_Type"]
FEATURES = NUMERIC + CATEGORICAL
LABELS = ["Family members", "Rooms", "Air conditioners", "Fans", "Lights",
          "Refrigerator (0 = no, 1 = yes)", "Televisions",
          "Washing machine (0 = no, 1 = yes)", "Other appliances"]
LIMITS = [(1, 20), (1, 30), (0, 20), (0, 100), (0, 200), (0, 1), (0, 30), (0, 1), (0, 100)]

def load_dataset():
    data = pd.read_csv(DATASET)
    original_rows = len(data)
    data = data.drop_duplicates().copy()
    duplicates = original_rows - len(data)
    for column in NUMERIC + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[data[column] < 0, column] = float("nan")
    for column in CATEGORICAL:
        data[column] = data[column].str.strip().replace("", float("nan"))
    missing = data[FEATURES].isna().sum().to_dict()
    invalid_targets = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    return data, {"source_rows": original_rows, "usable_rows": len(data),
        "duplicates_removed": duplicates, "invalid_targets_removed": invalid_targets,
        "missing_features": missing, "mean_monthly_kwh": round(float(data[TARGET].mean()), 2)}

def make_model():
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)), CATEGORICAL)
    ])
    return Pipeline([("preprocess", preprocessing), ("model", LinearRegression())])
