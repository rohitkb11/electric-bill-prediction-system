"""Read household features from Express and return monthly kWh."""
import json
import sys
import joblib
import pandas as pd
from preprocess import ROOT, FEATURES


def main():
    request = json.load(sys.stdin)
    row = pd.DataFrame([{name: request[name] for name in FEATURES}])
    model = joblib.load(ROOT / "models/consumption.joblib")
    monthly_kwh = max(0.0, float(model.predict(row)[0]))
    print(json.dumps({"monthlyKwh": round(monthly_kwh, 2)}))


if __name__ == "__main__":
    main()
