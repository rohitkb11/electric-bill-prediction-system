"""JSON stdin/stdout bridge called by Express; never accept a user-supplied model."""
import json
import sys
from pathlib import Path
import joblib
from features import forecast

ROOT = Path(__file__).resolve().parents[1]

def main():
    request = json.load(sys.stdin)
    model = joblib.load(ROOT / "models/consumption.joblib")
    result = forecast(model, request["history"], request["last_date"], request["days"])
    print(json.dumps({"predictions": result}))

if __name__ == "__main__":
    main()
