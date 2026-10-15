"""Download the public Kaggle source. Raw data stays local and out of Git."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/uciml/electric-power-consumption-data-set"

def main():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    archive = folder / "household.zip"
    print("Downloading UCI Household Electric Power Consumption from Kaggle...")
    urlretrieve(URL, archive)
    with ZipFile(archive) as source:
        name = next(n for n in source.namelist() if n.endswith("household_power_consumption.txt"))
        (folder / "household_power_consumption.txt").write_bytes(source.read(name))
    print("Ready: data/household_power_consumption.txt")

if __name__ == "__main__":
    main()
