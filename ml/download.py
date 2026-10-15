"""Download the public Kaggle source. Raw data stays local and out of Git."""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/uciml/electric-power-consumption-data-set"

def main():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    archive = folder / "household.zip"
    print("Downloading UCI Household Electric Power Consumption from Kaggle...")
    # Resume an interrupted download, and check that all expected bytes arrived.
    for attempt in range(5):
        size = archive.stat().st_size if archive.exists() else 0
        with urlopen(Request(URL, headers={"Range": f"bytes={size}-"}), timeout=60) as response:
            append = response.status == 206
            expected = int(response.headers.get("Content-Length", 0)) + (size if append else 0)
            with archive.open("ab" if append else "wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        if archive.stat().st_size == expected:
            break
        print("Interrupted; resuming download...")
    else:
        raise SystemExit("Download incomplete. Retry this script or download manually from Kaggle.")
    with ZipFile(archive) as source:
        name = next(n for n in source.namelist() if n.endswith("household_power_consumption.txt"))
        (folder / "household_power_consumption.txt").write_bytes(source.read(name))
    print("Ready: data/household_power_consumption.txt")

if __name__ == "__main__":
    main()
