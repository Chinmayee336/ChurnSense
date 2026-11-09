from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def main():
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 100_000:
        print(f"Dataset already exists: {DATA_PATH}")
        return
    print("Downloading the IBM Telco Customer Churn sample dataset...")
    urlretrieve(URL, DATA_PATH)
    print(f"Downloaded successfully: {DATA_PATH} ({DATA_PATH.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
