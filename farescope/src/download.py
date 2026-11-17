"""Download DOT Consumer Airfare Report Table 6 (Socrata) and CPI-U (FRED)."""
import argparse
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TABLE6_PATH = RAW / "table6.csv"
CPI_PATH = RAW / "cpi.csv"

TABLE6_URL = "https://data.transportation.gov/resource/yj5y-b2ir.csv"
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
PAGE = 50000

EXPECTED_COLUMNS = [
    "year", "quarter", "citymarketid_1", "citymarketid_2", "city1", "city2",
    "nsmiles", "passengers", "fare", "carrier_lg", "large_ms", "fare_lg",
    "carrier_low", "lf_ms", "fare_low", "table_1_flag", "location_1",
    "location_2", "tbl6pk",
]
INT_COLS = ["year", "quarter", "citymarketid_1", "citymarketid_2", "table_1_flag"]
FLOAT_COLS = ["nsmiles", "passengers", "fare", "large_ms", "fare_lg", "lf_ms", "fare_low"]
STR_COLS = ["city1", "city2", "carrier_lg", "carrier_low", "location_1", "location_2", "tbl6pk"]


def download_table6(refresh=False):
    if TABLE6_PATH.exists() and not refresh:
        print(f"[download] {TABLE6_PATH.name} exists, skipping (use --refresh to re-download)")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    pages, offset = [], 0
    while True:
        r = requests.get(
            TABLE6_URL,
            params={"$where": "year>=2012", "$order": "tbl6pk", "$limit": PAGE, "$offset": offset},
            timeout=300,
        )
        r.raise_for_status()
        page = pd.read_csv(io.StringIO(r.text), dtype=str)
        if page.empty:
            break
        pages.append(page)
        offset += PAGE
        print(f"[download] table6: {offset - PAGE + len(page):,} rows")
    pd.concat(pages, ignore_index=True).to_csv(TABLE6_PATH, index=False)


def download_cpi(refresh=False):
    if CPI_PATH.exists() and not refresh:
        print(f"[download] {CPI_PATH.name} exists, skipping")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(CPI_URL, timeout=120)
    r.raise_for_status()
    CPI_PATH.write_text(r.text)


def load_table6():
    """Read the raw CSV as strings, then cast every column explicitly."""
    df = pd.read_csv(TABLE6_PATH, dtype=str)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    assert not missing, f"table6.csv is missing expected columns: {missing}"
    # location_*_address/state/zip are empty and location_*_city duplicates city1/city2
    df = df[EXPECTED_COLUMNS].copy()
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in STR_COLS:
        df[c] = df[c].astype("string").str.strip()
    return df


def load_cpi_quarterly():
    """Monthly CPI-U -> quarterly average. Returns DataFrame[year, quarter, cpi]."""
    cpi = pd.read_csv(CPI_PATH)
    # Header names have changed over time on FRED (DATE vs observation_date), so detect them.
    date_col = next(c for c in cpi.columns if "date" in c.lower())
    value_col = next(c for c in cpi.columns if c != date_col)
    cpi[date_col] = pd.to_datetime(cpi[date_col])
    cpi[value_col] = pd.to_numeric(cpi[value_col], errors="coerce")  # FRED uses "." for missing
    cpi = cpi.dropna(subset=[value_col])
    out = (
        cpi.assign(year=cpi[date_col].dt.year, quarter=cpi[date_col].dt.quarter)
        .groupby(["year", "quarter"])[value_col]
        .agg(cpi="mean", n_months="count")
        .reset_index()
    )
    return out


def inspect(df):
    print("\n[inspect] columns:", list(df.columns))
    print("[inspect] dtypes:\n", df.dtypes.to_string())
    print(f"[inspect] rows: {len(df):,}")
    yq = df[["year", "quarter"]].drop_duplicates().sort_values(["year", "quarter"])
    first, last = yq.iloc[0], yq.iloc[-1]
    print(f"[inspect] range: {first.year} Q{first.quarter} -> {last.year} Q{last.quarter}")
    print("[inspect] nulls:\n", df.isna().sum().to_string())


def main(refresh=False):
    download_table6(refresh)
    download_cpi(refresh)
    df = load_table6()
    inspect(df)
    cpi = load_cpi_quarterly()
    print("\n[inspect] CPI quarterly tail:\n", cpi.tail(4).to_string(index=False))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="re-download even if files exist")
    main(p.parse_args().refresh)
