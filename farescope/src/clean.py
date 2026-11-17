"""Clean Table 6, build route keys, repair city coordinates, deflate fares."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.download import load_cpi_quarterly, load_table6
from src.features import TRAIN_END_YEAR

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FARES_PATH = PROCESSED / "fares.parquet"
CITIES_PATH = PROCESSED / "cities.parquet"

CARRIER_MERGES = {"US": "AA", "VX": "AS"}  # US Airways -> American, Virgin America -> Alaska
WINSOR = (0.005, 0.995)
COORD_TOLERANCE_MILES = 60  # a city's coordinate is "bad" if its routes disagree by more than this


def is_covid(year, quarter):
    """2020 Q2 through 2021 Q4."""
    return ((year == 2020) & (quarter >= 2)) | (year == 2021)


def add_route_keys(df):
    """Direction-independent key: the smaller city market ID always goes first."""
    df = df.copy()
    swap = df["citymarketid_1"] > df["citymarketid_2"]
    df["city_id_a"] = np.where(swap, df["citymarketid_2"], df["citymarketid_1"])
    df["city_id_b"] = np.where(swap, df["citymarketid_1"], df["citymarketid_2"])
    df["route_id"] = df["city_id_a"].astype(str) + "-" + df["city_id_b"].astype(str)
    return df


def parse_point(s):
    """'POINT (-75.469759 40.602753)' -> (lon, lat) float columns. WKT order is lon lat."""
    xy = s.str.extract(r"^POINT \((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)$").astype("float64")
    return xy[0], xy[1]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def repair_coordinates(cities, routes):
    """Validate the file's coordinates against route distances and re-estimate bad ones.

    The source geocodes are missing from 2021 on and wrong for some cities
    (e.g. Dallas/Fort Worth is placed in Pennsylvania). `nsmiles` is reliable,
    so a city whose coordinate disagrees with its route distances is flagged
    and, like a city with no coordinate, re-located by fitting its position to
    the distances from cities with trusted coordinates.

    cities: DataFrame indexed by city id with lat, lon (NaN if missing)
    routes: DataFrame with city_id_a, city_id_b, miles
    Returns cities with lat, lon, coord_source in {"dot", "estimated", "none"}.
    """
    cities = cities.copy()
    cities["coord_source"] = np.where(cities["lat"].notna(), "dot", "none")
    r = routes.copy()

    # 1) Drop the worst-fitting city repeatedly until every remaining city agrees
    #    with its route distances (a bad city inflates its neighbours' errors,
    #    so remove one at a time).
    while True:
        ok = cities["coord_source"] == "dot"
        rr = r[r.city_id_a.map(ok) & r.city_id_b.map(ok)]
        err = np.abs(
            haversine_miles(
                rr.city_id_a.map(cities.lat), rr.city_id_a.map(cities.lon),
                rr.city_id_b.map(cities.lat), rr.city_id_b.map(cities.lon),
            ) - rr.miles
        )
        per_city = pd.concat([
            pd.Series(err.values, index=rr.city_id_a.values),
            pd.Series(err.values, index=rr.city_id_b.values),
        ]).groupby(level=0).median()
        if per_city.empty or per_city.max() <= COORD_TOLERANCE_MILES:
            break
        cities.loc[per_city.idxmax(), ["lat", "lon", "coord_source"]] = [np.nan, np.nan, "none"]

    # 2) Estimate every city without a trusted coordinate from >= 3 trusted neighbours:
    #    coarse grid search over the contiguous US (avoids mirror-image local minima),
    #    then least-squares refinement.
    glat, glon = np.meshgrid(np.arange(24.0, 50.0, 0.5), np.arange(-125.0, -66.0, 0.5))
    glat, glon = glat.ravel(), glon.ravel()
    progress = True
    while progress:
        progress = False
        trusted = cities["coord_source"] != "none"
        for cid in cities.index[~trusted]:
            nb = pd.concat([
                r.loc[r.city_id_a == cid, ["city_id_b", "miles"]].set_axis(["other", "miles"], axis=1),
                r.loc[r.city_id_b == cid, ["city_id_a", "miles"]].set_axis(["other", "miles"], axis=1),
            ])
            nb = nb[nb.other.map(trusted)]
            if len(nb) < 3:
                continue
            alat, alon, d = nb.other.map(cities.lat).values, nb.other.map(cities.lon).values, nb.miles.values
            grid_err = np.abs(haversine_miles(glat[:, None], glon[:, None], alat[None, :], alon[None, :]) - d).mean(axis=1)
            start = [glat[grid_err.argmin()], glon[grid_err.argmin()]]
            fit = least_squares(lambda p: haversine_miles(p[0], p[1], alat, alon) - d, start, loss="soft_l1", f_scale=25)
            if np.median(np.abs(fit.fun)) <= COORD_TOLERANCE_MILES:
                cities.loc[cid, ["lat", "lon", "coord_source"]] = [fit.x[0], fit.x[1], "estimated"]
                progress = True
    return cities


def build_cities(df):
    """One row per city market: canonical (most recent) name and repaired coordinates."""
    lon1, lat1 = parse_point(df["location_1"])
    lon2, lat2 = parse_point(df["location_2"])
    long = pd.concat([
        pd.DataFrame({"id": df.citymarketid_1, "name": df.city1, "lat": lat1, "lon": lon1, "t": df.year * 10 + df.quarter}),
        pd.DataFrame({"id": df.citymarketid_2, "name": df.city2, "lat": lat2, "lon": lon2, "t": df.year * 10 + df.quarter}),
    ])
    cities = long.sort_values("t").groupby("id").agg(name=("name", "last"))
    coords = long.dropna(subset=["lat"]).groupby("id")[["lat", "lon"]].agg(lambda s: s.mode().iloc[0])
    cities = cities.join(coords)
    assert cities["name"].is_unique, "two city market IDs share a name"

    routes = df.groupby(["city_id_a", "city_id_b"])["nsmiles"].median().rename("miles").reset_index()
    n_dot = cities["lat"].notna().sum()
    cities = repair_coordinates(cities, routes)
    counts = cities["coord_source"].value_counts().to_dict()
    print(f"[clean] city coordinates: {n_dot} of {len(cities)} present in the file; after validation -> {counts}")
    return cities


def main():
    df = load_table6()
    n0 = len(df)

    df = df[df["year"] >= 2012]
    df = df[~is_covid(df["year"], df["quarter"])]
    df = df.dropna(subset=["fare", "nsmiles", "passengers", "large_ms", "lf_ms"])
    df = df[(df["fare"] > 0) & (df["nsmiles"] > 0)]
    df = add_route_keys(df)

    # A few route-quarters appear twice; keep the record with more passengers.
    dup = df.duplicated(["year", "quarter", "route_id"], keep=False)
    if dup.any():
        print(f"[clean] {dup.sum()} rows share a route-quarter; keeping the larger record:")
        print(df.loc[dup, ["year", "quarter", "city1", "city2", "passengers", "fare"]].to_string(index=False))
        df = df.sort_values("passengers").drop_duplicates(["year", "quarter", "route_id"], keep="last")
    print(f"[clean] rows: {n0:,} raw -> {len(df):,} after filters")

    # Carriers
    for col in ["carrier_lg", "carrier_low"]:
        df[col] = df[col].replace(CARRIER_MERGES).astype(str)
    t = df["year"] * 4 + df["quarter"]
    seen = pd.concat([
        pd.DataFrame({"carrier": df.carrier_lg, "t": t}),
        pd.DataFrame({"carrier": df.carrier_low, "t": t}),
    ]).groupby("carrier")["t"].agg(rows="size", last="max")
    seen["last_seen"] = ((seen["last"] - 1) // 4).astype(str) + " Q" + ((seen["last"] - 1) % 4 + 1).astype(str)
    seen["looks_retired"] = seen["last"] < t.max() - 4  # absent for more than a year
    print("[clean] carrier counts (largest + lowest-fare columns combined, after US->AA, VX->AS):")
    print(seen[["rows", "last_seen", "looks_retired"]].sort_values("rows", ascending=False).to_string())
    print("[clean] codes that look retired (left unmapped):", sorted(seen.index[seen.looks_retired]))

    # Cities: canonical names + coordinates (map only, never model features)
    cities = build_cities(df)
    df["city_a"] = df["city_id_a"].map(cities["name"])
    df["city_b"] = df["city_id_b"].map(cities["name"])

    # Winsorize fare with cutoffs from training years only
    lo, hi = df.loc[df["year"] <= TRAIN_END_YEAR, "fare"].quantile(list(WINSOR))
    clipped = ((df["fare"] < lo) | (df["fare"] > hi)).sum()
    df["fare"] = df["fare"].clip(lo, hi)
    print(f"[clean] winsorized fare to [{lo:.2f}, {hi:.2f}] (train-year cutoffs); {clipped:,} rows clipped")

    # Inflation: express every fare in dollars of the latest quarter in the data
    cpi = load_cpi_quarterly()
    df = df.merge(cpi[["year", "quarter", "cpi"]], on=["year", "quarter"], how="left", validate="m:1")
    assert df["cpi"].notna().all(), "CPI missing for some quarters"
    ref_year, ref_q = df.sort_values(["year", "quarter"]).iloc[-1][["year", "quarter"]].astype(int)
    ref = cpi[(cpi.year == ref_year) & (cpi.quarter == ref_q)].iloc[0]
    if ref["n_months"] < 3:
        print(f"[clean] WARNING: CPI for {ref_year} Q{ref_q} covers only {ref['n_months']} month(s)")
    df["real_fare"] = df["fare"] * ref["cpi"] / df["cpi"]
    df["cpi_ref_quarter"] = f"{ref_year} Q{ref_q}"
    print(f"[clean] real_fare is in {ref_year} Q{ref_q} dollars (CPI {ref['cpi']:.2f})")

    # fare_lg / fare_low are dropped here so they can never reach the model (target leakage).
    keep = ["year", "quarter", "route_id", "city_id_a", "city_id_b", "city_a", "city_b",
            "nsmiles", "passengers", "large_ms", "lf_ms", "carrier_lg", "carrier_low",
            "fare", "real_fare", "cpi_ref_quarter"]
    df = df[keep].sort_values(["year", "quarter", "route_id"]).reset_index(drop=True)
    for c in ["route_id", "city_a", "city_b", "carrier_lg", "carrier_low", "cpi_ref_quarter"]:
        df[c] = df[c].astype(str)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FARES_PATH, index=False)
    cities.reset_index().to_parquet(CITIES_PATH, index=False)
    print(f"[clean] saved {len(df):,} rows, {df.route_id.nunique():,} routes, {len(cities)} cities")
    return df, cities


if __name__ == "__main__":
    main()
