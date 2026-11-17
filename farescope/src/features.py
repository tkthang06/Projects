"""Model 2 feature definitions, the time-based split, and category handling."""
import numpy as np
import pandas as pd

# Time-based split (never random). COVID quarters are removed in clean.py.
TRAIN_END_YEAR = 2022   # train: 2012..2022
VAL_YEAR = 2023         # validation: 2023
TEST_START_YEAR = 2024  # test: 2024 .. latest quarter

TARGET = "real_fare"
NUMERIC = ["nsmiles", "log_passengers", "large_ms", "lf_ms", "low_is_largest", "prior_rel"]
CATEGORICAL = ["carrier_lg", "carrier_low", "quarter", "city_a", "city_b", "cluster"]
FEATURES = NUMERIC + CATEGORICAL

# Never model inputs: the first two are components of the target (leakage).
EXCLUDED = ["fare_lg", "fare_low", "fare", "real_fare", "year", "table_1_flag",
            "route_id", "city_id_a", "city_id_b", "lat_a", "lon_a", "lat_b", "lon_b"]


def split_masks(df):
    train = df["year"] <= TRAIN_END_YEAR
    val = df["year"] == VAL_YEAR
    test = df["year"] >= TEST_START_YEAR
    return train, val, test


def add_derived(df):
    """Add the engineered columns. Needs passengers, carrier_lg, carrier_low."""
    df = df.copy()
    df["log_passengers"] = np.log1p(df["passengers"])
    df["low_is_largest"] = (df["carrier_low"] == df["carrier_lg"]).astype("int8")
    return df


def add_time_context(df):
    """Handle the downward drift in real fares without using `year` as a feature.

    The model predicts a route's log fare RELATIVE to the national level of its
    quarter, so it never has to know the absolute price level. Adds:
      log_fare    log(real_fare)
      level       mean log_fare of all routes in the row's quarter (training target = log_fare - level)
      level_prev  level of the same quarter one year earlier
      level_forecast  level_prev + drift: the forecast of this quarter's level made a
                  year ahead, used to turn a prediction back into dollars. Drift is the
                  average year-over-year change in the level over every pair of
                  consecutive years up to the previous year ("seasonal naive with
                  drift"); it uses nothing from the row's own year.
      prior_rel   feature: this route's log_fare minus the national level, same
                  quarter one year earlier (NaN when the route has no such row)
    Returns (df, levels): levels has one row per (year, quarter) with `level` and
    `next_level`, the forecast for the same quarter of the following year.
    """
    keys = ["year", "quarter"]
    df = df.copy()
    df["log_fare"] = np.log(df[TARGET])
    levels = df.groupby(keys)["log_fare"].mean().rename("level").reset_index()
    before = levels.assign(year=levels["year"] + 1).rename(columns={"level": "level_before"})
    levels = levels.merge(before, on=keys, how="left")
    change = (levels["level"] - levels["level_before"]).groupby(levels["year"]).agg(["sum", "count"])
    change = change.sort_index().cumsum()  # all year-over-year changes observed through each year
    drift = (change["sum"] / change["count"]).where(change["count"] > 0, 0.0)
    levels["next_level"] = levels["level"] + levels["year"].map(drift)
    levels = levels.drop(columns="level_before")

    df = df.merge(levels[keys + ["level"]], on=keys, how="left")
    ahead = levels.assign(year=levels["year"] + 1).rename(
        columns={"level": "level_prev", "next_level": "level_forecast"})
    df = df.merge(ahead, on=keys, how="left")
    prior = df[["route_id", "quarter"]].assign(year=df["year"] + 1, prior_rel=df["log_fare"] - df["level"])
    df = df.merge(prior, on=["route_id", "year", "quarter"], how="left", validate="1:1")
    return df, levels


def fit_category_levels(df):
    """Category levels seen in `df` (pass training rows only)."""
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}


def make_X(df, levels, features=FEATURES):
    """Build the model matrix with fixed category levels.

    Values not in `levels` (e.g. a carrier that first appears after training)
    become NaN, which XGBoost treats as missing instead of crashing.
    """
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in CATEGORICAL:
            codes = pd.Index(levels[c]).get_indexer(df[c])  # -1 for unseen values
            X[c] = pd.Categorical.from_codes(codes, categories=levels[c])
        else:
            X[c] = df[c].astype("float64")
    return X
