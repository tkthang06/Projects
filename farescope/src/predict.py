"""Load saved artifacts and predict a fare with per-feature contributions."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import FEATURES, add_derived, make_X

MODELS = Path(__file__).resolve().parents[1] / "models"
ARTIFACTS = ["scaler.joblib", "kmeans.joblib", "cluster_names.joblib", "xgb_model.joblib",
             "category_levels.joblib", "cpi_ref_quarter.joblib", "fare_levels.joblib",
             "route_snapshot.joblib", "metrics.json"]

SEASONS = {1: "Winter (Q1)", 2: "Spring (Q2)", 3: "Summer (Q3)", 4: "Fall/Holiday (Q4)"}
CARRIERS = {
    "AA": "American", "DL": "Delta", "UA": "United", "WN": "Southwest", "AS": "Alaska",
    "B6": "JetBlue", "F9": "Frontier", "NK": "Spirit", "G4": "Allegiant", "SY": "Sun Country",
    "MX": "Breeze", "XP": "Avelo", "HA": "Hawaiian", "FL": "AirTran",
}


def carrier_name(code):
    return CARRIERS.get(code, code)


def missing_artifacts():
    return [a for a in ARTIFACTS if not (MODELS / a).exists()]


def load_artifacts():
    missing = missing_artifacts()
    if missing:
        raise FileNotFoundError(f"missing artifacts: {missing}")
    art = {Path(a).stem: joblib.load(MODELS / a) for a in ARTIFACTS if a.endswith(".joblib")}
    art["metrics"] = json.loads((MODELS / "metrics.json").read_text())
    return art


def build_row(route, quarter, passengers=None, large_ms=None, lf_ms=None):
    """One-row feature frame for a snapshot route and quarter, with optional what-if overrides."""
    row = dict(route)
    row["quarter"] = int(quarter)
    # the route's most recent fare for this season, relative to that quarter's national level
    row["prior_rel"] = row.get(f"prior_rel_q{int(quarter)}", np.nan)
    for key, value in (("passengers", passengers), ("large_ms", large_ms), ("lf_ms", lf_ms)):
        if value is not None:
            row[key] = float(value)
    return add_derived(pd.DataFrame([row]))


def predict_fare(art, route, quarter, passengers=None, large_ms=None, lf_ms=None):
    """Predict the average one-way fare for a route snapshot row and a quarter.

    `route` is a row (Series/dict) of route_snapshot; the three optional
    arguments override its latest values (what-if). Returns a dict with the
    fare in CPI-reference-quarter dollars and each feature's contribution.
    """
    frame = build_row(route, quarter, passengers, large_ms, lf_ms)
    X = make_X(frame, art["category_levels"])

    booster = art["xgb_model"].get_booster()
    contribs = booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0]
    # Contributions + bias add up to the fare relative to the national level for that
    # season; adding the latest known level turns it into dollars.
    log_fare = float(contribs.sum()) + art["fare_levels"][int(quarter)]["level"]
    return {
        "fare": float(np.exp(log_fare)),
        "contributions": dict(zip(FEATURES, contribs[:-1].astype(float))),  # log-fare units
        "inputs": frame.iloc[0].to_dict(),
    }


def describe_driver(feature, contribution, inputs, cluster_name):
    """One plain-English sentence for a feature contribution (log-fare units)."""
    prior = inputs.get("prior_rel", np.nan)
    if pd.isna(prior):
        history = "Having no earlier fare on record for this route and season"
    else:
        history = (f"This route's last fare for this season being {abs(np.exp(prior) - 1):.0%} "
                   f"{'above' if prior > 0 else 'below'} the national average")
    what = {
        "prior_rel": history,
        "nsmiles": f"A distance of {inputs['nsmiles']:,.0f} miles",
        "log_passengers": f"Traffic of about {inputs['passengers']:,.0f} passengers a day",
        "large_ms": f"The largest carrier ({carrier_name(inputs['carrier_lg'])}) holding "
                    f"{inputs['large_ms']:.0%} of the market",
        "lf_ms": f"The lowest-fare carrier ({carrier_name(inputs['carrier_low'])}) holding "
                 f"{inputs['lf_ms']:.0%} of the market",
        "carrier_lg": f"{carrier_name(inputs['carrier_lg'])} being the largest carrier",
        "carrier_low": f"{carrier_name(inputs['carrier_low'])} being the lowest-fare carrier",
        "low_is_largest": ("The largest carrier also being the cheapest" if inputs["low_is_largest"]
                           else "A different carrier undercutting the largest one"),
        "quarter": f"Travelling in {SEASONS[int(inputs['quarter'])].split(' (')[0].lower()}",
        "city_a": f"Flying to or from {inputs['city_a']}",
        "city_b": f"Flying to or from {inputs['city_b']}",
        "cluster": f"Being a “{cluster_name}” route",
    }[feature]
    pct = np.exp(contribution) - 1
    direction = "raises" if pct > 0 else "lowers"
    return f"{what} {direction} the fare by about {abs(pct):.0%}."


def top_drivers(result, cluster_name, n=3):
    ranked = sorted(result["contributions"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    return [(f, c, describe_driver(f, c, result["inputs"], cluster_name)) for f, c in ranked]
