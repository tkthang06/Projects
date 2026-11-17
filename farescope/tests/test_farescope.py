"""Run from the project root:  python -m pytest"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import clean, cluster, download, features, predict  # noqa: E402

needs_raw = pytest.mark.skipif(not download.TABLE6_PATH.exists(), reason="run the pipeline first")
needs_processed = pytest.mark.skipif(not clean.FARES_PATH.exists(), reason="run the pipeline first")
needs_models = pytest.mark.skipif(bool(predict.missing_artifacts()), reason="run the pipeline first")


# ---- expected columns exist ------------------------------------------------
@needs_raw
def test_raw_table6_has_expected_columns_and_types():
    df = download.load_table6()
    assert set(download.EXPECTED_COLUMNS) <= set(df.columns)
    assert df["year"].min() >= 2012
    for c in download.FLOAT_COLS:
        assert df[c].dtype == "float64"


@needs_raw
def test_cpi_is_quarterly():
    cpi = download.load_cpi_quarterly()
    assert {"year", "quarter", "cpi"} <= set(cpi.columns)
    assert not cpi.duplicated(["year", "quarter"]).any()
    assert cpi["quarter"].between(1, 4).all()


@needs_processed
def test_processed_columns_and_rules():
    df = pd.read_parquet(clean.FARES_PATH)
    assert {"route_id", "city_a", "city_b", "real_fare", *features.NUMERIC[:1]} <= set(df.columns)
    assert not clean.is_covid(df["year"], df["quarter"]).any()
    assert (df["real_fare"] > 0).all() and (df["nsmiles"] > 0).all()
    assert not df[["fare", "nsmiles", "passengers", "large_ms", "lf_ms"]].isna().any().any()
    assert not df.duplicated(["year", "quarter", "route_id"]).any()
    assert not df["carrier_lg"].isin(clean.CARRIER_MERGES).any()
    assert (df["city_id_a"] < df["city_id_b"]).all()


# ---- no leakage --------------------------------------------------------------
def test_leaky_columns_are_not_features():
    assert not set(features.FEATURES) & set(features.EXCLUDED)
    assert not {"fare_lg", "fare_low", "fare", "real_fare", "year"} & set(features.FEATURES)


@needs_processed
def test_leaky_columns_never_reach_processed_data():
    assert not {"fare_lg", "fare_low"} & set(pd.read_parquet(clean.FARES_PATH).columns)


def test_split_is_time_based_and_disjoint():
    df = pd.DataFrame({"year": range(2012, 2027)})
    tr, va, te = features.split_masks(df)
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert (tr | va | te).all()
    assert df.year[tr].max() < df.year[va].min() <= df.year[va].max() < df.year[te].min()


def test_time_context_only_looks_backwards():
    def frame(fare_2024):
        return pd.DataFrame({
            "route_id": ["r", "r", "s", "s"], "year": [2023, 2024, 2023, 2024], "quarter": 1,
            "real_fare": [100.0, fare_2024, 400.0, 300.0],
        })
    a, levels = features.add_time_context(frame(200.0))
    b, _ = features.add_time_context(frame(999.0))  # change only the current-year target
    row_a, row_b = a[(a.route_id == "r") & (a.year == 2024)].iloc[0], b[(b.route_id == "r") & (b.year == 2024)].iloc[0]
    # the feature and the level used for forecasting depend on last year only
    assert row_a["prior_rel"] == row_b["prior_rel"] and row_a["level_prev"] == row_b["level_prev"]
    level_2023 = np.log([100.0, 400.0]).mean()
    assert row_a["level_prev"] == pytest.approx(level_2023)
    assert row_a["prior_rel"] == pytest.approx(np.log(100.0) - level_2023)
    assert a.loc[a.year == 2023, ["prior_rel", "level_prev"]].isna().all().all()  # nothing earlier exists
    assert len(a) == 4 and len(levels) == 2
    # no year-over-year change is known before 2024, so the forecast has no drift yet
    assert row_a["level_forecast"] == row_b["level_forecast"] == pytest.approx(level_2023)


def test_level_forecast_adds_average_past_drift():
    # one route, level falls 0.10 then 0.20 in log terms; 2025's own value must not matter
    def frame(fare_2025):
        return pd.DataFrame({"route_id": "r", "year": [2022, 2023, 2024, 2025], "quarter": 1,
                             "real_fare": np.exp([6.0, 5.9, 5.7, fare_2025])})
    a, levels = features.add_time_context(frame(5.0))
    b, _ = features.add_time_context(frame(9.0))
    f = a.set_index("year")["level_forecast"]
    assert f[2024] == pytest.approx(5.9 - 0.10)            # last level + the one change seen so far
    assert f[2025] == pytest.approx(5.7 - 0.15)            # last level + mean of (-0.10, -0.20)
    assert b.set_index("year")["level_forecast"][2025] == pytest.approx(f[2025])
    assert levels.set_index("year")["next_level"][2024] == pytest.approx(f[2025])


@needs_models
@needs_processed
def test_kmeans_and_scaler_saw_training_routes_only():
    art = predict.load_artifacts()
    df = pd.read_parquet(clean.FARES_PATH)
    n_train_routes = df.loc[df["year"] <= features.TRAIN_END_YEAR, "route_id"].nunique()
    assert art["scaler"].n_samples_seen_ == n_train_routes
    assert sum(c["routes"] for c in art["metrics"]["clusters"]["centroids"]) == n_train_routes


@needs_models
def test_model_uses_exactly_the_declared_features():
    art = predict.load_artifacts()
    assert list(art["xgb_model"].get_booster().feature_names) == features.FEATURES


# ---- cleaning logic ----------------------------------------------------------
def test_route_key_ignores_direction():
    df = pd.DataFrame({"citymarketid_1": [30135, 30194], "citymarketid_2": [30194, 30135]})
    out = clean.add_route_keys(df)
    assert out["route_id"].nunique() == 1
    assert (out["city_id_a"] == 30135).all() and (out["city_id_b"] == 30194).all()


def test_covid_window():
    y = pd.Series([2020, 2020, 2021, 2021, 2022, 2019])
    q = pd.Series([1, 2, 1, 4, 1, 4])
    assert clean.is_covid(y, q).tolist() == [False, True, True, True, False, False]


def test_parse_point_is_lon_lat():
    lon, lat = clean.parse_point(pd.Series(["POINT (-75.469759 40.602753)", None], dtype="string"))
    assert (lon[0], lat[0]) == (-75.469759, 40.602753)
    assert np.isnan(lon[1])


def test_repair_coordinates_fixes_wrong_and_missing_cities():
    true = pd.DataFrame({
        "lat": [47.6, 34.05, 39.74, 32.9, 41.88, 33.75, 42.36, 25.8],
        "lon": [-122.3, -118.2, -105.0, -97.0, -87.6, -84.4, -71.1, -80.3],
    }, index=range(1, 9))
    routes = pd.DataFrame(
        [(a, b, clean.haversine_miles(true.lat[a], true.lon[a], true.lat[b], true.lon[b]))
         for a, b in itertools.combinations(true.index, 2)],
        columns=["city_id_a", "city_id_b", "miles"])
    given = true.copy()
    given.loc[4] = [40.1, -77.0]      # Dallas geocoded to Pennsylvania, as in the real file
    given.loc[8] = [np.nan, np.nan]   # Miami missing
    fixed = clean.repair_coordinates(given, routes)
    assert fixed.loc[4, "coord_source"] == "estimated" and fixed.loc[8, "coord_source"] == "estimated"
    assert (fixed.loc[[1, 2, 3, 5, 6, 7], "coord_source"] == "dot").all()
    err = clean.haversine_miles(fixed.lat, fixed.lon, true.lat, true.lon)
    assert err.max() < 5


# ---- features / clustering ---------------------------------------------------
def test_unseen_categories_become_missing_and_do_not_crash():
    import xgboost as xgb
    rng = np.random.default_rng(0)
    n = 200
    df = features.add_derived(pd.DataFrame({
        "nsmiles": rng.uniform(100, 2500, n), "passengers": rng.uniform(10, 500, n),
        "large_ms": rng.uniform(0.2, 1, n), "lf_ms": rng.uniform(0.1, 1, n),
        "carrier_lg": rng.choice(["AA", "DL"], n), "carrier_low": rng.choice(["AA", "WN"], n),
        "quarter": rng.integers(1, 5, n), "city_a": rng.choice(["X", "Y"], n),
        "city_b": rng.choice(["Z", "W"], n), "cluster": rng.integers(0, 3, n),
        "prior_rel": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 0.2, n)),
    }))
    levels = features.fit_category_levels(df)
    X = features.make_X(df, levels)
    model = xgb.XGBRegressor(n_estimators=5, enable_categorical=True, tree_method="hist").fit(X, rng.normal(size=n))

    new = df.head(3).assign(carrier_lg="ZZ", city_a="Nowhere", cluster=99)
    X_new = features.make_X(new, levels)
    assert X_new["carrier_lg"].isna().all() and X_new["city_a"].isna().all()
    assert list(X_new["carrier_lg"].cat.categories) == list(X["carrier_lg"].cat.categories)
    assert np.isfinite(model.predict(X_new)).all()


def test_low_is_largest():
    df = features.add_derived(pd.DataFrame(
        {"passengers": [10.0, 10.0], "carrier_lg": ["AA", "AA"], "carrier_low": ["AA", "WN"]}))
    assert df["low_is_largest"].tolist() == [1, 0]


def test_cluster_names_are_unique_and_rule_based():
    z = pd.DataFrame([
        [0.2, 1.4, -0.8, -0.6, 0.1],    # busy, competitive
        [-0.9, -0.5, 1.2, 1.0, -0.2],   # short, thin, dominated
        [0.1, -0.2, 0.0, 0.1, 2.0],     # seasonal
        [1.5, 0.9, 0.3, 0.2, 0.0],      # long and busy
        [0.1, -0.4, -0.1, -0.2, -0.1],  # nothing distinctive
        [0.3, 1.6, -0.9, -0.7, 0.0],    # second busy/competitive cluster -> must not reuse the name
    ], columns=cluster.CLUSTER_FEATURES)
    names = cluster.name_clusters(z)
    assert len(set(names.values())) == 6
    assert names[1] == "Small Regional" and names[2] == "Vacation Routes" and names[3] == "Long-Haul Trunk"
    assert names[5] == "Busy Hub-to-Hub" and names[0].startswith("Cluster 0:") and names[4].startswith("Cluster 4:")


def test_seasonality_is_clipped_and_filled():
    df = pd.DataFrame({
        "route_id": ["a", "a", "b", "c", "c"], "quarter": [1, 3, 2, 1, 3],
        "passengers": [10.0, 1000.0, 50.0, 100.0, 120.0],
        "nsmiles": 500.0, "large_ms": 0.5, "lf_ms": 0.3,
    })
    feats, fill = cluster.route_features(df)
    assert feats.loc["a", "seasonality"] == cluster.SEASONALITY_CLIP[1]
    assert feats.loc["b", "seasonality"] == fill
    assert feats.loc["c", "seasonality"] == pytest.approx(1.2)


# ---- saved model --------------------------------------------------------------
@needs_models
def test_prediction_matches_model_and_is_direction_free():
    art = predict.load_artifacts()
    route = art["route_snapshot"].iloc[0]
    res = predict.predict_fare(art, route, 3)
    X = features.make_X(predict.build_row(route, 3), art["category_levels"])
    expected = np.exp(art["xgb_model"].predict(X)[0] + art["fare_levels"][3]["level"])
    assert res["fare"] == pytest.approx(float(expected), rel=1e-3)
    assert 30 < res["fare"] < 1500
    assert len(predict.top_drivers(res, "x")) == 3
    changed = predict.predict_fare(art, route, 3, large_ms=0.99, lf_ms=0.99, passengers=5000)
    assert changed["fare"] != res["fare"]


@needs_models
def test_snapshot_has_one_row_per_route():
    snap = predict.load_artifacts()["route_snapshot"]
    assert snap["route_id"].is_unique
    assert not snap[["nsmiles", "passengers", "large_ms", "lf_ms", "cluster"]].isna().any().any()
