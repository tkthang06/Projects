"""Model 2: XGBoost fare model - tuning, baselines, evaluation, artifacts."""
import itertools
import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

from src import cluster
from src.clean import CITIES_PATH, FARES_PATH
from src.features import (FEATURES, TARGET, TEST_START_YEAR, TRAIN_END_YEAR, VAL_YEAR,
                          add_derived, add_time_context, fit_category_levels, make_X, split_masks)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

GRID = {
    "max_depth": [6, 9],
    "learning_rate": [0.15, 0.3],
    "min_child_weight": [1, 10],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 1.0],
}
# The full 32-cell grid takes about an hour on a 12-core laptop and its validation MAE
# only spans ~$1 (measured on a partial run), so by default a fixed random sample of
# N_COMBOS cells is searched. Set N_COMBOS = None to search every cell.
N_COMBOS = 8
MAX_ROUNDS = 2000
EARLY_STOPPING = 30
SAMPLING_CHANGE = (2025, 3)  # DOT ticket sample grew from 10% to ~40% in 2025 Q3


def metrics(actual, pred):
    """Error metrics in dollars."""
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(root_mean_squared_error(actual, pred)),
        "mape": float(mean_absolute_percentage_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
        "bias": float(np.mean(np.asarray(pred) - np.asarray(actual))),  # > 0 means predictions run high
        "n": int(len(actual)),
    }


def _model(params, n_estimators, early_stopping=None):
    return xgb.XGBRegressor(
        n_estimators=n_estimators, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=early_stopping, random_state=42, n_jobs=-1, **params,
    )


def to_dollars(model, X, level_forecast):
    """Relative prediction -> dollars, using the price level known a year in advance."""
    return np.exp(model.predict(X) + np.asarray(level_forecast))


def fit_with_early_stopping(params, X_tr, d_tr, X_va, d_va):
    """Train on fare relative to its own quarter's level; early-stop on the real
    forecasting task (validation fare relative to last year's level).
    Returns (best number of trees, validation MAE in dollars)."""
    m = _model(params, MAX_ROUNDS, EARLY_STOPPING)
    m.fit(X_tr, d_tr["log_fare"] - d_tr["level"],
          eval_set=[(X_va, d_va["log_fare"] - d_va["level_forecast"])], verbose=False)
    return m.best_iteration + 1, float(mean_absolute_error(d_va[TARGET], to_dollars(m, X_va, d_va["level_forecast"])))


def last_year_baseline(df):
    """real_fare of the same route in the same quarter one year earlier (NaN if absent)."""
    prior = df[["route_id", "year", "quarter", TARGET]].copy()
    prior["year"] += 1
    merged = df[["route_id", "year", "quarter"]].merge(
        prior, on=["route_id", "year", "quarter"], how="left", validate="1:1")
    return pd.Series(merged[TARGET].values, index=df.index)


def build_snapshot(df, names):
    """Latest feature values per route (what the app predicts from) + recent actual fares."""
    df = df.sort_values(["year", "quarter"])
    snap = df.groupby("route_id").last()[
        ["city_id_a", "city_id_b", "city_a", "city_b", "nsmiles", "passengers", "large_ms", "lf_ms",
         "carrier_lg", "carrier_low", "cluster", "year", "quarter"]]
    snap["latest_period"] = snap["year"].astype(str) + " Q" + snap["quarter"].astype(str)
    snap = snap.drop(columns=["year", "quarter"])
    snap["cluster_name"] = snap["cluster"].map(names)
    # Most recent actual fare for each season
    for q in (1, 2, 3, 4):
        last_q = df[df["quarter"] == q].groupby("route_id").last()
        snap[f"actual_q{q}"] = last_q[TARGET].reindex(snap.index)
        snap[f"actual_q{q}_year"] = last_q["year"].reindex(snap.index).astype("Int64")
        # the app's value for the prior_rel feature: that fare relative to its quarter's level
        snap[f"prior_rel_q{q}"] = (last_q["log_fare"] - last_q["level"]).reindex(snap.index)
    # Coordinates for the map only
    cities = pd.read_parquet(CITIES_PATH).set_index("id")
    for side in ("a", "b"):
        snap[f"lat_{side}"] = snap[f"city_id_{side}"].map(cities["lat"])
        snap[f"lon_{side}"] = snap[f"city_id_{side}"].map(cities["lon"])
    return snap.reset_index()


def main():
    df, fare_levels = add_time_context(add_derived(pd.read_parquet(FARES_PATH)))
    cpi_ref = df["cpi_ref_quarter"].iloc[0]

    # ---- Model 1 -----------------------------------------------------------
    labels, cluster_info = cluster.main(df)
    names = {int(k): v for k, v in cluster_info["names"].items()}
    df["cluster"] = df["route_id"].map(labels).astype(int)

    # ---- Split -------------------------------------------------------------
    tr, va, te = split_masks(df)
    assert not (tr & va).any() and not (va & te).any() and (tr | va | te).all()
    last = df.loc[te].sort_values(["year", "quarter"]).iloc[-1]
    print(f"[train] train 2012-{TRAIN_END_YEAR}: {tr.sum():,} | val {VAL_YEAR}: {va.sum():,} | "
          f"test {TEST_START_YEAR}-{last.year} Q{last.quarter}: {te.sum():,}")

    # One set of category levels for every matrix (train, val, test, app).
    levels = fit_category_levels(df[tr | va])
    levels["cluster"] = sorted(names)
    feats_nc = [f for f in FEATURES if f != "cluster"]
    assert df.loc[va | te, "level_forecast"].notna().all(), "no prior-year price level for a val/test quarter"

    # ---- Grid search on validation with early stopping ---------------------
    X_tr, X_va = make_X(df[tr], levels), make_X(df[va], levels)
    grid = []
    combos = list(itertools.product(*GRID.values()))
    if N_COMBOS and N_COMBOS < len(combos):
        combos = random.Random(42).sample(combos, N_COMBOS)
    print(f"[train] searching {len(combos)} hyperparameter combinations")
    for combo in combos:
        params = dict(zip(GRID, combo))
        n_trees, val_mae = fit_with_early_stopping(params, X_tr, df[tr], X_va, df[va])
        grid.append({**params, "n_trees": n_trees, "val_mae": val_mae})
        print(f"[train] {params} -> trees={n_trees}, val MAE=${val_mae:.2f}")
    best = min(grid, key=lambda g: g["val_mae"])
    params = {k: best[k] for k in GRID}
    print(f"[train] best: {params}, trees={best['n_trees']}, val MAE=${best['val_mae']:.2f}")

    # ---- Refit on train+validation with the best iteration count ------------
    tv = tr | va
    y_rel = (df["log_fare"] - df["level"])[tv]
    model = _model(params, best["n_trees"]).fit(make_X(df[tv], levels), y_rel)

    # Baseline (b): identical model without the cluster feature
    nc_trees, nc_val_mae = fit_with_early_stopping(
        params, X_tr[feats_nc], df[tr], X_va[feats_nc], df[va])
    model_nc = _model(params, nc_trees).fit(make_X(df[tv], levels, feats_nc), y_rel)

    # ---- Test evaluation (dollars) ------------------------------------------
    test = df[te].copy()
    X_te = make_X(test, levels)
    test["pred"] = to_dollars(model, X_te, test["level_forecast"])
    test["pred_nc"] = to_dollars(model_nc, X_te[feats_nc], test["level_forecast"])
    test["pred_last_year"] = last_year_baseline(df)[te]

    overall = metrics(test[TARGET], test["pred"])
    by_cluster = {names[c]: metrics(g[TARGET], g["pred"]) for c, g in test.groupby("cluster")}
    after = (test["year"] > SAMPLING_CHANGE[0]) | (
        (test["year"] == SAMPLING_CHANGE[0]) & (test["quarter"] >= SAMPLING_CHANGE[1]))
    sampling = {
        "before_2025Q3": metrics(test.loc[~after, TARGET], test.loc[~after, "pred"]),
        "from_2025Q3": metrics(test.loc[after, TARGET], test.loc[after, "pred"]),
    }

    has_prior = test["pred_last_year"].notna()
    sub = test[has_prior]
    baseline_last_year = {
        "coverage": float(has_prior.mean()),
        "baseline": metrics(sub[TARGET], sub["pred_last_year"]),
        "model_same_rows": metrics(sub[TARGET], sub["pred"]),
    }
    no_cluster = metrics(test[TARGET], test["pred_nc"])
    no_cluster.update(n_trees=nc_trees, val_mae=nc_val_mae)

    diff = no_cluster["mae"] - overall["mae"]  # > 0 means the cluster feature helps
    rel = diff / no_cluster["mae"]
    if rel > 0.005:
        verdict = (f"Model 1 improves results: adding the cluster feature lowers test MAE by "
                   f"${diff:.2f} ({rel:.1%}), from ${no_cluster['mae']:.2f} to ${overall['mae']:.2f}.")
    elif rel < -0.005:
        verdict = (f"Model 1 does NOT improve results: adding the cluster feature raises test MAE by "
                   f"${-diff:.2f} ({-rel:.1%}), from ${no_cluster['mae']:.2f} to ${overall['mae']:.2f}.")
    else:
        verdict = (f"Model 1 does NOT meaningfully change results: test MAE is ${overall['mae']:.2f} with "
                   f"the cluster feature and ${no_cluster['mae']:.2f} without it (difference "
                   f"${diff:+.2f}, under 0.5%). The clusters are useful for describing routes, "
                   f"not for accuracy.")

    def show(label, m):
        print(f"  {label:<34} n={m['n']:>6,}  MAE ${m['mae']:6.2f}  RMSE ${m['rmse']:6.2f}  "
              f"MAPE {m['mape']:6.2%}  R2 {m['r2']:.3f}  bias ${m['bias']:+6.2f}")
    print("[train] TEST RESULTS (latest-quarter dollars)")
    show("XGBoost + cluster (overall)", overall)
    for k, m in by_cluster.items():
        show(f"  cluster: {k}", m)
    show("before 2025 Q3", sampling["before_2025Q3"])
    show("from 2025 Q3", sampling["from_2025Q3"])
    show("baseline (b) XGBoost, no cluster", no_cluster)
    print(f"  baseline (a) exists for {baseline_last_year['coverage']:.1%} of test rows:")
    show("baseline (a) same quarter last yr", baseline_last_year["baseline"])
    show("XGBoost + cluster, same rows", baseline_last_year["model_same_rows"])
    print("[train]", verdict)

    # ---- Feature importance (gain) ------------------------------------------
    gain = model.get_booster().get_score(importance_type="gain")
    importance = {f: float(gain.get(f, 0.0)) for f in FEATURES}
    imp = pd.Series(importance).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.barh(imp.index, imp.values, color="#0F1B2D")
    ax.set(title="Feature importance (average gain per split)", xlabel="Gain")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "feature_importance.png", facecolor="white")
    plt.close(fig)

    # ---- Save artifacts -------------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "xgb_model.joblib")
    joblib.dump(levels, MODELS / "category_levels.joblib")
    joblib.dump(cpi_ref, MODELS / "cpi_ref_quarter.joblib")
    # Forecast price level for the next occurrence of each season (the app's level_forecast)
    latest = fare_levels.sort_values("year").groupby("quarter").last()
    joblib.dump({int(q): {"level": float(r.next_level), "year": int(r.year) + 1} for q, r in latest.iterrows()},
                MODELS / "fare_levels.joblib")
    joblib.dump(build_snapshot(df, names), MODELS / "route_snapshot.joblib")
    out = {
        "cpi_ref_quarter": cpi_ref,
        "split": {"train": f"2012-{TRAIN_END_YEAR} (2020 Q2-2021 Q4 removed)", "validation": str(VAL_YEAR),
                  "test": f"{TEST_START_YEAR} Q1 - {last.year} Q{last.quarter}",
                  "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
        "features": FEATURES,
        "best_params": params, "n_trees": best["n_trees"], "val_mae": best["val_mae"], "grid": grid,
        "test": {"overall": overall, "by_cluster": by_cluster, "sampling_change": sampling},
        "baselines": {"same_quarter_last_year": baseline_last_year, "xgboost_no_cluster": no_cluster},
        "cluster_verdict": verdict,
        "feature_importance_gain": importance,
        "clusters": cluster_info,
    }
    (MODELS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"[train] artifacts saved to {MODELS}")
    return out


if __name__ == "__main__":
    main()
