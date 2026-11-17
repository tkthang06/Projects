"""Model 1: K-Means route clustering on training-year route profiles."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features import split_masks

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

CLUSTER_FEATURES = ["nsmiles", "log_passengers", "large_ms", "lf_ms", "seasonality"]
SEASONALITY_CLIP = (1 / 3, 3.0)  # Q3/Q1 passenger ratio; ~1st-99th percentile of training routes
K_RANGE = range(3, 9)
NAVY, GOLD = "#0F1B2D", "#C9A96E"


def route_features(df, seasonality_fill=None):
    """One row per route_id: mean distance, size, competition, and summer/winter ratio.

    seasonality = mean Q3 passengers / mean Q1 passengers. Routes missing Q1 or
    Q3 get `seasonality_fill` (the median of `df`'s routes when None).
    Returns (features, fill_value_used).
    """
    d = df.assign(log_passengers=np.log1p(df["passengers"]))
    feats = d.groupby("route_id")[["nsmiles", "log_passengers", "large_ms", "lf_ms"]].mean()
    q = d.pivot_table(index="route_id", columns="quarter", values="passengers", aggfunc="mean")
    q = q.reindex(columns=[1, 2, 3, 4])
    seas = (q[3] / q[1]).clip(*SEASONALITY_CLIP)
    if seasonality_fill is None:
        seasonality_fill = float(seas.median())
    feats["seasonality"] = seas.reindex(feats.index).fillna(seasonality_fill)
    return feats[CLUSTER_FEATURES], seasonality_fill


# (name, rule) - the rule returns a match score from a centroid's z-scores, or None.
NAME_RULES = [
    ("Vacation Routes", lambda z: z.seasonality if z.seasonality >= 0.5 else None),
    ("Long-Haul Trunk", lambda z: z.nsmiles + z.log_passengers if z.nsmiles >= 0.5 and z.log_passengers >= 0.3 else None),
    ("Busy Hub-to-Hub", lambda z: z.log_passengers - z.large_ms if z.log_passengers >= 0.5 and z.large_ms <= 0 else None),
    ("Small Regional", lambda z: z.large_ms - z.nsmiles - z.log_passengers
        if z.nsmiles < 0 and z.log_passengers < 0 and z.large_ms > 0 else None),
]
TRAITS = {
    "nsmiles": ("long distance", "short distance"),
    "log_passengers": ("high traffic", "low traffic"),
    "large_ms": ("one dominant carrier", "many competing carriers"),
    "lf_ms": ("low-fare carrier dominant", "low-fare carrier marginal"),
    "seasonality": ("summer-peaking", "winter-peaking"),
}


def name_clusters(centroids_z):
    """Name clusters from centroid z-scores. Names are unique by construction.

    Each rule name goes to the cluster that matches it best, each cluster gets
    at most one name, and anything left is "Cluster N: <top trait>".
    """
    candidates = []
    for cid, z in centroids_z.iterrows():
        for name, rule in NAME_RULES:
            score = rule(z)
            if score is not None:
                candidates.append((score, cid, name))
    names = {}
    for _, cid, name in sorted(candidates, reverse=True):
        if cid not in names and name not in names.values():
            names[cid] = name
    for cid, z in centroids_z.iterrows():
        if cid not in names:
            top = z.abs().idxmax()
            names[cid] = f"Cluster {cid}: {TRAITS[top][0] if z[top] > 0 else TRAITS[top][1]}"
    assert len(set(names.values())) == len(names)
    return {int(k): v for k, v in names.items()}


def _plot(x, y, title, ylabel, path, mark=None):
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    ax.plot(x, y, color=NAVY, marker="o", lw=1.5)
    if mark is not None:
        ax.scatter([mark], [y[list(x).index(mark)]], s=140, color=GOLD, zorder=3, label=f"chosen k = {mark}")
        ax.legend(frameon=False)
    ax.set(title=title, xlabel="k (number of clusters)", ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(df):
    """Fit on training-year routes, label every route, save artifacts.

    Returns (route_id -> cluster Series for all routes, info dict for metrics.json).
    """
    train_mask, _, _ = split_masks(df)
    train_feats, fill = route_features(df[train_mask])

    scaler = StandardScaler().fit(train_feats)
    Z = scaler.transform(train_feats)
    inertia, sil, fits = {}, {}, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        fits[k], inertia[k], sil[k] = km, float(km.inertia_), float(silhouette_score(Z, km.labels_))
        print(f"[cluster] k={k}: inertia={inertia[k]:,.0f}  silhouette={sil[k]:.4f}")
    best_k = max(sil, key=sil.get)
    kmeans = fits[best_k]
    print(f"[cluster] chosen k={best_k} (highest silhouette)")

    REPORTS.mkdir(exist_ok=True)
    ks = list(K_RANGE)
    _plot(ks, [inertia[k] for k in ks], "Elbow plot", "Inertia (within-cluster SS)", REPORTS / "elbow.png", best_k)
    _plot(ks, [sil[k] for k in ks], "Silhouette score by k", "Mean silhouette", REPORTS / "silhouette.png", best_k)

    centroids_z = pd.DataFrame(kmeans.cluster_centers_, columns=CLUSTER_FEATURES)
    names = name_clusters(centroids_z)

    # Human-checkable centroid table: z-scores plus original units
    raw = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=CLUSTER_FEATURES)
    table = pd.DataFrame({
        "cluster": centroids_z.index,
        "name": [names[i] for i in centroids_z.index],
        "routes": np.bincount(kmeans.labels_, minlength=best_k),
        "miles": raw.nsmiles.round(0),
        "passengers_per_day": np.expm1(raw.log_passengers).round(1),
        "largest_carrier_share": raw.large_ms.round(3),
        "low_fare_carrier_share": raw.lf_ms.round(3),
        "q3_q1_passenger_ratio": raw.seasonality.round(3),
    })
    for c in CLUSTER_FEATURES:
        table[f"z_{c}"] = centroids_z[c].round(2)
    print("[cluster] centroids (check the names against these):")
    print(table.to_string(index=False))
    table.to_csv(REPORTS / "cluster_centroids.csv", index=False)

    # Routes absent from training: same features from whatever rows they have
    new_routes = df[~df["route_id"].isin(train_feats.index)]
    new_feats, _ = route_features(new_routes, seasonality_fill=fill)
    all_feats = pd.concat([train_feats, new_feats])
    labels = pd.Series(kmeans.predict(scaler.transform(all_feats)), index=all_feats.index, name="cluster")
    print(f"[cluster] labelled {len(train_feats):,} training routes + {len(new_feats):,} routes absent from training")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS / "scaler.joblib")
    joblib.dump(kmeans, MODELS / "kmeans.joblib")
    joblib.dump({"names": names, "seasonality_fill": fill}, MODELS / "cluster_names.joblib")

    info = {
        "k": best_k,
        "inertia": {str(k): v for k, v in inertia.items()},
        "silhouette": {str(k): v for k, v in sil.items()},
        "names": {str(k): v for k, v in names.items()},
        "centroids": table.to_dict(orient="records"),
    }
    return labels, info
