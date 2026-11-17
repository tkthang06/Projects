"""Compare genes: one feature table, windowed statistics, PCA and charts.

Run: python compare.py   (ADAMTS family + GAPDH outgroup; saves charts to figures/)
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

from sequence import cpg_islands, cpg_obs_exp, find_orfs, gc_content, kmer_counts

DEFAULT_GENES = ["ADAMTS1", "ADAMTS4", "ADAMTS5", "ADAMTS8", "ADAMTS9", "ADAMTS15", "GAPDH"]


def gene_features(g):
    """One row of numbers describing a gene (g comes from ncbi.fetch_gene)."""
    region, cds = g["region"], g["mrna"][g["cds_start"]:g["cds_end"]]
    islands = cpg_islands(region)
    region_orfs = find_orfs(region, min_aa=100)
    return {
        "region_kb": len(region) / 1000,
        "protein_aa": len(g["protein"]),
        "utr5_bp": g["cds_start"],
        "utr3_bp": len(g["mrna"]) - g["cds_end"],
        "coding_share_%": len(cds) / len(region) * 100,
        "gc_region_%": gc_content(region),
        "gc_cds_%": gc_content(cds),
        "gc3_%": gc_content(cds[2::3]),  # third codon positions, where synonymous changes sit
        "cpg_oe_region": cpg_obs_exp(region),
        "cpg_islands": len(islands),
        "island_cover_%": sum(e - s for s, e in islands) / len(region) * 100,
        "island_at_5p": int(any(s < 2000 for s, _ in islands)),
        "orfs_per_kb": len(region_orfs) / len(region) * 1000,  # ORFs >= 100 aa, mostly chance in introns
        "longest_region_orf_aa": len(region_orfs[0]["protein"]) if region_orfs else 0,
    }


def feature_table(genes):
    return pd.DataFrame({g["symbol"]: gene_features(g) for g in genes}).T


def kmer_table(genes, k=3):
    """Gene x k-mer frequency table (each row sums to 1), from the genomic region."""
    rows = {}
    for g in genes:
        counts = kmer_counts(g["region"], k)
        total = sum(counts.values())
        rows[g["symbol"]] = {kmer: n / total for kmer, n in counts.items()}
    return pd.DataFrame(rows).T.fillna(0).sort_index(axis=1)


def pca(table, n=2):
    """PCA on z-scored columns via SVD. Returns (scores DataFrame, % variance per PC)."""
    X = table.loc[:, table.std() > 0]
    X = (X - X.mean()) / X.std()
    U, S, _ = np.linalg.svd(X.to_numpy(float), full_matrices=False)
    var = S ** 2 / (S ** 2).sum() * 100
    cols = [f"PC{i + 1}" for i in range(n)]
    return pd.DataFrame(U[:, :n] * S[:n], index=table.index, columns=cols), var[:n]


def window_table(genes, window=1000):
    """GC and CpG obs/exp in non-overlapping windows of each gene region."""
    rows = []
    for g in genes:
        r = g["region"]
        for s in range(0, len(r) - window + 1, window):
            w = r[s:s + window]
            rows.append({"gene": g["symbol"], "start": s, "gc_%": gc_content(w), "cpg_oe": cpg_obs_exp(w)})
    return pd.DataFrame(rows)


def kruskal_by_gene(windows, column="gc_%"):
    """Kruskal-Wallis across genes on window values.

    Exploratory only: neighbouring windows are correlated, so the p-value is
    optimistic. Read it as 'distributions look different', not proof.
    """
    groups = [grp[column].dropna().to_numpy() for _, grp in windows.groupby("gene")]
    return kruskal(*groups)


def plot_feature_heatmap(features):
    z = (features - features.mean()) / features.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(features) + 2))
    im = ax.imshow(z.to_numpy(float), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(z.columns)), z.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(z.index)), z.index)
    for i in range(len(features.index)):  # print the real value, color shows the z-score
        for j in range(len(features.columns)):
            ax.text(j, i, f"{features.iat[i, j]:.3g}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="z-score across genes")
    ax.set_title("Gene features (color = how unusual vs the other genes)")
    fig.tight_layout()
    return fig


def plot_pca(scores, var, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(scores["PC1"], scores["PC2"])
    for name, (x, y) in scores.iterrows():
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel(f"PC1 ({var[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}% of variance)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_windows(windows, column="gc_%"):
    order = windows.groupby("gene")[column].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([windows.loc[windows.gene == g, column].dropna() for g in order], tick_labels=order)
    stat, p = kruskal_by_gene(windows, column)
    ax.set_ylabel(column)
    ax.set_title(f"{column} in 1 kb windows  (Kruskal-Wallis H={stat:.1f}, p={p:.2g}, exploratory)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from ncbi import fetch_gene

    matplotlib.use("Agg")
    genes = [fetch_gene(s) for s in DEFAULT_GENES]
    features = feature_table(genes)
    windows = window_table(genes)
    pd.set_option("display.width", 200)
    print(features.round(2).to_string())
    for col in ("gc_%", "cpg_oe"):
        stat, p = kruskal_by_gene(windows, col)
        print(f"\nKruskal-Wallis on 1 kb window {col}: H={stat:.1f}, p={p:.2g} (exploratory)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out, exist_ok=True)
    plot_feature_heatmap(features).savefig(os.path.join(out, "features.png"), dpi=150)
    plot_pca(*pca(features), "PCA of gene features").savefig(os.path.join(out, "pca_features.png"), dpi=150)
    plot_pca(*pca(kmer_table(genes)), "PCA of 3-mer composition").savefig(os.path.join(out, "pca_kmers.png"), dpi=150)
    plot_windows(windows).savefig(os.path.join(out, "window_gc.png"), dpi=150)
    print(f"\nCharts saved to {out}")
