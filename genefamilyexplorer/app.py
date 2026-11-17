"""Gene Family Explorer: Streamlit interface. All analysis lives in sequence.py / compare.py / ncbi.py.

Run: streamlit run app.py
"""
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from compare import (DEFAULT_GENES, feature_table, kmer_table, kruskal_by_gene, pca,
                     plot_feature_heatmap, plot_pca, plot_windows, window_table)
from ncbi import fetch_gene
from sequence import (cpg_islands, find_orfs, gc_content, parse_fasta, reverse_complement,
                      sliding_gc, translate)

st.set_page_config(page_title="Gene Family Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def load_gene(symbol):
    return fetch_gene(symbol)


@st.cache_data(show_spinner=False)
def compare_tables(symbols):
    genes = [load_gene(s) for s in symbols]
    return feature_table(genes), kmer_table(genes), window_table(genes)


def show(fig):
    """Draw a matplotlib figure, then free it (Streamlit reruns would pile them up)."""
    st.pyplot(fig)
    plt.close(fig)


def plot_gc_islands(seq, title):
    """Sliding-window GC with CpG islands shaded."""
    window = 200 if len(seq) >= 400 else max(10, len(seq) // 4)
    step = max(1, len(seq) // 2000)  # about 2000 points whatever the length
    pos, gc = sliding_gc(seq, window, step)
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(pos, gc, lw=0.8)
    for s, e in cpg_islands(seq):
        ax.axvspan(s, e, color="tab:orange", alpha=0.3, lw=0)
    ax.set_xlim(0, len(seq))
    ax.set_xlabel("position (bp)")
    ax.set_ylabel(f"GC % ({window} bp window)")
    ax.set_title(f"{title}: GC content, CpG islands shaded orange")
    fig.tight_layout()
    return fig


def plot_orf_map(seq, orfs, title, highlight=None):
    """ORFs as bars, one row per strand and frame; highlight=(start, end) is drawn in red."""
    rows = [("+", 0), ("+", 1), ("+", 2), ("-", 0), ("-", 1), ("-", 2)]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    for o in orfs:
        hit = highlight == (o["start"], o["end"])
        ax.barh(rows.index((o["strand"], o["frame"])), o["end"] - o["start"], left=o["start"],
                height=0.6, color="tab:red" if hit else "tab:blue")
    ax.set_yticks(range(6), [f"{s} frame {f + 1}" for s, f in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, len(seq))
    ax.set_xlabel("position (bp)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def orf_frame(orfs):
    return pd.DataFrame([{"strand": o["strand"], "frame": o["frame"] + 1, "start": o["start"] + 1,
                          "end": o["end"], "length (aa)": len(o["protein"]), "protein": o["protein"]}
                         for o in orfs])


# ---------------- sidebar: choose genes ----------------
st.sidebar.title("Gene Family Explorer")
text = st.sidebar.text_area("Human gene symbols", ", ".join(DEFAULT_GENES), height=120,
                            help="Any human gene symbols, separated by commas or spaces. "
                                 "Data comes from NCBI RefSeq and is cached after the first load.")
min_aa = st.sidebar.slider("Minimum ORF length (amino acids)", 30, 300, 100, 10)
symbols = list(dict.fromkeys(s.upper() for s in re.split(r"[,\s]+", text) if s))

genes, failed = [], []
with st.spinner("Fetching genes from NCBI (first time only)..."):
    for s in symbols:
        try:
            genes.append(load_gene(s))
        except Exception as e:  # one bad symbol shouldn't break the whole page
            failed.append(f"{s}: {e}")
for msg in failed:
    st.sidebar.warning(msg)
if not genes:
    st.info("Enter at least one human gene symbol in the sidebar.")
    st.stop()
symbols = tuple(g["symbol"] for g in genes)

tab_genes, tab_one, tab_cmp, tab_own = st.tabs(["Genes", "Single gene", "Compare genes", "Your sequence"])

# ---------------- tab 1: overview ----------------
with tab_genes:
    st.subheader("Loaded genes")
    st.dataframe(pd.DataFrame([{
        "gene": g["symbol"], "mRNA": g["mrna_id"], "location": f"{g['chrom']}:{g['start']:,}-{g['end']:,} ({g['strand']})",
        "gene region (bp)": len(g["region"]), "mRNA (bp)": len(g["mrna"]), "protein (aa)": len(g["protein"]),
    } for g in genes]), hide_index=True, use_container_width=True)
    st.caption("Gene region = exons + introns from NCBI Gene, given 5'→3' in the gene's direction. "
               "mRNA = the RefSeq Select transcript.")

# ---------------- tab 2: one gene ----------------
with tab_one:
    g = load_gene(st.selectbox("Gene", symbols))
    region, mrna = g["region"], g["mrna"]
    cds = (g["cds_start"], g["cds_end"])
    islands = cpg_islands(region)
    c = st.columns(5)
    c[0].metric("Gene region", f"{len(region):,} bp")
    c[1].metric("GC (region)", f"{gc_content(region):.1f}%")
    c[2].metric("CpG islands", len(islands))
    c[3].metric("Protein", f"{len(g['protein'])} aa")
    c[4].metric("Coding share", f"{(cds[1] - cds[0]) / len(region) * 100:.1f}%")

    show(plot_gc_islands(region, f"{g['symbol']} gene region"))

    mrna_orfs = find_orfs(mrna, min_aa)
    longest_is_cds = bool(mrna_orfs) and (mrna_orfs[0]["start"], mrna_orfs[0]["end"]) == cds
    show(plot_orf_map(mrna, mrna_orfs, f"{g['mrna_id']} mRNA: ORFs ≥ {min_aa} aa, annotated CDS in red", cds))
    if longest_is_cds:
        same = "matches" if mrna_orfs[0]["protein"] == g["protein"] else "does NOT match"
        st.success(f"The longest ORF ({mrna_orfs[0]['start'] + 1}..{mrna_orfs[0]['end']}) is exactly the "
                   f"annotated CDS, and its translated protein {same} NCBI's.")
    else:
        st.warning("The longest ORF is not the annotated CDS for this gene.")
    with st.expander(f"ORF table ({len(mrna_orfs)} ORFs in the mRNA)"):
        st.dataframe(orf_frame(mrna_orfs), hide_index=True, use_container_width=True)
    with st.expander("ORFs in the gene region instead (mostly chance hits in introns)"):
        region_orfs = find_orfs(region, min_aa)
        show(plot_orf_map(region, region_orfs, f"{g['symbol']} gene region: ORFs ≥ {min_aa} aa"))
        st.caption(f"{len(region_orfs)} ORFs; the longest is {len(region_orfs[0]['protein']) if region_orfs else 0} aa, "
                   f"while the real protein is {len(g['protein'])} aa split across exons.")
    with st.expander("CpG islands"):
        st.dataframe(pd.DataFrame([{"start": s + 1, "end": e, "length": e - s, "GC %": round(gc_content(region[s:e]), 1)}
                                   for s, e in islands]), hide_index=True)

# ---------------- tab 3: compare ----------------
with tab_cmp:
    if len(genes) < 3:
        st.info("Load at least 3 genes to compare them.")
    else:
        features, kmers, windows = compare_tables(symbols)
        show(plot_feature_heatmap(features))
        st.download_button("Download feature table (CSV)", features.to_csv(), "gene_features.csv")

        left, right = st.columns(2)
        with left:
            show(plot_pca(*pca(features), "PCA of gene features"))
        with right:
            show(plot_pca(*pca(kmers), "PCA of 3-mer composition"))
        st.caption("3-mer composition tracks GC content and genomic neighborhood more than family history: "
                   "neighbors on the same chromosome often land close together.")

        metric = st.radio("Window metric", ["gc_%", "cpg_oe"], horizontal=True,
                          format_func={"gc_%": "GC %", "cpg_oe": "CpG observed/expected"}.get)
        show(plot_windows(windows, metric))
        stat, p = kruskal_by_gene(windows, metric)
        st.caption(f"Kruskal-Wallis across genes on 1 kb windows: H = {stat:.1f}, p = {p:.2g}. Exploratory only: "
                   "neighboring windows are correlated, so this p-value is too optimistic.")

# ---------------- tab 4: user's own sequence ----------------
with tab_own:
    st.subheader("Analyze your own sequence")
    upload = st.file_uploader("Upload a FASTA file", type=["fa", "fasta", "fna", "txt"])
    pasted = st.text_area("...or paste a sequence (FASTA or plain)", height=150,
                          placeholder=">my_seq\nATGGCC...")
    raw = upload.read().decode(errors="replace") if upload else pasted
    records = parse_fasta(raw) if raw.strip() else []
    if not records:
        st.info("Upload or paste a DNA sequence to see GC content, CpG islands and ORFs.")
    else:
        names = [h or f"sequence {i + 1}" for i, (h, _) in enumerate(records)]
        idx = st.selectbox("Record", range(len(records)), format_func=lambda i: names[i]) if len(records) > 1 else 0
        name, seq = names[idx], records[idx][1]
        bad = len(seq) - sum(seq.count(b) for b in "ACGT")
        c = st.columns(4)
        c[0].metric("Length", f"{len(seq):,} bp")
        c[1].metric("GC", f"{gc_content(seq):.1f}%")
        c[2].metric("CpG islands", len(cpg_islands(seq)))
        c[3].metric("Non-ACGT bases", bad)
        if len(seq) < 50:
            st.warning("Very short sequence; windowed GC and ORFs won't say much.")
        show(plot_gc_islands(seq, name))
        orfs = find_orfs(seq, min_aa)
        show(plot_orf_map(seq, orfs, f"{name}: ORFs ≥ {min_aa} aa"))
        st.dataframe(orf_frame(orfs), hide_index=True, use_container_width=True)
        with st.expander("Reverse complement and translation (frame +1)"):
            st.code(reverse_complement(seq), language=None, wrap_lines=True)
            st.code(translate(seq), language=None, wrap_lines=True)
