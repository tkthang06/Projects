"""Gene Family Explorer: Streamlit interface. All analysis lives in sequence.py / compare.py / ncbi.py.

Run: streamlit run app.py
"""
import re

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import py3Dmol
import streamlit as st

from compare import (DEFAULT_GENES, feature_table, kruskal_by_gene, pca, plot_distance, plot_feature_heatmap,
                     plot_pca, plot_windows, signature_distance, signature_table, window_table)
from ncbi import fetch_gene
from sequence import (cpg_islands, find_orfs, gc_content, parse_fasta, reverse_complement,
                      sliding_gc, translate)
from structure import fetch_structure, find_motif, residue_plddt

st.set_page_config(page_title="Gene Family Explorer", layout="wide")

ROWS = [f"{s} frame {f}" for s in "+-" for f in (1, 2, 3)]
# AlphaFold's own confidence colors
PLDDT_BINS = [(90, 101, "#0053D6", "very high (≥90)"), (70, 90, "#65CBF3", "confident (70–90)"),
              (50, 70, "#FFDB13", "low (50–70)"), (0, 50, "#FF7D45", "very low (<50)")]


@st.cache_data(show_spinner=False)
def load_gene(symbol):
    return fetch_gene(symbol)


@st.cache_data(show_spinner=False)
def load_structure(symbol):
    return fetch_structure(symbol)


@st.cache_data(show_spinner=False)
def compare_tables(symbols):
    genes = [load_gene(s) for s in symbols]
    return feature_table(genes), signature_table(genes), window_table(genes)


def show(fig):
    """Draw a matplotlib figure, then free it (Streamlit reruns would pile them up)."""
    st.pyplot(fig)
    plt.close(fig)


def x_axis(length):
    return alt.X("start:Q", title="position (bp)", scale=alt.Scale(domain=[0, length]))


def gc_chart(seq, title):
    """Sliding-window GC with CpG islands as orange bands. Scroll to zoom, drag to pan."""
    window = 200 if len(seq) >= 400 else max(10, len(seq) // 4)
    step = max(1, len(seq) // 2000)  # about 2000 points whatever the length
    pos, gc = sliding_gc(seq, window, step)
    line_df = pd.DataFrame({"start": pos, "GC %": gc})
    isl_df = pd.DataFrame([{"start": s + 1, "end": e, "length (bp)": e - s, "GC %": round(gc_content(seq[s:e]), 1)}
                           for s, e in cpg_islands(seq)], columns=["start", "end", "length (bp)", "GC %"])
    bands = alt.Chart(isl_df).mark_rect(color="orange", opacity=0.3).encode(
        x_axis(len(seq)), x2="end:Q", tooltip=["start:Q", "end:Q", "length (bp):Q", "GC %:Q"])
    line = alt.Chart(line_df).mark_line(strokeWidth=1).encode(
        x_axis(len(seq)), alt.Y("GC %:Q", title=f"GC % ({window} bp window)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("start:Q", title="position"), alt.Tooltip("GC %:Q", format=".1f")],
    ).add_params(alt.selection_interval(bind="scales", encodings=["x"]))
    return alt.layer(bands, line).properties(
        height=220, title=f"{title}: GC content, CpG islands in orange (scroll to zoom, drag to pan)")


def orf_chart(seq, orfs, title, highlight=None):
    """ORFs as bars, one row per strand and frame; highlight=(start, end) is colored as the annotated CDS."""
    df = pd.DataFrame([{"row": f"{o['strand']} frame {o['frame'] + 1}", "start": o["start"] + 1, "end": o["end"],
                        "length (aa)": len(o["protein"]),
                        "kind": "annotated CDS" if highlight == (o["start"], o["end"]) else "ORF"} for o in orfs],
                      columns=["row", "start", "end", "length (aa)", "kind"])
    return alt.Chart(df).mark_bar().encode(
        x_axis(len(seq)), x2="end:Q",
        y=alt.Y("row:N", title=None, scale=alt.Scale(domain=ROWS), axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["ORF", "annotated CDS"],
                                                              range=["#4c78a8", "#e45756"])),
        tooltip=["row:N", "start:Q", "end:Q", "length (aa):Q"],
    ).add_params(alt.selection_interval(bind="scales", encodings=["x"])).properties(height=240, title=title)


def protein_viewer(pdb, plddt, key_residues, focus):
    """3Dmol.js view: cartoon colored by AlphaFold confidence, active-site residues as magenta sticks."""
    view = py3Dmol.view(width="100%", height=520)
    view.addModel(pdb.replace("<", ""), "pdb")  # py3Dmol doesn't escape model text; real PDB files never contain "<"
    for lo, hi, color, _ in PLDDT_BINS:
        residues = [r for r, p in plddt.items() if lo <= p < hi]
        if residues:
            view.setStyle({"resi": residues}, {"cartoon": {"color": color}})
    if key_residues:
        view.addStyle({"resi": key_residues}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.3}})
    view.zoomTo({"resi": key_residues} if focus and key_residues else {})
    st.iframe(view._make_html(), height=540)


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

tab_genes, tab_one, tab_prot, tab_cmp, tab_own = st.tabs(
    ["Genes", "Single gene", "Protein structure", "Compare genes", "Your sequence"])

# ---------------- tab 1: overview ----------------
with tab_genes:
    st.subheader("Loaded genes")
    st.dataframe(pd.DataFrame([{
        "gene": g["symbol"], "mRNA": g["mrna_id"], "location": f"{g['chrom']}:{g['start']:,}-{g['end']:,} ({g['strand']})",
        "gene region (bp)": len(g["region"]), "mRNA (bp)": len(g["mrna"]), "protein (aa)": len(g["protein"]),
    } for g in genes]), hide_index=True)
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

    st.altair_chart(gc_chart(region, f"{g['symbol']} gene region"), width="stretch")

    mrna_orfs = find_orfs(mrna, min_aa)
    longest_is_cds = bool(mrna_orfs) and (mrna_orfs[0]["start"], mrna_orfs[0]["end"]) == cds
    st.altair_chart(orf_chart(mrna, mrna_orfs, f"{g['mrna_id']} mRNA: ORFs ≥ {min_aa} aa", cds), width="stretch")
    if longest_is_cds:
        same = "matches" if mrna_orfs[0]["protein"] == g["protein"] else "does NOT match"
        st.success(f"The longest ORF ({mrna_orfs[0]['start'] + 1}..{mrna_orfs[0]['end']}) is exactly the "
                   f"annotated CDS, and its translated protein {same} NCBI's.")
    else:
        st.warning("The longest ORF is not the annotated CDS for this gene.")
    with st.expander(f"ORF table ({len(mrna_orfs)} ORFs in the mRNA)"):
        st.dataframe(orf_frame(mrna_orfs), hide_index=True)
    with st.expander("ORFs in the gene region instead (mostly chance hits in introns)"):
        region_orfs = find_orfs(region, min_aa)
        st.altair_chart(orf_chart(region, region_orfs, f"{g['symbol']} gene region: ORFs ≥ {min_aa} aa"),
                        width="stretch")
        st.caption(f"{len(region_orfs)} ORFs; the longest is {len(region_orfs[0]['protein']) if region_orfs else 0} aa, "
                   f"while the real protein is {len(g['protein'])} aa split across exons.")
    with st.expander("CpG islands"):
        st.dataframe(pd.DataFrame([{"start": s + 1, "end": e, "length": e - s, "GC %": round(gc_content(region[s:e]), 1)}
                                   for s, e in islands]), hide_index=True)

# ---------------- tab 3: protein structure ----------------
with tab_prot:
    g = load_gene(st.selectbox("Gene", symbols, key="protein_gene"))
    try:
        with st.spinner("Fetching the AlphaFold model (first time only)..."):
            info = load_structure(g["symbol"])
    except Exception as e:
        st.warning(f"No structure available for {g['symbol']}: {e}")
    else:
        plddt = residue_plddt(info["pdb"])
        motifs = find_motif(info["sequence"])
        # the four residues that do the work: H at 1, E at 2, H at 5, H at 11 of each motif
        key = sorted({s + k for s, _ in motifs for k in (0, 1, 4, 10)})
        c = st.columns(4)
        c[0].metric("UniProt", info["uniprot"])
        c[1].metric("Length", f"{len(info['sequence'])} aa")
        c[2].metric("Mean pLDDT", f"{info['mean_plddt']:.0f}")
        c[3].metric("Zinc active site", ", ".join(f"{s}–{e}" for s, e in motifs) or "none")

        same = info["sequence"] == g["protein"]
        st.markdown(("✅ The AlphaFold sequence is identical to the protein translated from the mRNA's ORF, "
                     "so this is the same molecule the Single gene tab found.") if same else
                    "⚠️ The AlphaFold sequence (UniProt canonical isoform) differs from the RefSeq protein.")
        focus = st.toggle("Zoom to the active site", value=bool(motifs), disabled=not motifs)
        protein_viewer(info["pdb"], plddt, key, focus)
        st.markdown("  ".join(f"<span style='background:{col};padding:2px 8px;border-radius:3px'>&nbsp;</span> {label}"
                              for _, _, col, label in PLDDT_BINS) +
                    "  <span style='color:#c0c'>●</span> active-site H/E side chains", unsafe_allow_html=True)
        for s, e in motifs:
            st.caption(f"Motif {info['sequence'][s - 1:e]} at residues {s}–{e} (pattern HExxH.....H: three histidines "
                       f"hold the zinc, E is the catalytic base). Mean pLDDT there: "
                       f"{sum(plddt[r] for r in range(s, e + 1)) / (e - s + 1):.0f}.")
        if not motifs:
            st.caption("No HExxH.....H zinc motif: this protein is not a zinc metalloprotease.")

        pl = pd.DataFrame({"residue": list(plddt), "pLDDT": list(plddt.values())})
        line = alt.Chart(pl).mark_line(strokeWidth=1).encode(
            alt.X("residue:Q", scale=alt.Scale(domain=[1, len(plddt)])),
            alt.Y("pLDDT:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["residue:Q", "pLDDT:Q"]).add_params(alt.selection_interval(bind="scales", encodings=["x"]))
        marks = alt.Chart(pd.DataFrame({"residue": key})).mark_rule(color="#c0c").encode(x="residue:Q")
        st.altair_chart(alt.layer(line, marks).properties(
            height=180, title="AlphaFold confidence along the protein (magenta = active-site residues)"), width="stretch")

# ---------------- tab 4: compare ----------------
with tab_cmp:
    if len(genes) < 3:
        st.info("Load at least 3 genes to compare them.")
    else:
        features, signatures, windows = compare_tables(symbols)
        show(plot_feature_heatmap(features))
        st.download_button("Download feature table (CSV)", features.to_csv(), "gene_features.csv")

        left, right = st.columns(2)
        with left:
            show(plot_distance(signature_distance(signatures)))
            gc_r = pca(signatures)[0]["PC1"].corr(features["gc_region_%"])
            st.caption("Karlin's genomic signature: each dinucleotide's frequency divided by what its bases' "
                       f"frequencies predict, so overall GC content is factored out (signature PC1 vs GC: r = {gc_r:.2f}). "
                       "Pairs that stay close here share dinucleotide preferences, not just GC.")
        with right:
            show(plot_pca(*pca(features), "PCA of gene features"))
            st.caption("With only a handful of genes, read this as a picture of the heatmap above, not as clusters.")

        metric = st.radio("Window metric", ["gc_%", "cpg_oe"], horizontal=True,
                          format_func={"gc_%": "GC %", "cpg_oe": "CpG observed/expected"}.get)
        show(plot_windows(windows, metric))
        stat, p = kruskal_by_gene(windows, metric)
        st.caption(f"Kruskal-Wallis across genes on 1 kb windows: H = {stat:.1f}, p = {p:.2g}. Exploratory only: "
                   "neighboring windows are correlated, so this p-value is too optimistic.")

# ---------------- tab 5: user's own sequence ----------------
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
        st.altair_chart(gc_chart(seq, name), width="stretch")
        orfs = find_orfs(seq, min_aa)
        st.altair_chart(orf_chart(seq, orfs, f"{name}: ORFs ≥ {min_aa} aa"), width="stretch")
        st.dataframe(orf_frame(orfs), hide_index=True)
        with st.expander("Reverse complement and translation (frame +1)"):
            st.code(reverse_complement(seq), language=None, wrap_lines=True)
            st.code(translate(seq), language=None, wrap_lines=True)
