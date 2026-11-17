"""Every visual in the Streamlit app: Altair charts (zoom, pan, hover) and the 3Dmol protein viewer.

The matplotlib plots in compare.py are only for the README figures.
"""
import altair as alt
import pandas as pd
import py3Dmol

from compare import FEATURES, WINDOW_METRICS, kruskal_by_gene
from sequence import cpg_islands, gc_content, sliding_gc

INK, BLUE, CYAN, YELLOW, ORANGE, PALE = "#12264A", "#0053D6", "#65CBF3", "#FFDB13", "#FF7D45", "#EAF0F8"
MAGENTA = "#C026D3"
ROWS = [f"{s} frame {f}" for s in "+-" for f in (1, 2, 3)]
# AlphaFold's own confidence colors
PLDDT_BINS = [(90, 101, BLUE, "very high (≥ 90)"), (70, 90, CYAN, "confident (70–90)"),
              (50, 70, YELLOW, "low (50–70)"), (0, 50, ORANGE, "very low (< 50)")]
THREE_LETTER = {"H": "His", "E": "Glu"}


def _zoom():
    return alt.selection_interval(bind="scales", encodings=["x"])


def _x(length, field="start"):
    return alt.X(f"{field}:Q", title="position (bp)", scale=alt.Scale(domain=[0, length]), axis=alt.Axis(format=",d"))


def _fmt(v):
    """Readable number for a heatmap cell: no scientific notation."""
    if float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


# ---------------- single sequence tracks ----------------

def gc_chart(seq, title):
    """Sliding-window GC with CpG islands as orange bands."""
    window = 200 if len(seq) >= 400 else max(10, len(seq) // 4)
    step = max(1, len(seq) // 2000)  # about 2000 points whatever the length
    pos, gc = sliding_gc(seq, window, step)
    line_df = pd.DataFrame({"start": pos, "GC %": gc})
    isl_df = pd.DataFrame([{"start": s + 1, "end": e, "length (bp)": e - s, "GC %": round(gc_content(seq[s:e]), 1)}
                           for s, e in cpg_islands(seq)], columns=["start", "end", "length (bp)", "GC %"])
    bands = alt.Chart(isl_df).mark_rect(color=ORANGE, opacity=0.28).encode(
        _x(len(seq)), x2="end:Q",
        tooltip=[alt.Tooltip("start:Q", title="CpG island start", format=","), alt.Tooltip("end:Q", format=","),
                 "length (bp):Q", "GC %:Q"])
    line = alt.Chart(line_df).mark_line(strokeWidth=1.2, color=BLUE).encode(
        _x(len(seq)), alt.Y("GC %:Q", title=f"GC % ({window} bp window)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("start:Q", title="position", format=","), alt.Tooltip("GC %:Q", format=".1f")],
    ).add_params(_zoom())
    return alt.layer(bands, line).properties(height=230, title=alt.Title(
        title, subtitle="GC content along the sequence; orange bands are CpG islands. Scroll to zoom, drag to pan."))


def orf_chart(seq, orfs, title, highlight=None):
    """ORFs as bars, one row per strand and frame; highlight=(start, end) is colored as the annotated CDS."""
    df = pd.DataFrame([{"row": f"{o['strand']} frame {o['frame'] + 1}", "start": o["start"] + 1, "end": o["end"],
                        "length (aa)": len(o["protein"]),
                        "kind": "annotated coding sequence" if highlight == (o["start"], o["end"]) else "ORF"}
                       for o in orfs], columns=["row", "start", "end", "length (aa)", "kind"])
    return alt.Chart(df).mark_bar(cornerRadius=2).encode(
        _x(len(seq)), x2="end:Q",
        y=alt.Y("row:N", title=None, scale=alt.Scale(domain=ROWS), axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["ORF", "annotated coding sequence"],
                                                              range=["#7FA6E0", ORANGE]),
                        legend=alt.Legend(orient="top", labelLimit=260)),
        tooltip=[alt.Tooltip("row:N", title="strand and frame"), alt.Tooltip("start:Q", format=","),
                 alt.Tooltip("end:Q", format=","), "length (aa):Q"],
    ).add_params(_zoom()).properties(height=alt.Step(26), title=alt.Title(
        title, subtitle="Each bar runs from an ATG to the next stop codon, in all six reading frames."))


# ---------------- protein ----------------

def plddt_chart(plddt, key_residues):
    """AlphaFold confidence along the protein over its four colored bands, active-site residues marked."""
    bands = pd.DataFrame([{"lo": lo, "hi": min(hi, 100), "label": label, "color": c} for lo, hi, c, label in PLDDT_BINS])
    back = alt.Chart(bands).mark_rect(opacity=0.18).encode(
        y="lo:Q", y2="hi:Q", color=alt.Color("color:N", scale=None), tooltip=[alt.Tooltip("label:N", title="band")])
    pl = pd.DataFrame({"residue": list(plddt), "pLDDT": list(plddt.values())})
    line = alt.Chart(pl).mark_line(strokeWidth=1.2, color=INK).encode(
        alt.X("residue:Q", scale=alt.Scale(domain=[1, len(plddt)])),
        alt.Y("pLDDT:Q", scale=alt.Scale(domain=[0, 100]), title="confidence (pLDDT)"),
        tooltip=["residue:Q", alt.Tooltip("pLDDT:Q", format=".0f")]).add_params(_zoom())
    marks = alt.Chart(pd.DataFrame({"residue": key_residues})).mark_rule(color=MAGENTA, strokeWidth=2).encode(
        x="residue:Q", tooltip=[alt.Tooltip("residue:Q", title="active-site residue")])
    return alt.layer(back, line, marks).properties(height=200, title=alt.Title(
        "AlphaFold confidence along the protein",
        subtitle="Magenta lines mark the active-site residues. Low stretches are usually flexible or disordered."))


def protein_viewer_html(pdb, plddt, sequence, key_residues, style="Cartoon", color="AlphaFold confidence",
                        spin=False, focus=True, labels=True, dark=False, height=560):
    """3Dmol.js page for the model. Hovering a residue shows its name, number and confidence."""
    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb.replace("<", ""), "pdb")  # py3Dmol doesn't escape model text; real PDB files never contain "<"
    view.setBackgroundColor("#0E1A33" if dark else "#FFFFFF")
    if style == "Sticks":
        view.setStyle({}, {"stick": {"radius": 0.18, "colorscheme": "grayCarbon"}})
    elif color == "AlphaFold confidence":
        for lo, hi, c, _ in PLDDT_BINS:
            residues = [r for r, p in plddt.items() if lo <= p < hi]
            if residues:
                view.setStyle({"resi": residues}, {"cartoon": {"color": c, "thickness": 0.5}})
    elif color == "Rainbow, N to C":
        view.setStyle({}, {"cartoon": {"color": "spectrum", "thickness": 0.5}})
    else:  # secondary structure
        view.setStyle({}, {"cartoon": {"colorscheme": "ssPyMOL", "thickness": 0.5}})
    if style == "Cartoon + surface":
        view.addSurface(py3Dmol.VDW, {"opacity": 0.55, "color": "#DCE3EE" if not dark else "#5F7391"})
    if key_residues:
        view.addStyle({"resi": key_residues}, {"stick": {"color": MAGENTA, "radius": 0.32}})
        if labels:
            for r in key_residues:
                view.addLabel(f"{THREE_LETTER.get(sequence[r - 1], sequence[r - 1])}{r}",
                              {"fontSize": 12, "fontColor": "white", "backgroundColor": MAGENTA,
                               "backgroundOpacity": 0.9, "inFront": True}, {"resi": r, "atom": "CA"})
    view.setHoverDuration(150)
    view.setHoverable({}, True,
                      """function(atom, viewer) {
                           if (!atom.label) {
                             atom.label = viewer.addLabel(atom.resn + " " + atom.resi + "  pLDDT " + atom.b.toFixed(0),
                               {position: atom, backgroundColor: "BG", fontColor: "FG", fontSize: 12,
                                backgroundOpacity: 0.92, inFront: true});
                           }
                         }""".replace("BG", "#FFFFFF" if dark else INK).replace("FG", INK if dark else "#FFFFFF"),
                      """function(atom, viewer) {
                           if (atom.label) { viewer.removeLabel(atom.label); delete atom.label; }
                         }""")
    if focus and key_residues:
        view.zoomTo({"resi": key_residues})
        view.zoom(0.55)  # back off so the fold around the site stays in view
    else:
        view.zoomTo()
    if spin:
        view.spin("y", 0.6)
    return "<style>html,body{margin:0;overflow:hidden}</style>" + view._make_html()


# ---------------- comparison ----------------

def feature_heatmap(features):
    """Features (rows) x genes (columns). Color = how unusual a gene is for that feature (z-score); text = the real
    value. Features run down the side so their full names read horizontally."""
    z = (features - features.mean()) / features.std().replace(0, 1)
    rows = []
    for gene in features.index:
        for col in features.columns:
            label, meaning = FEATURES[col]
            v = float(features.at[gene, col])
            shown = ("yes" if v else "no") if col == "island_at_5p" else _fmt(v)
            rows.append({"gene": gene, "feature": label, "meaning": meaning, "value": v, "shown": shown,
                         "z": round(float(z.at[gene, col]), 2)})
    df = pd.DataFrame(rows)
    order = [FEATURES[c][0] for c in features.columns]
    base = alt.Chart(df).encode(
        y=alt.Y("feature:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=280, labelFontSize=12, labelOverlap=False, labelPadding=8)),
        x=alt.X("gene:N", sort=list(features.index), title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=12, labelFontWeight=600, labelOverlap=False)))
    cells = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        color=alt.Color("z:Q", title="vs other genes (z)",
                        scale=alt.Scale(domain=[-2.5, 0, 2.5], range=[BLUE, "#F7F9FC", ORANGE], clamp=True,
                                        interpolate="rgb"), legend=alt.Legend(orient="bottom", direction="horizontal",
                                                                             gradientLength=220)),
        tooltip=[alt.Tooltip("gene:N"), alt.Tooltip("feature:N"), alt.Tooltip("shown:N", title="value"),
                 alt.Tooltip("z:Q", title="z-score"), alt.Tooltip("meaning:N", title="what it means")])
    text = base.mark_text(fontSize=12).encode(
        text="shown:N", color=alt.condition("abs(datum.z) > 1.6", alt.value("white"), alt.value(INK)))
    return (cells + text).properties(height=alt.Step(30), title=alt.Title(
        "Gene features at a glance",
        subtitle="Each cell shows the real value. Color shows how unusual it is next to the other genes: "
                 "orange is high, blue is low. Hover a cell for what the feature means."))


def distance_heatmap(dist):
    df = dist.reset_index(names="gene a").melt(id_vars="gene a", var_name="gene b", value_name="delta")
    order = list(dist.index)
    base = alt.Chart(df).encode(
        x=alt.X("gene b:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
        y=alt.Y("gene a:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)))
    cells = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        color=alt.Color("delta:Q", title="δ* × 1000", scale=alt.Scale(range=[BLUE, PALE], interpolate="rgb"),
                        legend=alt.Legend(
            orient="bottom", direction="horizontal")),
        tooltip=["gene a:N", "gene b:N", alt.Tooltip("delta:Q", title="δ* × 1000", format=".0f")])
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("delta:Q", format=".0f"),
        color=alt.condition(f"datum.delta < {float(df.delta.max()) / 2}", alt.value("white"), alt.value(INK)))
    return (cells + text).properties(height=alt.Step(34), title=alt.Title(
        "Genomic signature distance", subtitle="Lower = more similar dinucleotide preferences, with GC content factored out."))


def pca_scatter(scores, var, title, subtitle):
    df = scores.reset_index(names="gene")
    base = alt.Chart(df).encode(
        x=alt.X("PC1:Q", title=f"PC1 ({var[0]:.0f}% of variance)", scale=alt.Scale(zero=False, padding=30)),
        y=alt.Y("PC2:Q", title=f"PC2 ({var[1]:.0f}% of variance)", scale=alt.Scale(zero=False, padding=30)))
    dots = base.mark_circle(size=140, color=BLUE, opacity=0.9).encode(
        tooltip=["gene:N", alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")])
    names = base.mark_text(align="left", dx=9, dy=-8, fontSize=12, color=INK).encode(text="gene:N")
    return (dots + names).properties(height=320, title=alt.Title(title, subtitle=subtitle))


def window_boxplot(windows, metric):
    order = list(windows.groupby("gene")[metric].median().sort_values().index)
    stat, p = kruskal_by_gene(windows, metric)
    df = windows.rename(columns={metric: "value"})
    return alt.Chart(df).mark_boxplot(size=26, color=BLUE, median={"color": ORANGE, "strokeWidth": 2},
                                      outliers={"color": BLUE, "opacity": 0.5}).encode(
        x=alt.X("gene:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12, labelOverlap=False)),
        y=alt.Y("value:Q", title=WINDOW_METRICS[metric], scale=alt.Scale(zero=False)),
    ).properties(height=300, title=alt.Title(
        f"{WINDOW_METRICS[metric]} in 1 kb windows",
        subtitle=f"Genes sorted by median (orange line). Kruskal-Wallis H = {stat:.1f}, p = {p:.2g}; exploratory, "
                 "because neighboring windows are correlated."))
