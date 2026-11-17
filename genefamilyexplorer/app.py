"""Gene Family Explorer: Streamlit interface. Analysis lives in sequence.py / compare.py / ncbi.py /
structure.py; every chart and the 3D viewer are in charts.py.

Run: streamlit run app.py
"""
import html
import re

import altair as alt
import pandas as pd
import streamlit as st

from charts import (PLDDT_BINS, distance_heatmap, feature_heatmap, gc_chart, orf_chart, pca_scatter, plddt_chart,
                    protein_viewer_html, window_boxplot)
from compare import (DEFAULT_GENES, FEATURES, WINDOW_METRICS, feature_table, pca, signature_distance,
                     signature_table, window_table)
from ncbi import fetch_gene, search_genes
from sequence import cpg_islands, find_orfs, gc_content, parse_fasta, reverse_complement, translate
from structure import fetch_structure, find_motif, residue_plddt

st.set_page_config(page_title="Gene Family Explorer", page_icon=":material/genetics:", layout="wide")

GENE_SETS = {
    "ADAMTS proteases, with GAPDH for contrast": DEFAULT_GENES,
    "Matrix metalloproteinases (MMPs), with GAPDH": ["MMP1", "MMP2", "MMP3", "MMP9", "MMP13", "MMP14", "GAPDH"],
    "Housekeeping genes": ["ACTB", "GAPDH", "TUBB", "B2M", "HPRT1", "PGK1"],
    "Cancer genes": ["TP53", "BRCA1", "KRAS", "EGFR", "MYC", "PTEN"],
}
NCBI = "https://www.ncbi.nlm.nih.gov"

st.html("""<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@600&display=swap");
.gfe-hero{background:linear-gradient(105deg,#0A2463 0%,#0053D6 38%,#3FA9E8 64%,#FFDB13 86%,#FF7D45 100%);
  color:#fff;border-radius:10px;padding:28px 32px 24px;margin:0 0 8px}
.gfe-hero h1{font-family:'IBM Plex Sans Condensed',sans-serif;font-weight:600;font-size:clamp(2rem,3.6vw,3.1rem);
  line-height:1.05;letter-spacing:-.01em;margin:0 0 6px;padding:0;color:#fff}
.gfe-hero p{font-size:1.05rem;line-height:1.5;max-width:62ch;margin:0 0 18px;color:rgba(255,255,255,.92)}
.gfe-hero .gfe-focus{font-size:.92rem;margin:0 0 8px;color:#fff}
.gfe-hero .gfe-focus b{font-weight:600}
.gfe-hero ol.gfe-chain{display:flex;flex-wrap:wrap;gap:8px 22px;list-style:none;margin:0 !important;padding:0 !important}
.gfe-chain li{background:rgba(255,255,255,.94);color:#12264A;border-radius:6px;padding:7px 12px;
  display:flex;flex-direction:column;flex:1 1 6.5rem;max-width:13rem;position:relative}
.gfe-chain li span{font-size:.75rem;color:#5F7391;white-space:nowrap}
.gfe-chain li strong{font-size:1rem;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
.gfe-chain li+li::before{content:"→";position:absolute;left:-11px;top:50%;transform:translate(-50%,-50%);
  color:#fff;font-weight:600;text-shadow:0 0 4px rgba(10,36,99,.6)}
.gfe-chain li.gfe-missing strong{color:#8A99B3;font-weight:500}
.gfe-chart-title{font-family:'IBM Plex Sans Condensed',sans-serif;font-weight:600;font-size:1.15rem;
  color:#12264A;margin:14px 0 0}
.gfe-chart-sub{font-size:.88rem;color:#5F7391;line-height:1.45;max-width:90ch;margin:2px 0 4px}
.gfe-legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.85rem;color:#12264A;margin:4px 0 0}
.gfe-legend i{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
@media (max-width:640px){.gfe-hero{padding:20px}.gfe-chain{gap:8px 22px}.gfe-chain li{flex-basis:calc(50% - 11px);max-width:none}.gfe-chain li::before{display:none}}
</style>""")


@st.cache_data(show_spinner=False)
def load_gene(symbol):
    return fetch_gene(symbol)


@st.cache_data(show_spinner=False)
def load_structure(symbol):
    return fetch_structure(symbol)


@st.cache_data(show_spinner=False, ttl=3600)
def find_on_ncbi(query):
    return search_genes(query)


@st.cache_data(show_spinner=False)
def compare_tables(symbols):
    genes = [load_gene(s) for s in symbols]
    return feature_table(genes), signature_table(genes), window_table(genes)


def show_chart(chart):
    """Render the chart's title and subtitle as page text (in the app's type), then the chart without them."""
    title = chart.title
    if isinstance(title, alt.Title):
        sub = title.subtitle if title.subtitle is not alt.Undefined else ""
        st.html(f'<div class="gfe-chart-title">{html.escape(str(title.text))}</div>'
                + (f'<div class="gfe-chart-sub">{html.escape(str(sub))}</div>' if sub else ""))
        chart = chart.copy()
        chart.title = alt.Undefined
    st.altair_chart(chart, width="stretch")


def fasta(name, seq, width=70):
    return f">{name}\n" + "\n".join(seq[i:i + width] for i in range(0, len(seq), width)) + "\n"


def orf_frame(orfs):
    return pd.DataFrame([{"strand": o["strand"], "frame": o["frame"] + 1, "start": o["start"] + 1,
                          "end": o["end"], "length (aa)": len(o["protein"]), "protein": o["protein"]}
                         for o in orfs], columns=["strand", "frame", "start", "end", "length (aa)", "protein"])


def parse_symbols(text):
    return list(dict.fromkeys(s.upper() for s in re.split(r"[,\s]+", text) if s))


# ---------------- sidebar: choose genes ----------------
if "symbols_text" not in st.session_state:
    st.session_state.symbols_text = ", ".join(DEFAULT_GENES)


def use_gene_set():
    st.session_state.symbols_text = ", ".join(GENE_SETS[st.session_state.gene_set])


def add_found_genes():
    current = parse_symbols(st.session_state.symbols_text)
    picked = [p.split(" ")[0] for p in st.session_state.found]
    st.session_state.symbols_text = ", ".join(dict.fromkeys(current + picked))
    st.session_state.found = []


with st.sidebar:
    st.subheader("Genes to explore")
    st.selectbox("Start from a gene set", list(GENE_SETS), key="gene_set", on_change=use_gene_set,
                 help="Replaces the gene list below. You can still edit it afterwards.")
    query = st.text_input("Search NCBI Gene", placeholder="MMP, ADAMTS*, collagen, tumor protein…",
                          help="One word matches gene symbols and prefixes; several words match gene and protein "
                               "names. Human protein-coding genes only, most-studied first.")
    if query.strip():
        with st.spinner("Searching NCBI Gene…"):
            try:
                hits = find_on_ncbi(query)
            except Exception as e:
                hits = []
                st.warning(f"NCBI search failed: {e}. Try again in a moment.")
        if hits:
            st.multiselect(f"{len(hits)} matches: pick genes to add", key="found",
                           options=[f"{h['symbol']} ({h['band']}): {h['name']}" for h in hits])
            st.button("Add to gene list", on_click=add_found_genes, disabled=not st.session_state.get("found"),
                      icon=":material/add:", width="stretch")
        else:
            st.caption("No human protein-coding genes match. Try a symbol prefix like MMP or a name like collagen.")
    st.text_area("Gene list", key="symbols_text", height=110,
                 help="Human gene symbols separated by commas or spaces. New genes download from NCBI on first use.")

symbols = parse_symbols(st.session_state.symbols_text)
genes, failed = [], []
with st.spinner("Fetching genes from NCBI (first time only)…"):
    for s in symbols:
        try:
            genes.append(load_gene(s))
        except Exception as e:  # one bad symbol shouldn't break the whole page
            failed.append(f"{s}: {e}")
symbols = tuple(g["symbol"] for g in genes)
by_symbol = {g["symbol"]: g for g in genes}

with st.sidebar:
    for msg in failed:
        st.warning(msg, icon=":material/error:")
    if genes:
        if st.session_state.get("focus") not in symbols:
            st.session_state.focus = symbols[0]
        st.selectbox("Focus gene", symbols, key="focus",
                     format_func=lambda s: f"{s}: {by_symbol[s]['name']}",
                     help="The gene shown in the header, the Gene tab and the Protein tab.")
    min_aa = st.slider("Shortest ORF to show (amino acids)", 30, 300, 100, 10,
                       help="ORFs shorter than this are hidden. Random DNA has many short ORFs; real proteins "
                            "are usually 100+ aa.")
    with st.expander("About this app", icon=":material/info:"):
        st.markdown(
            "Follows each gene through the chain **chromosome → gene → mRNA → ORF → protein → structure**, "
            "then compares genes on composition and structure.\n\n"
            f"Data: [NCBI Gene]({NCBI}/gene/) and RefSeq for sequences, [UniProt](https://www.uniprot.org) and the "
            "[AlphaFold Database](https://alphafold.ebi.ac.uk) for structures. Downloads are cached, so a gene only "
            "loads slowly the first time.\n\n"
            "The analysis (ORFs, CpG islands, genomic signatures) is hand-written Python, tested against Biopython. "
            "[Source on GitHub](https://github.com/tkthang06/Projects/tree/main/genefamilyexplorer)")

if not genes:
    st.html('<div class="gfe-hero"><h1>Gene Family Explorer</h1>'
            "<p>Add at least one human gene symbol in the sidebar to begin.</p></div>")
    st.stop()

# ---------------- header: the focus gene's chain from chromosome to structure ----------------
focus = by_symbol[st.session_state.focus]
try:
    focus_struct = load_structure(focus["symbol"])
except Exception:
    focus_struct = None
chain = [("Chromosome band", focus["band"] or focus["chrom"]),
         ("Gene", f"{len(focus['region']):,} bp"),
         ("mRNA", f"{len(focus['mrna']):,} nt"),
         ("Coding sequence", f"{focus['cds_start'] + 1:,}–{focus['cds_end']:,}"),
         ("Protein", f"{len(focus['protein']):,} aa"),
         ("Structure", f"AlphaFold {focus_struct['uniprot']}" if focus_struct else "no model")]
st.html(
    '<div class="gfe-hero"><h1>Gene Family Explorer</h1>'
    "<p>Follow a human gene from its place on the chromosome to its folded protein, then line it up "
    "against its relatives.</p>"
    f'<div class="gfe-focus">Following <b>{html.escape(focus["symbol"])}</b>, {html.escape(focus["name"])}</div>'
    '<ol class="gfe-chain">' + "".join(
        f'<li class="{"gfe-missing" if v == "no model" else ""}"><span>{k}</span><strong>{html.escape(v)}</strong></li>'
        for k, v in chain) + "</ol></div>")

tab_overview, tab_gene, tab_prot, tab_cmp, tab_own = st.tabs([
    ":material/table_rows: Overview", ":material/genetics: Gene", ":material/deployed_code: Protein structure",
    ":material/compare_arrows: Compare genes", ":material/edit_note: Your sequence"])

# ---------------- tab 1: overview ----------------
with tab_overview:
    st.subheader(f"{len(genes)} genes loaded")
    st.dataframe(pd.DataFrame([{
        "gene": g["symbol"], "name": g["name"], "band": g["band"],
        "location": f"{g['chrom']}:{g['start']:,}-{g['end']:,} ({g['strand']})",
        "gene (bp)": len(g["region"]), "mRNA (nt)": len(g["mrna"]), "protein (aa)": len(g["protein"]),
        "NCBI Gene": f"{NCBI}/gene/{g['gene_id']}", "RefSeq mRNA": f"{NCBI}/nuccore/{g['mrna_id']}",
    } for g in genes]), hide_index=True, column_config={
        "name": st.column_config.TextColumn(width="large"),
        "gene (bp)": st.column_config.NumberColumn(format="localized"),
        "mRNA (nt)": st.column_config.NumberColumn(format="localized"),
        "NCBI Gene": st.column_config.LinkColumn(display_text="Gene page"),
        "RefSeq mRNA": st.column_config.LinkColumn(display_text=r"nuccore/(.*)"),
    })
    with st.expander("How to read this app", icon=":material/menu_book:"):
        st.markdown(
            "- **Gene** is the stretch of chromosome that holds the gene: exons plus the much longer introns.\n"
            "- **mRNA** is what's left after splicing removes the introns. Its coding sequence runs from a start "
            "codon to a stop codon; this app finds it again from scratch as the longest open reading frame (ORF).\n"
            "- **Protein** is the ORF translated three bases at a time. **Structure** is AlphaFold's predicted fold of "
            "that same protein, checked letter for letter.\n"
            "- **Compare genes** lines genes up on these measurements. It's most interesting within one family "
            "(ADAMTS or MMP) plus one unrelated gene for contrast.")

# ---------------- tab 2: one gene ----------------
with tab_gene:
    g = focus
    region, mrna = g["region"], g["mrna"]
    cds = (g["cds_start"], g["cds_end"])
    islands = cpg_islands(region)
    st.subheader(f"{g['symbol']}: {g['name']}")
    links = st.columns([1, 1.3, 2.7])
    links[0].link_button("NCBI Gene page", f"{NCBI}/gene/{g['gene_id']}", icon=":material/open_in_new:",
                         width="stretch")
    links[1].link_button(f"mRNA {g['mrna_id']}", f"{NCBI}/nuccore/{g['mrna_id']}", icon=":material/open_in_new:",
                         width="stretch")
    c = st.columns(5)
    c[0].metric("Gene length", f"{len(region):,} bp", help="Exons plus introns on the chromosome.")
    c[1].metric("GC content", f"{gc_content(region):.1f}%", help="Share of G and C bases across the whole gene.")
    c[2].metric("CpG islands", len(islands), help="CpG-rich stretches, often marking promoters.")
    c[3].metric("Protein", f"{len(g['protein']):,} aa")
    c[4].metric("Coding share", f"{(cds[1] - cds[0]) / len(region) * 100:.1f}%",
                help="How much of the gene ends up as protein code.")

    show_chart(gc_chart(region, f"{g['symbol']} gene, {g['band']}"))

    mrna_orfs = find_orfs(mrna, min_aa)
    show_chart(orf_chart(mrna, mrna_orfs, f"{g['mrna_id']} mRNA: ORFs of {min_aa}+ aa", cds))
    start_codon = mrna[cds[0]:cds[0] + 3]
    if mrna_orfs and (mrna_orfs[0]["start"], mrna_orfs[0]["end"]) == cds:
        same = "matches" if mrna_orfs[0]["protein"] == g["protein"] else "does not match"
        st.success(f"The longest ORF ({mrna_orfs[0]['start'] + 1:,}–{mrna_orfs[0]['end']:,}) is exactly the "
                   f"annotated coding sequence, and its translated protein {same} NCBI's.",
                   icon=":material/check_circle:")
    elif start_codon != "ATG":
        st.info(f"NCBI's coding sequence for {g['symbol']} starts with {start_codon}, not ATG. A few proteins start "
                "at a non-ATG codon; this ORF finder only uses ATG, so it can't recover this one.",
                icon=":material/info:")
    else:
        st.warning("The longest ORF is not the annotated coding sequence for this gene.", icon=":material/warning:")

    d = st.columns(3)
    d[0].download_button("mRNA (FASTA)", fasta(g["mrna_id"], mrna), f"{g['symbol']}_mRNA.fasta",
                         icon=":material/download:", width="stretch")
    d[1].download_button("Protein (FASTA)", fasta(f"{g['symbol']} protein", g["protein"]),
                         f"{g['symbol']}_protein.fasta", icon=":material/download:", width="stretch")
    d[2].download_button("ORF table (CSV)", orf_frame(mrna_orfs).to_csv(index=False), f"{g['symbol']}_ORFs.csv",
                         icon=":material/download:", width="stretch")
    with st.expander(f"ORF table: {len(mrna_orfs)} ORFs in the mRNA", icon=":material/table_rows:"):
        st.dataframe(orf_frame(mrna_orfs), hide_index=True)
    with st.expander("ORFs in the gene region instead, mostly chance hits in introns", icon=":material/search:"):
        region_orfs = find_orfs(region, min_aa)
        show_chart(orf_chart(region, region_orfs, f"{g['symbol']} gene region: ORFs of {min_aa}+ aa"))
        st.caption(f"{len(region_orfs)} ORFs; the longest is {len(region_orfs[0]['protein']) if region_orfs else 0} aa, "
                   f"while the real protein is {len(g['protein'])} aa split across exons.")
    with st.expander(f"CpG islands: {len(islands)}", icon=":material/location_on:"):
        st.dataframe(pd.DataFrame([{"start": s + 1, "end": e, "length (bp)": e - s,
                                    "GC %": round(gc_content(region[s:e]), 1)} for s, e in islands]), hide_index=True)

# ---------------- tab 3: protein structure ----------------
with tab_prot:
    g = focus
    if focus_struct is None:
        st.warning(f"No AlphaFold model found for {g['symbol']}: UniProt has no reviewed human entry, or the "
                   "download failed. Pick another focus gene in the sidebar.", icon=":material/error:")
    else:
        info = focus_struct
        plddt = residue_plddt(info["pdb"])
        motifs = find_motif(info["sequence"])
        # the four residues that do the work: H at 1, E at 2, H at 5, H at 11 of each motif
        key = sorted({s + k for s, _ in motifs for k in (0, 1, 4, 10)})
        st.subheader(f"{g['symbol']} protein, predicted by AlphaFold")
        links = st.columns([1, 1.3, 2.7])
        links[0].link_button(f"UniProt {info['uniprot']}", f"https://www.uniprot.org/uniprotkb/{info['uniprot']}",
                             icon=":material/open_in_new:", width="stretch")
        links[1].link_button("AlphaFold DB entry", f"https://alphafold.ebi.ac.uk/entry/{info['uniprot']}",
                             icon=":material/open_in_new:", width="stretch")
        c = st.columns(4)
        c[0].metric("Length", f"{len(info['sequence']):,} aa")
        c[1].metric("Mean confidence", f"{info['mean_plddt']:.0f} / 100",
                    help="AlphaFold's pLDDT averaged over the protein. Above 90 is very high, below 50 is "
                         "likely disordered.")
        c[2].metric("Zinc active site", ", ".join(f"{s}–{e}" for s, e in motifs) or "none",
                    help="Residues matching HExxH.....H: three histidines hold a zinc ion and the glutamate does "
                         "the cutting.")
        c[3].metric("Matches the mRNA's ORF", "yes" if info["sequence"] == g["protein"] else "no",
                    help="Whether AlphaFold's sequence (UniProt's main isoform) is identical to the protein "
                         "translated from the RefSeq mRNA.")

        with st.container(border=True):
            left, right = st.columns(2)
            style = left.segmented_control("Style", ["Cartoon", "Cartoon + surface", "Sticks"], default="Cartoon",
                                           key="mol_style") or "Cartoon"
            color = right.segmented_control("Color by", ["AlphaFold confidence", "Rainbow, N to C",
                                                         "Secondary structure"], default="AlphaFold confidence",
                                            key="mol_color", disabled=style == "Sticks") or "AlphaFold confidence"
            tg = st.columns(4)
            focus_site = tg[0].toggle("Zoom to active site", value=bool(motifs), disabled=not motifs)
            labels = tg[1].toggle("Label active site", value=True, disabled=not motifs,
                                  help="Name the histidines and glutamate.")
            spin = tg[2].toggle("Spin", value=False)
            dark = tg[3].toggle("Dark background", value=False)
            st.iframe(protein_viewer_html(info["pdb"], plddt, info["sequence"], key, style, color, spin,
                                          focus_site, labels, dark), height=570)
            if color == "AlphaFold confidence" and style != "Sticks":
                st.html('<div class="gfe-legend">' + "".join(f"<span><i style='background:{col}'></i>{label}</span>"
                                                             for _, _, col, label in PLDDT_BINS) +
                        "<span><i style='background:#C026D3'></i>active-site side chains</span></div>")
            st.caption("Drag to rotate, scroll to zoom, hover a residue to see its name and confidence.")

        for s, e in motifs:
            st.markdown(f"Active site **{info['sequence'][s - 1:e]}** at residues {s}–{e}: the histidines at {s}, "
                        f"{s + 4} and {s + 10} hold the zinc, and the glutamate at {s + 1} is the catalytic base. "
                        f"Mean confidence there: **{sum(plddt[r] for r in range(s, e + 1)) / (e - s + 1):.0f}**.")
        if not motifs:
            st.caption("No HExxH.....H zinc motif, so this protein isn't a zinc metalloprotease.")
        show_chart(plddt_chart(plddt, key))
        st.download_button("AlphaFold model (PDB)", info["pdb"], f"AF-{info['uniprot']}.pdb",
                           icon=":material/download:")

# ---------------- tab 4: compare ----------------
with tab_cmp:
    if len(genes) < 3:
        st.info("Load at least 3 genes to compare them. Pick a gene set in the sidebar to start.",
                icon=":material/info:")
    else:
        features, signatures, windows = compare_tables(symbols)
        show_chart(feature_heatmap(features))
        g1, g2 = st.columns([3, 1])
        with g1.expander("What each feature means", icon=":material/menu_book:"):
            st.dataframe(pd.DataFrame([{"feature": label, "what it means": meaning}
                                       for label, meaning in FEATURES.values()]), hide_index=True,
                         column_config={"what it means": st.column_config.TextColumn(width="large")})
        g2.download_button("Feature table (CSV)", features.rename(columns={k: v[0] for k, v in FEATURES.items()})
                           .to_csv(), "gene_features.csv", icon=":material/download:", width="stretch")

        left, right = st.columns(2)
        with left:
            show_chart(distance_heatmap(signature_distance(signatures)))
            gc_r = pca(signatures)[0]["PC1"].corr(features["gc_region_%"])
            st.caption("Karlin's genomic signature: each dinucleotide's frequency divided by what its bases' "
                       f"frequencies predict, which factors out overall GC (signature PC1 vs GC: r = {gc_r:.2f}). "
                       "Pairs that stay close share dinucleotide preferences, not just GC.")
        with right:
            show_chart(pca_scatter(*pca(features), "Genes placed by all features at once",
                                        "PCA of the heatmap above. With a handful of genes, read it as a map, "
                                        "not as clusters."))

        metric = st.segmented_control("Window measurement", list(WINDOW_METRICS), default="gc_%",
                                      format_func=WINDOW_METRICS.get, key="window_metric") or "gc_%"
        show_chart(window_boxplot(windows, metric))

# ---------------- tab 5: user's own sequence ----------------
with tab_own:
    st.subheader("Analyze your own sequence")
    st.caption("Paste or upload DNA in FASTA or plain text. Spaces, numbers and line breaks are ignored.")

    def load_example():
        ex = load_gene("GAPDH")
        st.session_state.own_text = fasta(f"{ex['mrna_id']} GAPDH mRNA", ex["mrna"])

    st.button("Load an example", on_click=load_example, icon=":material/science:", help="Fills in GAPDH's mRNA.")
    upload = st.file_uploader("Upload a FASTA file", type=["fa", "fasta", "fna", "txt"])
    pasted = st.text_area("Or paste a sequence", key="own_text", height=150, placeholder=">my_sequence\nATGGCC…")
    raw = upload.read().decode(errors="replace") if upload else pasted
    records = parse_fasta(raw) if raw.strip() else []
    if not records:
        st.info("Nothing to analyze yet. Paste a sequence, upload a file or load the example.", icon=":material/info:")
    else:
        names = [h or f"sequence {i + 1}" for i, (h, _) in enumerate(records)]
        idx = st.selectbox("Record", range(len(records)), format_func=lambda i: names[i]) if len(records) > 1 else 0
        name, seq = names[idx], records[idx][1]
        bad = len(seq) - sum(seq.count(b) for b in "ACGT")
        c = st.columns(4)
        c[0].metric("Length", f"{len(seq):,} bp")
        c[1].metric("GC content", f"{gc_content(seq):.1f}%")
        c[2].metric("CpG islands", len(cpg_islands(seq)))
        c[3].metric("Non-ACGT bases", bad, help="Ambiguous bases such as N. They're skipped in GC and ORF counts.")
        if len(seq) < 50:
            st.warning("Very short sequence; windowed GC and ORFs won't say much.", icon=":material/warning:")
        show_chart(gc_chart(seq, name))
        orfs = find_orfs(seq, min_aa)
        show_chart(orf_chart(seq, orfs, f"{name}: ORFs of {min_aa}+ aa"))
        st.dataframe(orf_frame(orfs), hide_index=True)
        d = st.columns(3)
        d[0].download_button("ORF table (CSV)", orf_frame(orfs).to_csv(index=False), "ORFs.csv",
                             icon=":material/download:", width="stretch")
        d[1].download_button("Reverse complement (FASTA)", fasta(f"{name} reverse complement", reverse_complement(seq)),
                             "reverse_complement.fasta", icon=":material/download:", width="stretch")
        d[2].download_button("Translation, frame +1 (FASTA)", fasta(f"{name} frame +1", translate(seq)),
                             "translation.fasta", icon=":material/download:", width="stretch")
        with st.expander("Reverse complement and translation (frame +1)", icon=":material/swap_horiz:"):
            st.code(reverse_complement(seq), language=None, wrap_lines=True)
            st.code(translate(seq), language=None, wrap_lines=True)
