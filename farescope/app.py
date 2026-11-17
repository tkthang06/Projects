"""FareScope - U.S. Airfare Intelligence. Loads saved artifacts only; never retrains."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import MODELS, SEASONS, carrier_name, load_artifacts, missing_artifacts, predict_fare, top_drivers

NAVY, IVORY, GOLD, SLATE, SAGE = "#0F1B2D", "#F7F4EE", "#C9A96E", "#5B6B7F", "#8FA89B"
CLUSTER_COLORS = [NAVY, GOLD, SAGE, "#B5654A", "#5B6B7F", "#7A6C9C", "#4F7CAC", "#A44A3F"]

st.set_page_config(page_title="FareScope", page_icon="✈", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, input, button, [data-testid="stSidebar"] * {{
  font-family: 'Inter', sans-serif; }}
h1, h2, h3, h4, .fs-title, .fs-fare, .fs-route {{ font-family: 'Playfair Display', serif !important; color: {NAVY}; }}
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"] {{ visibility: hidden; }}
.block-container {{ padding-top: 3.6rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #EFEAE0; border-right: 1px solid rgba(201,169,110,.35); }}
.fs-title {{ font-size: 2.9rem; font-weight: 700; letter-spacing: .01em; line-height: 1.15; margin: 0; }}
.fs-sub {{ color: {SLATE}; letter-spacing: .28em; text-transform: uppercase; font-size: .74rem; margin-top: .55rem; }}
.fs-rule {{ height: 1px; border: 0; margin: 1.3rem 0 1.6rem;
  background: linear-gradient(90deg, {GOLD}, rgba(201,169,110,.15)); }}
.fs-card {{ background: #fff; border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.1rem;
  box-shadow: 0 6px 24px rgba(15,27,45,.07); border: 1px solid rgba(15,27,45,.05); }}
.fs-hero {{ border-top: 3px solid {GOLD}; }}
.fs-eyebrow {{ color: {SLATE}; font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; }}
.fs-fare {{ font-size: 4.4rem; font-weight: 600; line-height: 1.1; margin: .35rem 0 .2rem; }}
.fs-range {{ color: {SLATE}; font-size: .95rem; }}
.fs-route {{ font-size: 1.55rem; font-weight: 600; margin: 0; }}
.fs-meta {{ color: {SLATE}; font-size: .88rem; margin: .3rem 0 1.2rem; }}
.fs-badge {{ display: inline-block; margin-top: 1rem; padding: .32rem .95rem; border-radius: 999px;
  border: 1px solid {GOLD}; color: {NAVY}; background: rgba(201,169,110,.14);
  font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 500; }}
.fs-good {{ display: inline-block; padding: .18rem .7rem; border-radius: 999px; background: rgba(143,168,155,.22);
  color: #4d6a5c; font-size: .74rem; font-weight: 600; letter-spacing: .04em; }}
.fs-big {{ font-family: 'Playfair Display', serif; font-size: 1.9rem; color: {NAVY}; margin: .3rem 0 .1rem; }}
.fs-driver {{ display: flex; gap: .8rem; align-items: baseline; padding: .6rem 0;
  border-bottom: 1px solid rgba(201,169,110,.3); font-size: .95rem; }}
.fs-driver:last-child {{ border-bottom: 0; }}
.fs-up {{ color: #B5654A; font-weight: 600; }} .fs-down {{ color: #5f8573; font-weight: 600; }}
.fs-season {{ display: flex; gap: .7rem; }}
.fs-season div {{ flex: 1; text-align: center; padding: .7rem .3rem; border-radius: 12px; background: {IVORY};
  font-size: .8rem; color: {SLATE}; }}
.fs-season div b {{ display: block; font-family: 'Playfair Display', serif; font-size: 1.25rem; color: {NAVY}; font-weight: 600; }}
.fs-season div.sel {{ outline: 1.5px solid {GOLD}; background: #fff; }}
.fs-season div.best {{ background: rgba(143,168,155,.22); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid rgba(201,169,110,.4); }}
.stTabs [data-baseweb="tab"] {{ font-size: .95rem; padding: .6rem 0; }}
.stTabs [aria-selected="true"] {{ color: {NAVY}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {GOLD}; }}
[data-testid="stExpander"] {{ background: #fff; border-radius: 14px; border: 1px solid rgba(15,27,45,.06);
  box-shadow: 0 4px 16px rgba(15,27,45,.05); }}
.stButton button {{ border-radius: 999px; border: 1px solid {GOLD}; background: transparent; color: {NAVY}; }}
.stButton button:hover {{ background: {GOLD}; color: #fff; border-color: {GOLD}; }}
</style>
""", unsafe_allow_html=True)


def usd(x, decimals=0):
    """Dollar amount for HTML/markdown (a bare $ would start LaTeX in Streamlit markdown)."""
    return f"&#36;{x:,.{decimals}f}"


def card(html, extra=""):
    st.markdown(f'<div class="fs-card {extra}">{html}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=NAVY, size=13),
        legend=dict(orientation="h", y=-0.12, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(15,27,45,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(15,27,45,.08)", zeroline=False)
    return fig


@st.cache_resource(show_spinner="Loading FareScope models...")
def get_artifacts(version):
    """`version` is metrics.json's modification time, so a pipeline re-run reloads the models."""
    return load_artifacts()


@st.cache_data
def route_lookup(_snap, version):
    """city -> sorted list of connected cities, and (city, city) -> snapshot row index (either direction)."""
    neighbours, index = {}, {}
    for i, a, b in zip(_snap.index, _snap["city_a"], _snap["city_b"]):
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
        index[(a, b)] = index[(b, a)] = i
    return {c: sorted(v) for c, v in neighbours.items()}, index


@st.cache_data
def map_figure(_snap, top_n, names, version):
    top = _snap.dropna(subset=["lat_a", "lat_b"]).nlargest(top_n, "passengers")
    fig = go.Figure()
    for cid, name in names.items():
        part = top[top["cluster"] == cid]
        if part.empty:
            continue
        lats, lons = [], []
        for r in part.itertuples():
            lats += [r.lat_a, r.lat_b, None]
            lons += [r.lon_a, r.lon_b, None]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines", name=f"{name} ({len(part)})", hoverinfo="skip",
            line=dict(width=1.1, color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]), opacity=0.55))
    pts = pd.concat([
        top[["city_a", "lat_a", "lon_a"]].set_axis(["city", "lat", "lon"], axis=1),
        top[["city_b", "lat_b", "lon_b"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city")
    fig.add_trace(go.Scattergeo(
        lat=pts["lat"], lon=pts["lon"], text=pts["city"], mode="markers", hoverinfo="text",
        marker=dict(size=5, color=NAVY, line=dict(width=1, color=IVORY)), showlegend=False))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#EFEAE0",
                    subunitcolor="#fff", countrycolor="#fff", showlakes=False, bgcolor="rgba(0,0,0,0)")
    return style_fig(fig, 520), len(top)


# ---------------------------------------------------------------- header
st.markdown('<div class="fs-title">FareScope</div><div class="fs-sub">U.S. Airfare Intelligence</div>'
            '<hr class="fs-rule">', unsafe_allow_html=True)

if missing_artifacts():
    card("<h3>The models haven't been built yet</h3>"
         "<p>FareScope only loads saved models; it never trains inside the app. "
         "Run the pipeline once from the project folder, then reload this page:</p>"
         "<pre>python run_pipeline.py</pre>"
         f"<p class='fs-range'>Missing: {', '.join(missing_artifacts())}</p>", "fs-hero")
    st.stop()

version = (MODELS / "metrics.json").stat().st_mtime
art = get_artifacts(version)
snap = art["route_snapshot"]
metrics = art["metrics"]
names = {int(k): v for k, v in art["cluster_names"]["names"].items()}
ref_q = art["cpi_ref_quarter"]
neighbours, route_index = route_lookup(snap, version)
cities = sorted(neighbours)

# ---------------------------------------------------------------- sidebar
if "origin" not in st.session_state:
    busiest = snap.loc[snap["passengers"].idxmax()]
    st.session_state["origin"], st.session_state["dest"] = busiest["city_a"], busiest["city_b"]


def swap():
    st.session_state["origin"], st.session_state["dest"] = st.session_state["dest"], st.session_state["origin"]


with st.sidebar:
    st.markdown("### Plan a route")
    origin = st.selectbox("Origin", cities, key="origin", help="Type to search. Cities are DOT city markets, "
                          "so a metro area with several airports is a single entry.")
    options = neighbours[origin]
    if st.session_state.get("dest") not in options:
        st.session_state["dest"] = options[0]
    dest = st.selectbox("Destination", options, key="dest",
                        help=f"Only cities with a reported route from the origin are listed ({len(options)} here).")
    st.button("⇄  Swap", on_click=swap, help="Swap origin and destination. Fares are averages over both "
              "directions, so the prediction is the same either way.", width="stretch")
    quarter = st.radio("Season", list(SEASONS), format_func=SEASONS.get, index=2,
                       help="The data is quarterly, so a season is one calendar quarter.")
    st.markdown(f"<hr class='fs-rule'><span class='fs-range'>All fares are in {ref_q} dollars.</span>",
                unsafe_allow_html=True)

route = snap.loc[route_index[(origin, dest)]]
cluster_name = names[int(route["cluster"])]
cluster_mape = metrics["test"]["by_cluster"].get(cluster_name, metrics["test"]["overall"])["mape"]

tab_predict, tab_explore, tab_insights, tab_about = st.tabs(
    ["Predict", "Explore Routes", "Model Insights", "About the Data"])

# ---------------------------------------------------------------- predict
with tab_predict:
    by_season = {q: predict_fare(art, route, q) for q in SEASONS}
    result = by_season[quarter]
    fare = result["fare"]
    best_q = min(by_season, key=lambda q: by_season[q]["fare"])

    st.markdown(
        f'<div class="fs-route">{origin} &nbsp;⇄&nbsp; {dest}</div>'
        f'<div class="fs-meta">{route["nsmiles"]:,.0f} miles &nbsp;·&nbsp; {route["passengers"]:,.0f} passengers a day '
        f'&nbsp;·&nbsp; largest carrier {carrier_name(route["carrier_lg"])} ({route["large_ms"]:.0%}) '
        f'&nbsp;·&nbsp; route data as of {route["latest_period"]}</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        card(f'<div class="fs-eyebrow">Predicted average one-way fare, in {ref_q} dollars</div>'
             f'<div class="fs-fare">{usd(fare)}</div>'
             f'<div class="fs-range">Likely range {usd(fare * (1 - cluster_mape))} – {usd(fare * (1 + cluster_mape))}'
             f' &nbsp;·&nbsp; ±{cluster_mape:.0%}, the typical test error for this route type</div>'
             f'<span class="fs-badge">{cluster_name}</span>', "fs-hero")

        cells = "".join(
            f'<div class="{"sel " if q == quarter else ""}{"best" if q == best_q else ""}">'
            f'{SEASONS[q].split(" (")[0]}<b>{usd(by_season[q]["fare"])}</b>'
            f'{"<span class=fs-good>Best value</span>" if q == best_q else "&nbsp;"}</div>'
            for q in SEASONS)
        card(f'<div class="fs-eyebrow">Across the year</div><div class="fs-season" style="margin-top:.9rem">{cells}</div>')

    with right:
        actual, actual_year = route[f"actual_q{quarter}"], route[f"actual_q{quarter}_year"]
        if pd.notna(actual):
            gap = fare - actual
            card(f'<div class="fs-eyebrow">Most recent actual fare</div>'
                 f'<div class="fs-big">{usd(actual)}</div>'
                 f'<div class="fs-range">{SEASONS[quarter].split(" (")[0]} {int(actual_year)} average, '
                 f'restated in {ref_q} dollars. The prediction is {usd(abs(gap))} '
                 f'{"above" if gap > 0 else "below"} it.</div>')
        else:
            card(f'<div class="fs-eyebrow">Most recent actual fare</div>'
                 f'<div class="fs-range" style="margin-top:.6rem">This route has no reported '
                 f'{SEASONS[quarter].split(" (")[0].lower()} quarter in the data.</div>')

        rows = "".join(
            f'<div class="fs-driver"><span class="{"fs-up" if c > 0 else "fs-down"}">{"▲" if c > 0 else "▼"}</span>'
            f'<span>{text}</span></div>' for _, c, text in top_drivers(result, cluster_name))
        card(f'<div class="fs-eyebrow">Top 3 drivers of this prediction</div>'
             f'<div style="margin-top:.5rem">{rows}</div>'
             f'<div class="fs-range" style="font-size:.8rem;margin-top:.6rem">Relative to the model\'s '
             f'average prediction across all routes.</div>')

    with st.expander("What-if: change the market"):
        st.caption("Sliders start at this route's latest reported values. Move them to see how the "
                   "prediction responds; everything else about the route stays fixed.")
        rid = route["route_id"]
        c1, c2, c3 = st.columns(3)
        wi_lg = c1.slider("Largest carrier's market share", 0.10, 1.00, float(route["large_ms"]), 0.01,
                          key=f"lg_{rid}", help="Share of passengers flown by the biggest carrier on the "
                          "route. Higher means less competition.")
        wi_lf = c2.slider("Lowest-fare carrier's market share", 0.01, 1.00, float(route["lf_ms"]), 0.01,
                          key=f"lf_{rid}", help="Share of passengers flown by the carrier with the lowest "
                          "average fare. Higher means more travellers get the cheap fare.")
        pax_now = float(route["passengers"])
        wi_pax = c3.slider("Passengers per day", 10.0, float(max(1000, round(pax_now * 3, -2))), pax_now, 1.0,
                           format="%.0f", key=f"px_{rid}", help="Average daily passengers in both directions combined. "
                           "A proxy for route size and demand.")
        what_if = predict_fare(art, route, quarter, passengers=wi_pax, large_ms=wi_lg, lf_ms=wi_lf)["fare"]
        st.metric("What-if fare", f"${what_if:,.0f}", delta=f"{what_if - fare:+,.0f} vs. current prediction",
                  delta_color="off")

# ---------------------------------------------------------------- explore
with tab_explore:
    st.markdown("### The route network")
    top_n = st.slider("Show the top N routes by passengers", 50, 1500, 300, 50,
                      help="Fewer routes keeps the map fast and readable.")
    fig, shown = map_figure(snap, top_n, names, version)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
    located = snap["lat_a"].notna() & snap["lat_b"].notna()
    st.caption(f"Lines are coloured by route cluster. City positions are approximate: the DOT file's "
               f"coordinates were validated against route distances and re-estimated where wrong or missing. "
               f"{(~located).sum()} small routes ({1 - located.mean():.1%}) could not be placed and are not drawn.")

    st.markdown("### All routes")
    f1, f2 = st.columns([2, 3])
    pick = f1.multiselect("Cluster", list(names.values()), help="Leave empty to show every cluster.")
    query = f2.text_input("Search by city", placeholder="e.g. Denver", help="Matches either end of the route.")
    table = snap.copy()
    if pick:
        table = table[table["cluster_name"].isin(pick)]
    if query:
        q = query.strip().lower()
        table = table[table["city_a"].str.lower().str.contains(q, regex=False)
                      | table["city_b"].str.lower().str.contains(q, regex=False)]
    table = table.assign(
        latest_fare=table[[f"actual_q{q}" for q in SEASONS]].mean(axis=1),
        carrier=table["carrier_lg"].map(carrier_name),
    ).sort_values("passengers", ascending=False)
    st.dataframe(
        table[["city_a", "city_b", "cluster_name", "nsmiles", "passengers", "carrier", "large_ms", "latest_fare"]],
        hide_index=True, width="stretch", height=420,
        column_config={
            "city_a": "City", "city_b": "City ", "cluster_name": "Cluster",
            "nsmiles": st.column_config.NumberColumn("Miles", format="%d"),
            "passengers": st.column_config.NumberColumn("Passengers / day", format="%d"),
            "carrier": "Largest carrier",
            "large_ms": st.column_config.NumberColumn("Its share", format="percent"),
            "latest_fare": st.column_config.NumberColumn(
                "Recent avg fare", format="dollar",
                help=f"Mean of the most recent actual fare in each season, in {ref_q} dollars."),
        })
    st.caption(f"{len(table):,} of {len(snap):,} routes")

# ---------------------------------------------------------------- insights
with tab_insights:
    test, base = metrics["test"], metrics["baselines"]
    ly = base["same_quarter_last_year"]

    def metric_rows(pairs):
        return pd.DataFrame([{"Model": k, "MAE": m["mae"], "RMSE": m["rmse"], "MAPE": m["mape"],
                              "R²": m["r2"], "Bias": m.get("bias"), "Test rows": m["n"]} for k, m in pairs])

    metric_cfg = {
        "Bias": st.column_config.NumberColumn(format="$%.2f", help="Average of prediction minus actual. "
                                              "Positive means predictions run high."),
        "MAE": st.column_config.NumberColumn(format="$%.2f", help="Mean absolute error in dollars."),
        "RMSE": st.column_config.NumberColumn(format="$%.2f"),
        "MAPE": st.column_config.NumberColumn(format="percent", help="Mean absolute percentage error."),
        "R²": st.column_config.NumberColumn(format="%.3f"),
        "Test rows": st.column_config.NumberColumn(format="localized"),
    }

    st.markdown("### How accurate is it?")
    st.caption(f"Tested on quarters the model never saw: {metrics['split']['test']}. "
               f"Trained on {metrics['split']['train']} plus {metrics['split']['validation']}. "
               f"All errors are in {ref_q} dollars.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"${test['overall']['mae']:.2f}")
    m2.metric("RMSE", f"${test['overall']['rmse']:.2f}")
    m3.metric("MAPE", f"{test['overall']['mape']:.1%}")
    m4.metric("R²", f"{test['overall']['r2']:.3f}")

    st.markdown("#### Against the baselines")
    st.dataframe(metric_rows([
        ("FareScope (XGBoost + cluster), all test rows", test["overall"]),
        ("Baseline: same XGBoost without the cluster feature", base["xgboost_no_cluster"]),
        ("FareScope, rows that have a year-ago fare", ly["model_same_rows"]),
        ("Baseline: same route, same quarter, one year earlier", ly["baseline"]),
    ]), hide_index=True, width="stretch", column_config=metric_cfg)
    st.caption(f"The year-ago baseline only exists for {ly['coverage']:.0%} of test rows, so both models are "
               f"compared on exactly those rows.")
    if ly["baseline"]["mae"] < ly["model_same_rows"]["mae"]:
        card(f'<div class="fs-eyebrow">Read this first</div><p style="margin:.6rem 0 0">'
             f'On the test period, simply repeating a route\'s fare from the same quarter a year earlier '
             f'(MAE {usd(ly["baseline"]["mae"], 2)}) is <b>more accurate</b> than FareScope '
             f'(MAE {usd(ly["model_same_rows"]["mae"], 2)}). FareScope explains <i>why</i> routes differ '
             f'in price and covers routes with no year-ago fare; it is not the best forecaster of a '
             f'route that already has a fare history.</p>')
    card(f'<div class="fs-eyebrow">Does the clustering step help?</div>'
         f'<p style="margin:.6rem 0 0">{metrics["cluster_verdict"].replace("$", "&#36;")}</p>')

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Error by cluster")
        st.dataframe(metric_rows(test["by_cluster"].items()).rename(columns={"Model": "Cluster"}),
                     hide_index=True, width="stretch", column_config=metric_cfg)
    with c2:
        st.markdown("#### Before and after the 2025 Q3 sampling change")
        s = test["sampling_change"]
        st.dataframe(metric_rows([("Before 2025 Q3 (10% ticket sample)", s["before_2025Q3"]),
                                  ("From 2025 Q3 (~40% ticket sample)", s["from_2025Q3"])]).rename(
                                      columns={"Model": "Period"}),
                     hide_index=True, width="stretch", column_config=metric_cfg)

    st.markdown("#### What the model relies on")
    labels = {"nsmiles": "Distance", "log_passengers": "Passengers (log)", "large_ms": "Largest carrier share",
              "lf_ms": "Low-fare carrier share", "carrier_lg": "Largest carrier", "carrier_low": "Low-fare carrier",
              "low_is_largest": "Largest is also cheapest", "quarter": "Season", "city_a": "City (first)",
              "city_b": "City (second)", "cluster": "Route cluster",
              "prior_rel": "Route's fare a year earlier"}
    imp = pd.Series(metrics["feature_importance_gain"]).rename(labels).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=NAVY,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Average gain per split")
    st.plotly_chart(style_fig(fig, 380), width="stretch", config={"displayModeBar": False})
    st.caption("Gain measures how much a feature improves the fit each time a tree splits on it. "
               "It rewards features used in a few decisive splits, so it is not a ranking of total influence.")

    st.markdown("#### Route clusters (Model 1)")
    cl = metrics["clusters"]
    cent = pd.DataFrame(cl["centroids"])
    st.dataframe(
        cent[["name", "routes", "miles", "passengers_per_day", "largest_carrier_share",
              "low_fare_carrier_share", "q3_q1_passenger_ratio"]],
        hide_index=True, width="stretch",
        column_config={
            "name": "Cluster", "routes": st.column_config.NumberColumn("Routes", format="localized"),
            "miles": st.column_config.NumberColumn("Avg miles", format="%d"),
            "passengers_per_day": st.column_config.NumberColumn("Passengers / day", format="%.0f"),
            "largest_carrier_share": st.column_config.NumberColumn("Largest carrier share", format="percent"),
            "low_fare_carrier_share": st.column_config.NumberColumn("Low-fare carrier share", format="percent"),
            "q3_q1_passenger_ratio": st.column_config.NumberColumn(
                "Summer / winter traffic", format="%.2f",
                help="Mean Q3 passengers divided by mean Q1 passengers. Above 1 means summer-peaking."),
        })
    st.caption("Cluster centres from training-year routes. Names are assigned by rule from these values; "
               "a cluster that fits no rule is labelled with its most distinctive trait.")

    ks = sorted(int(k) for k in cl["inertia"])
    e1, e2 = st.columns(2, gap="large")
    for col, key, title in ((e1, "inertia", "Elbow plot (inertia)"), (e2, "silhouette", "Silhouette score")):
        ys = [cl[key][str(k)] for k in ks]
        fig = go.Figure(go.Scatter(x=ks, y=ys, mode="lines+markers", line=dict(color=NAVY, width=2),
                                   marker=dict(size=8, color=NAVY), showlegend=False))
        fig.add_trace(go.Scatter(x=[cl["k"]], y=[cl[key][str(cl["k"])]], mode="markers", name=f"chosen k = {cl['k']}",
                                 marker=dict(size=16, color=GOLD, line=dict(color=NAVY, width=1))))
        fig.update_layout(title=title, xaxis_title="k (number of clusters)", xaxis=dict(dtick=1))
        style_fig(fig, 320).update_layout(legend=dict(orientation="h", y=1.0, yanchor="top", x=1, xanchor="right"))
        col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("k is chosen by the highest silhouette score; the elbow plot is a sanity check.")

# ---------------------------------------------------------------- about
with tab_about:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(f"""
### Sources
- **Fares:** U.S. DOT *Consumer Airfare Report*, Table 6, "Contiguous State City-Pair Markets That
  Average At Least 10 Passengers Per Day" (data.transportation.gov, dataset `yj5y-b2ir`), 2012 onward.
- **Inflation:** CPI-U, all items, seasonally adjusted (FRED series `CPIAUCSL`), averaged by quarter.

### What a "fare" means here
- **Quarterly averages, not live prices.** Each number is the average of what passengers actually
  paid on a route during a whole quarter. It cannot tell you what a ticket costs today.
- **All fare classes together**, from basic economy to first class, **including taxes and fees**.
- **One-way and directionless.** A record covers travel in both directions between two city markets;
  a metro area with several airports counts as one market.
- **Inflation-adjusted.** Every fare is restated in **{ref_q} dollars**.

### Scope
- **Contiguous 48 states only.** Alaska, Hawaii and the territories are not in Table 6.
- **Routes averaging at least 10 passengers a day.** Very thin routes are absent.
- **2020 Q2 through 2021 Q4 are removed** because pandemic-era fares and traffic would distort the model.
""")
    with a2:
        st.markdown("""
### The 2025 Q3 methodology change
From 2025 Q3 the DOT's ticket survey grew from a 10% sample of tickets to roughly 40%. Averages for
small routes are less noisy from then on, and some small carriers appear for the first time. The
Model Insights tab reports accuracy separately before and after the change.

### What the model uses
Distance, passenger volume, the market shares of the largest and the lowest-fare carriers, who
those carriers are, whether they are the same airline, the season, the two cities, the route's cluster,
and how the route's fare compared with the national average in the same season a year earlier.

Inflation-adjusted fares have drifted down for years, so the model predicts each fare *relative to
the national average for its quarter* and converts back to dollars with a forecast of that average:
the latest one known for the season, moved along by the average yearly change seen so far. A
prediction is therefore for the **next** occurrence of the chosen season.

### What was deliberately left out
- **The largest carrier's fare and the lowest fare.** They are components of the average being
  predicted, so using them would be target leakage.
- **Holiday features.** With quarterly data a holiday is fully determined by the quarter.
- **Year.** Fares are already inflation-adjusted, and tree models cannot extrapolate to years
  they have not seen. The price-level drift is handled outside the model instead (see above).
- **The Table 1 flag, city IDs and coordinates.** They repeat what passengers, city and distance
  already say. Coordinates are used only to draw the map.

### Reading a prediction
The prediction uses each route's most recently reported traffic and market shares. It answers
"what would this route average in a given season, in a market like the latest one," not
"what will my ticket cost."
""")
