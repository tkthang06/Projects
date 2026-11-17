# FareScope — U.S. Airfare Intelligence

Predicts the **average one-way fare** on U.S. domestic routes with a two-stage pipeline
(K-Means route clustering → XGBoost), and presents it in a Streamlit app.

All dollar figures are **inflation-adjusted to 2026 Q1 dollars** (the latest quarter in the data).

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py            # download -> clean -> cluster -> train -> evaluate (~15 min)
python run_pipeline.py --refresh  # same, but re-download the raw data first
streamlit run app.py
python -m pytest                  # 20 tests
```

The app only loads the saved artifacts in `models/`; it never retrains. If they are missing it
tells you to run the pipeline.

## Results (test set: 2024 Q1 – 2026 Q1, 58,035 route-quarters)

| Model | MAE | RMSE | MAPE | R² | Bias | Rows |
|---|---|---|---|---|---|---|
| **FareScope: XGBoost + cluster** | $24.47 | $32.72 | 8.7% | 0.825 | +$4.80 | 58,035 |
| Baseline (b): same XGBoost, no cluster feature | $24.70 | $32.98 | 8.8% | 0.822 | +$5.11 | 58,035 |
| **FareScope, rows that have a year-ago fare** | **$23.12** | $30.43 | 8.0% | 0.845 | +$3.79 | 53,284 |
| Baseline (a): same route, same quarter, one year earlier | $28.95 | $39.74 | 10.0% | 0.736 | +$13.77 | 53,284 |

Bias is the mean of prediction minus actual; positive means predictions run high.

Three things to state plainly:

1. **The model beats the year-ago baseline** by $5.83 of MAE (20%) on the 92% of test rows where a
   year-ago fare exists, and it also covers the 8% that have none.
2. **Model 1 (clustering) makes no reliable difference.** In this run the cluster feature lowers
   test MAE by $0.22 (0.9%); in the previous run, before the drift adjustment, it raised it by
   $0.07. A difference that changes sign between runs is noise. The clusters are useful for
   describing routes; the model already sees distance, traffic and market shares directly.
3. **No accuracy drop at the 2025 Q3 sampling change.** MAE is $24.71 before (38,092 rows) and
   $24.02 from 2025 Q3 (19,943 rows).

Predictions still run $4.80 high on average, almost all of it on low-traffic routes (+$8.54;
hub routes are at -$0.68). Real fares fell faster in 2024-2025 than their long-run average, and a
forecast made a year ahead cannot know that.

### The remaining low-traffic bias was investigated and left alone

- It is not stable: on low-traffic routes the model runs +$8.9 high in 2024, +$12.7 in 2025 and
  -$10.9 **low** in 2026 Q1. A fixed correction would help two years and hurt the third.
- It comes from a reversal that starts in the test period. Low-traffic routes were priced 7.7% above
  the national level in 2012, rising to 11.7% by 2022, then falling back to 8.7% in 2025. The
  routes that were priciest a year earlier are the ones over-predicted most (+$12 to +$13).
- Nothing knowable in advance fixes it. On the 2023 validation year the low-traffic bias is already
  -$0.7. A separate price level and drift per cluster made validation worse (MAE $30.87 vs. $29.84),
  and recency-weighted training changed validation MAE by under $0.40 without touching this bias.

Any correction would therefore have to be fitted to the test set itself, which would make the
reported accuracy meaningless.

### How price-level drift is handled

The first version of this model followed the original feature list exactly, with no time signal.
It scored MAE $41.24 on the test set and lost to the year-ago baseline ($28.95). The cause was
drift: inflation-adjusted fares have fallen almost every year (mean real fare $369 in 2012, $339 in
2019, $320 in 2023, $305 in 2024, $288 in 2025), so a model that learns the 2012–2023 level ran
$13–$55 high on 2024–2025 quarters.

Two changes fixed it, both chosen on the validation year (2023) before looking at the test set:

- **The target is relative.** The model predicts `log(real_fare)` minus the national mean log fare
  of the row's quarter. To get dollars back it adds a forecast of that level made a year ahead:
  the level of the same quarter one year earlier, plus the average year-over-year change in the
  level over all earlier years ("seasonal naive with drift"; about -1.4% to -2.2% a year here).
  `year` is still not a feature.
- **One added feature, `prior_rel`:** the route's fare in the same quarter a year earlier, relative
  to that quarter's national level (missing when the route has no such row). This is a departure
  from the original feature list. It only uses the previous year, so it is not leakage.

On validation these moved MAE from $32.00 to $29.84 and cut the bias from +$19 to a few dollars.
Without the drift term the test MAE was $25.61 with a bias of +$10.09. The
validation gain is small because 2023 follows the post-pandemic rebound year; the test gain is much
larger.

In the app, `prior_rel` comes from the route's most recent fare for the chosen season, and the
price level is the forecast for the next occurrence of that season.

### Per cluster (test)

| Cluster | MAE | MAPE | R² | Bias | Rows |
|---|---|---|---|---|---|
| Small Regional | $25.12 | 12.5% | 0.857 | +$1.74 | 8,498 |
| Busy Hub-to-Hub | $19.19 | 6.7% | 0.874 | -$0.68 | 17,240 |
| Cluster 2: low traffic | $27.12 | 8.8% | 0.704 | +$8.54 | 32,297 |

## Data

- **DOT Consumer Airfare Report, Table 6** — "Contiguous State City-Pair Markets That Average At
  Least 10 Passengers Per Day" (Socrata `yj5y-b2ir`), 2012 Q1 – 2026 Q1, 326,198 rows. Records are
  directionless quarterly averages across all fare classes, including taxes and fees.
- **CPI-U** (FRED `CPIAUCSL`), monthly, averaged to quarters.

### Cleaning

- 2020 Q2 – 2021 Q4 removed (COVID); rows with missing/non-positive values removed; 2 duplicated
  route-quarters collapsed → **293,909 rows, 8,967 routes, 375 city markets**.
- Fare winsorized to [$81.21, $483.32], the 0.5th/99.5th percentiles of training years only.
- Carriers: `US → AA`, `VX → AS`. Codes flagged as retired but left unmapped: `FL` (last seen
  2014), `KS`, `OO`, `ZK`. Several small-carrier codes first appear in 2025 (`4B`, `9X`, `AN`,
  `KG`, `LF`, `M5`, `NE`, `TJ`, `HA`); the model treats them as an unknown carrier.
- `route_id` puts the smaller city market ID first, so a route has one key in either direction.
- `real_fare = fare × CPI(2026 Q1) / CPI(row's quarter)`.

### Things in the raw data that differ from the documentation

- **Coordinates are unreliable.** `location_1/2` are `POINT (lon lat)` strings, but they are empty
  from 2021 on, absent for 56 cities, and wrong for many others (Dallas/Fort Worth is placed in
  Pennsylvania, Phoenix in Florida, Miami on top of Minneapolis). `clean.py` checks every city's
  coordinate against its route distances (`nsmiles`), keeps the 177 that agree, and re-estimates
  128 by fitting a position to the distances from trusted cities. 70 tiny markets with fewer than
  three usable routes stay unplaced: 92 routes (1.0%, 0.2% of passengers) are not drawn on the map.
  Estimated positions are near the airports, not city centres. Coordinates are map-only.
- `location_*_city` is populated (it duplicates `city1`/`city2`) and there is an extra `tbl` column.
- City market 32544 appears under two names; the most recent is used.

## Models

**Model 1 — K-Means.** One row per route from training years (2012–2022): mean distance, mean
log passengers, mean largest-carrier share, mean low-fare-carrier share, and seasonality (mean Q3
÷ mean Q1 passengers, clipped to [1/3, 3], median-filled). Silhouette chose **k = 3** (0.290, vs.
0.287 for k = 5 — nearly a tie; the elbow plot does not contradict it). Names come from a rule
function on centroid z-scores; the third cluster matched no rule and got the fallback name
"Cluster 2: low traffic". 699 routes absent from training are assigned with `kmeans.predict`.

**Model 2 — XGBoost** on `log(real_fare)` relative to its quarter's national level (see above),
native categoricals. Split by time: train 2012–2022,
validate 2023, test 2024 →. Hyperparameters are tuned on validation with early stopping, then the
model is refit on train + validation with the best tree count.
The search covers 8 randomly sampled cells (fixed seed) of a 32-cell grid over `max_depth`,
`learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`; the full grid takes about an
hour and a partial run showed validation MAE varying by only ~$1 across it (`N_COMBOS = None` in
`src/train.py` runs all 32). Chosen: depth 6, learning rate 0.15, min child weight 10,
row subsampling 0.8, column subsampling 0.7, 136 trees.

### Features

`nsmiles`, `log_passengers`, `large_ms`, `lf_ms`, `carrier_lg`, `carrier_low`, `low_is_largest`,
`quarter`, `city_a`, `city_b`, `cluster`, plus `prior_rel` (added to handle drift; see above).

Note that passengers, market shares and carriers are measured in the same quarter as the fare. The
app predicts from each route's most recent values.

### Deliberately excluded

| Excluded | Why |
|---|---|
| `fare_lg`, `fare_low` | Target leakage: they are components of the average fare being predicted. Dropped in `clean.py` so they cannot reach the model. |
| Holiday features | With quarterly data, holidays are fully determined by the quarter. |
| `year` | The target is inflation-adjusted, and trees cannot extrapolate to unseen years. Drift is handled by the relative target instead. |
| `table_1_flag`, IDs, lat/lon | Redundant with passengers, city and distance. |

## Layout

```
data/raw/  data/processed/  models/  reports/
src/  download.py  clean.py  features.py  cluster.py  train.py  predict.py
run_pipeline.py   app.py   .streamlit/config.toml   tests/
```

`reports/` holds the elbow, silhouette and feature-importance plots and the centroid table;
`models/metrics.json` holds every number shown in the app.
