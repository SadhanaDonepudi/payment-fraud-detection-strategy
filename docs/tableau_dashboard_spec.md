# Tableau dashboard spec — Fraud Strategy Daily Risk View

> No Tableau Desktop license is available on the build machine, so this
> document is the **dashboard design spec** a Tableau developer would build
> from, backed by the rendered reference charts in `docs/figures/`
> (matplotlib, from measured model outputs) and the repeatable SQL
> monitoring framework in `sql/monitoring.sql`.

## Data source
`fraud_scores` (one row per scored transaction): `score_date`, `fraud_score`,
`tier` (approve/review/decline from the cutoffs in `docs/strategy.md`),
`is_fraud` (lands with ~7-day label lag), plus the 35 feature columns.
Refresh: daily extract at 06:00 local.

## Dashboard 1 — Score distribution & tier mix
- **Histogram** of `fraud_score` (100 bins, log y-axis), colored by
  `is_fraud` once labels land; vertical reference lines at the review and
  decline cutoffs.
- **KPI tiles:** % of traffic in review tier, % in decline tier, fraud dollars
  caught at current review rate, review precision (7-day labeled).
- Reference render: `docs/figures/score_distribution.png`

## Dashboard 2 — Alert volumes over time
- **Stacked bars** per day: review-queue cases + auto-declines; overlay line
  for trailing-7-day mean ± 2σ bands (from `sql/monitoring.sql` check C).
- **Filters:** channel, state, merchant category.
- Reference render: `docs/figures/alert_volumes.png`

## Dashboard 3 — Model drift & feature health
- **Line chart:** weekly PSI of the score-band distribution vs the baseline
  week, with reference lines at 0.10 (watch) and 0.25 (retrain candidate).
- **Heatmap:** per-feature PSI (35 features × weeks), red/amber/green per
  the thresholds in `sql/monitoring.sql` checks A/B.
- **Table:** top drifting features with current vs baseline decile mix.
- Reference render: `docs/figures/model_drift.png`

## Dashboard 4 — Strategy P&L (executive view)
- **Waterfall:** baseline fraud loss → fraud $ blocked (decline) →
  fraud $ caught (review) → review operating cost → false-decline cost →
  **net annual savings** (measured: see `metrics.json`
  `annual_fraud_loss_savings_usd`).
- **What-if parameters:** review-cost $/case, decline cutoff slider —
  recompute net benefit live from the scored extract.

## Build notes for the Tableau developer
1. All four dashboards read the single `fraud_scores` extract; no blends.
2. Cutoffs (`review_score_cutoff`, `decline_score_cutoff`) are Tableau
   parameters so strategy can re-tune without republishing.
3. PSI reference lines (0.10 / 0.25) are constant reference lines.
4. `is_fraud` is NULL until labels land — every fraud-rate KPI uses
   `ZN()`-safe 7-day-labeled cohorts only.
