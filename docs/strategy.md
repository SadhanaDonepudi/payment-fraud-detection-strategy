# 3-tier decision strategy: approve / review / decline

> All cutoffs, volumes, and savings below are **measured** on the full
> 14-day holdout of the trained LightGBM model (`metrics.json`). See
> `docs/tableau_dashboard_spec.md` for the analyst-facing dashboard.

## Strategy summary

| tier | score rule | action | expected daily volume |
|---|---|---|---|
| **Approve** | score < 0.84955 | auto-approve, no friction | 104,463 / day |
| **Review** | 0.84955 ≤ score < 0.99701 | manual review queue (fraud analysts) | 472 / day |
| **Decline** | score ≥ 0.99701 | auto-decline + step-up auth / customer notification | 52 / day |

Total scored traffic: 104,988 transactions/day (synthetic).

## How the cutoffs were chosen

1. **Review cutoff** — set at the 1% review-rate operating point: the top 1%
   of holdout scores. At this point the model catches
   **88.04% of fraud dollars** (and 88.37% of fraud cases)
   with review precision 0.1111.
2. **Decline cutoff** — chosen by cost-based sweep. We searched
   review rates {0.5% … 2%} × decline rates {0.02% … 0.2%} and kept the pair
   maximizing **incremental net benefit vs a rules baseline**
   (`(night & amount>750) | (new device & amount>300)`):

   net benefit = (fraud $ blocked in decline + fraud $ caught in review
   − baseline fraud $ caught) − (extra reviews × $6.00 review cost)
   − (false declines × $25.00 customer-insult cost)

   The winning operating point (review 0.5%, decline 0.05%)
   yields an estimated **$967,616/year in fraud-loss savings**.

## Operational notes

- **Review queue SLA:** review-tier alerts should clear within 4 business
  hours; the queue is sized at ~472/day, i.e. ~47/analyst/day
  at 10 analysts and ~5 min per case.
- **Decline handling:** auto-declined transactions trigger an immediate
  step-up authentication push; confirmed-fraud devices are added to the
  device blocklist, which feeds back into `dev_distinct_cust`.
- **Adverse-action logging:** every decline stores the top-3 SHAP features as
  reason codes for audit (SR 11-7 model-risk alignment).
- **Monitoring:** score-band PSI, per-feature PSI, and alert-volume checks run
  daily via `sql/monitoring.sql`; PSI > 0.25 on the score mix pages the model
  owner for a retrain candidate review.
- **Retraining cadence:** monthly, or sooner if drift checks fire; labels land
  with a ~7-day lag (chargebacks / confirmed-fraud feeds).

## Cost assumptions (documented, synthetic)

| assumption | value |
|---|---|
| manual review cost | $6.00 / case |
| false-decline (customer insult) cost | $25.00 / case |
| baseline strategy | rules: (night & amount > $750) OR (new device & amount > $300) |
| annualization | holdout covers 14 of 60 synthetic days → ×365/60 |
| what "savings" means | incremental net benefit of the 3-tier score strategy **vs the rules baseline**, after review and false-decline costs |
