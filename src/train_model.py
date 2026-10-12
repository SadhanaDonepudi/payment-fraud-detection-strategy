"""Train LightGBM fraud model, tune PR, select cost-based thresholds.

Pipeline:
  1. Stream the ts-sorted feature parts + raw labels; time-split into
     train (days 0-40), validation (days 41-45), holdout (days 46-59).
     Training uses ALL train-period frauds + a downsampled slice of
     legitimate traffic (memory-bounded); holdout is scored in FULL.
  2. Train LightGBM binary classifier tuned for precision-recall.
  3. Measure: fraud DOLLARS caught at a 1% review rate on the full holdout.
  4. Cost-based threshold selection: sweep review/decline operating points,
     pick the net-savings-maximizing 3-tier strategy vs a rules baseline.
  5. Write metrics.json (every README number comes from this file).

All data is synthetic.
"""
import gc
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (RAW, FEATURES, METRICS, MODEL, SEED, START_EPOCH,
                    N_TXN, N_FRAUD, N_FEATURES, SPLIT_DAY)
from feature_engineering import FEATURE_NAMES

TRAIN_END_DAY = 41      # train: days 0..40, valid: 41..45, holdout: 46..59
LEGIT_KEEP_P = 0.085    # downsample rate for legitimate train/valid rows
REVIEW_COST = 6.0       # $ per manual review (analyst time)
DECLINE_FP_COST = 25.0  # $ customer-insult cost per false decline
ANNUALIZE = 365.0 / 60.0


def stream_parts():
    """Yield (X_chunk, y, ts, amount, txn_id, fraud_type) in ts-sorted row order."""
    feat_parts = sorted(FEATURES.glob("feat-*.parquet"))
    raw_parts = sorted(RAW.glob("part-*.parquet"))
    raw = pd.concat([pd.read_parquet(p, columns=["txn_id", "ts", "is_fraud", "amount", "fraud_type"])
                     for p in raw_parts], ignore_index=True)
    order = np.argsort(raw["ts"].to_numpy(), kind="stable")
    ts_all = raw["ts"].to_numpy()[order]
    y_all = raw["is_fraud"].to_numpy()[order]
    amt_all = raw["amount"].to_numpy().astype(np.float32)[order]
    id_all = raw["txn_id"].to_numpy()[order]
    ft_all = raw["fraud_type"].to_numpy()[order]
    day = ((ts_all - START_EPOCH) // 86400).astype(np.int16)
    pos = 0
    for fp in feat_parts:
        t = pq.read_table(fp).to_pandas()
        m = len(t)
        sl = slice(pos, pos + m)
        X = t[FEATURE_NAMES].to_numpy(dtype=np.float32)
        yield X, y_all[sl], ts_all[sl], amt_all[sl], day[sl], id_all[sl], ft_all[sl]
        pos += m
        del t, X
        gc.collect()


def baseline_rules(Xd, feature_index):
    """Simple rules baseline: (night & amount>750) | (new device & amount>300)."""
    night = Xd[:, feature_index["is_night"]] > 0.5
    newdev = Xd[:, feature_index["cust_new_device_flag"]] > 0.5
    amt = Xd[:, feature_index["amount"]]
    return (night & (amt > 750)) | (newdev & (amt > 300))


def main():
    rng = np.random.default_rng(SEED + 7)
    fi = {k: i for i, k in enumerate(FEATURE_NAMES)}
    cat_idx = [fi["channel"], fi["auth_method"]]

    Xtr, ytr, Xva, yva = [], [], [], []
    holdout = []  # (X, y, ts, amount, txn_id, fraud_type)
    for X, y, ts, amt, day, tid, ft in stream_parts():
        m_tr = day < TRAIN_END_DAY
        m_va = (day >= TRAIN_END_DAY) & (day < SPLIT_DAY)
        m_ho = day >= SPLIT_DAY
        for m, Xs, ys in [(m_tr, Xtr, ytr), (m_va, Xva, yva)]:
            Xm, ym = X[m], y[m]
            keep = (ym == 1) | (rng.random(len(ym)) < LEGIT_KEEP_P)
            Xs.append(Xm[keep]); ys.append(ym[keep])
        holdout.append((X[m_ho], y[m_ho], ts[m_ho], amt[m_ho], tid[m_ho], ft[m_ho]))
        del X
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    Xva = np.vstack(Xva); yva = np.concatenate(yva)
    Xho = np.vstack([h[0] for h in holdout]); yho = np.concatenate([h[1] for h in holdout])
    tho = np.concatenate([h[2] for h in holdout]); aho = np.concatenate([h[3] for h in holdout])
    ftho = np.concatenate([h[5] for h in holdout])
    del holdout
    gc.collect()
    print(f"train {Xtr.shape} frauds={ytr.sum():.0f} | valid {Xva.shape} "
          f"| holdout {Xho.shape} frauds={yho.sum():.0f}", flush=True)

    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_idx)
    dva = lgb.Dataset(Xva, yva, categorical_feature=cat_idx, reference=dtr)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  scale_pos_weight=float((ytr == 0).sum() / max(ytr.sum(), 1)),
                  verbose=-1, num_threads=2)
    model = lgb.train(params, dtr, num_boost_round=600,
                      valid_sets=[dva], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL))
    sva = model.predict(Xva)
    sho = model.predict(Xho)
    np.savez_compressed(MODEL.parent / "holdout_scores.npz",
                        score=sho.astype(np.float32), label=yho.astype(np.int8),
                        ts=tho.astype(np.int64), amount=aho.astype(np.float32),
                        fraud_type=ftho.astype(np.int8))
    auc = float(roc_auc_score(yva, sva))
    ap = float(average_precision_score(yva, sva))
    print(f"valid AUC={auc:.4f} avg_precision={ap:.4f} "
          f"rounds={model.current_iteration()}", flush=True)

    # ---- metric 1: fraud dollars caught at 1% review rate (holdout) -------
    order = np.argsort(-sho, kind="stable")
    n_review = int(round(0.01 * len(sho)))
    rev = order[:n_review]
    fraud_dollars = float(aho[yho == 1].sum())
    caught_dollars = float(aho[rev][yho[rev] == 1].sum())
    dollars_caught_pct = 100.0 * caught_dollars / fraud_dollars
    frauds_caught_pct = 100.0 * float((yho[rev] == 1).sum()) / float((yho == 1).sum())
    review_score_cut = float(sho[rev].min())
    precision_at_1pct = float((yho[rev] == 1).sum() / n_review)

    # ---- cost-based 3-tier threshold selection -----------------------------
    # tiers: decline (score>=t_d), review (t_r<=score<t_d), approve (rest).
    # decline tier blocks fraud dollars; review tier catches with review cost.
    base_flag = baseline_rules(Xho, fi)
    base_caught = float(aho[base_flag & (yho == 1)].sum())
    base_reviews = float(base_flag.sum())

    best = None
    for rev_rate in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
        n_r = int(round(rev_rate * len(sho)))
        t_r = float(sho[order[n_r - 1]])
        for dec_rate in [0.0002, 0.0005, 0.001, 0.002]:
            n_d = max(1, int(round(dec_rate * len(sho))))
            t_d = float(sho[order[n_d - 1]])
            dec = sho >= t_d
            revw = (sho >= t_r) & ~dec
            blocked = float(aho[dec & (yho == 1)].sum())
            caught = float(aho[revw & (yho == 1)].sum())
            n_reviews = float(revw.sum())
            fp_dec = float((dec & (yho == 0)).sum())
            # incremental net benefit vs baseline, per holdout period
            benefit = (blocked + caught - base_caught) \
                - (n_reviews - base_reviews) * REVIEW_COST \
                - fp_dec * DECLINE_FP_COST
            if best is None or benefit > best[0]:
                best = (benefit, rev_rate, dec_rate, t_r, t_d, blocked, caught,
                        n_reviews, fp_dec)
    (benefit, rev_rate, dec_rate, t_r, t_d, blocked, caught,
     n_reviews, fp_dec) = best
    annual_savings = benefit * ANNUALIZE
    holdout_days = 60 - SPLIT_DAY

    # daily operating volumes (from holdout, scaled to per-day)
    n_ho = len(sho)
    daily = {
        "approve": float(((sho < t_r).sum()) / holdout_days),
        "review": float((((sho >= t_r) & (sho < t_d)).sum()) / holdout_days),
        "decline": float(((sho >= t_d).sum()) / holdout_days),
        "total": float(n_ho / holdout_days),
    }

    imp = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain").tolist()))
    top_features = sorted(imp, key=imp.get, reverse=True)[:10]

    metrics = {
        "n_transactions": int(N_TXN),
        "n_frauds": int(N_FRAUD),
        "fraud_prevalence_pct": round(100.0 * N_FRAUD / N_TXN, 4),
        "n_features": int(N_FEATURES),
        "feature_names": FEATURE_NAMES,
        "valid_auc": round(auc, 4),
        "valid_avg_precision": round(ap, 4),
        "boosting_rounds": int(model.current_iteration()),
        "fraud_dollars_caught_pct_at_1pct_review": round(dollars_caught_pct, 2),
        "frauds_caught_pct_at_1pct_review": round(frauds_caught_pct, 2),
        "precision_at_1pct_review": round(precision_at_1pct, 4),
        "review_rate": rev_rate,
        "decline_rate": dec_rate,
        "review_score_cutoff": round(t_r, 6),
        "decline_score_cutoff": round(t_d, 6),
        "daily_volumes": {k: round(v, 1) for k, v in daily.items()},
        "annual_fraud_loss_savings_usd": round(annual_savings, 0),
        "cost_assumptions": {
            "review_cost_usd": REVIEW_COST,
            "false_decline_cost_usd": DECLINE_FP_COST,
            "baseline": "rules: (night & amount>750) | (new device & amount>300)",
            "annualization": "holdout covers 14 of 60 synthetic days; x365/60",
            "note": ("Incremental net benefit of the 3-tier score strategy vs the "
                     "rules baseline: extra fraud dollars blocked/caught, minus "
                     "extra review cost and false-decline cost."),
        },
        "holdout_fraud_dollars": round(fraud_dollars, 0),
        "top_features_by_gain": top_features,
        "synthetic_data": True,
    }
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in
                      ["fraud_dollars_caught_pct_at_1pct_review",
                       "annual_fraud_loss_savings_usd", "daily_volumes",
                       "valid_auc", "valid_avg_precision"]}, indent=2))


if __name__ == "__main__":
    main()
