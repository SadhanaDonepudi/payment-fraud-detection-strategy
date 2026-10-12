"""Render matplotlib figures for docs/figures (Tableau dashboard spec inputs).

Figures (all from the measured holdout scores of the trained model):
  1. score_distribution.png - fraud score histogram, fraud vs legitimate
  2. alert_volumes.png      - daily review/decline alert volumes over time
  3. model_drift.png        - PSI of score-band distribution vs baseline, by week
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import METRICS, MODEL, FIGURES, START_EPOCH

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


def psi(expected_pct, actual_pct, eps=1e-4):
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text())
    z = np.load(MODEL.parent / "holdout_scores.npz")
    score, y, ts = z["score"], z["label"], z["ts"]
    t_r, t_d = m["review_score_cutoff"], m["decline_score_cutoff"]
    day = ((ts - START_EPOCH) // 86400).astype(int)

    # 1. score distribution ------------------------------------------------
    rng = np.random.default_rng(0)
    leg = score[y == 0]
    leg_s = leg[rng.choice(len(leg), size=min(200_000, len(leg)), replace=False)]
    frd = score[y == 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(leg_s, bins=80, alpha=0.6, label=f"legitimate (n={len(leg):,})", color="#4a7c59")
    ax.hist(frd, bins=80, alpha=0.8, label=f"fraud (n={len(frd):,})", color="#c0392b")
    ax.axvline(t_r, color="black", ls="--", lw=1.2, label=f"review cutoff {t_r:.4f}")
    ax.axvline(t_d, color="#7d3c98", ls=":", lw=1.2, label=f"decline cutoff {t_d:.4f}")
    ax.set_xlabel("fraud score"); ax.set_ylabel("transactions")
    ax.set_title("Model score distribution on holdout (synthetic data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "score_distribution.png"); plt.close(fig)

    # 2. alert volumes over time -------------------------------------------
    days = np.arange(day.min(), day.max() + 1)
    rev = ((score >= t_r) & (score < t_d)).astype(int)
    dec = (score >= t_d).astype(int)
    rev_d = np.array([rev[day == d].sum() for d in days])
    dec_d = np.array([dec[day == d].sum() for d in days])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(days, rev_d, label="review queue", color="#e67e22", alpha=0.8)
    ax.bar(days, dec_d, bottom=rev_d, label="auto-decline", color="#c0392b", alpha=0.9)
    ax.set_xlabel("synthetic day"); ax.set_ylabel("alerts per day")
    ax.set_title("Daily alert volumes: review queue + auto-declines (synthetic data)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "alert_volumes.png"); plt.close(fig)

    # 3. model drift: weekly PSI of score-band mix vs first week --------------
    band_edges = sorted(set([0.01, 0.05, 0.2, 0.5, t_r, t_d]))
    bands = np.digitize(score, bins=band_edges)
    weeks = (day - day.min()) // 7
    base = np.bincount(bands[weeks == 0], minlength=8) / (weeks == 0).sum()
    wk, psis = [], []
    for w in sorted(np.unique(weeks)):
        if w == 0:
            continue
        cur = np.bincount(bands[weeks == w], minlength=8) / (weeks == w).sum()
        wk.append(w); psis.append(psi(base, cur))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(wk, psis, marker="o", color="#2471a3")
    ax.axhline(0.10, color="#c0392b", ls="--", lw=1.2, label="PSI alert threshold 0.10")
    ax.axhline(0.25, color="#7d3c98", ls=":", lw=1.2, label="PSI critical 0.25")
    ax.set_xlabel("week (vs baseline week 0)"); ax.set_ylabel("PSI")
    ax.set_title("Score-band distribution drift vs baseline week (synthetic data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "model_drift.png"); plt.close(fig)
    print("figures written:", sorted(p.name for p in FIGURES.glob('*.png')))


if __name__ == "__main__":
    main()
