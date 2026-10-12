"""Fill measured numbers from metrics.json into README.md and docs/strategy.md.

This is what guarantees the hard rule: every number in the README equals the
actual pipeline output. Run after train_model.py.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "metrics.json").read_text())
dv = m["daily_volumes"]


def fmt_int(x):
    return f"{x:,.0f}"


repl = {
    "{{valid_auc}}": f"{m['valid_auc']:.4f}",
    "{{valid_avg_precision}}": f"{m['valid_avg_precision']:.4f}",
    "{{dollars_caught}}": f"{m['fraud_dollars_caught_pct_at_1pct_review']:.2f}",
    "{{frauds_caught}}": f"{m['frauds_caught_pct_at_1pct_review']:.2f}",
    "{{precision_at_1}}": f"{m['precision_at_1pct_review']:.4f}",
    "{{review_cutoff}}": f"{m['review_score_cutoff']:.5f}",
    "{{decline_cutoff}}": f"{m['decline_score_cutoff']:.5f}",
    "{{approve}}": fmt_int(dv["approve"]),
    "{{review}}": fmt_int(dv["review"]),
    "{{decline}}": fmt_int(dv["decline"]),
    "{{savings}}": fmt_int(m["annual_fraud_loss_savings_usd"]),
    "{{top_features}}": ", ".join(f"`{f}`" for f in m["top_features_by_gain"][:6]),
    "{{total}}": fmt_int(dv["total"]),
    "{{rev_rate}}": f"{m['review_rate']:.1%}",
    "{{dec_rate}}": f"{m['decline_rate']:.2%}",
    "{{review_per_analyst}}": f"{dv['review'] / 10:.0f}",
}

for rel in ["README.md", "docs/strategy.md"]:
    p = ROOT / rel
    text = p.read_text()
    for k, v in repl.items():
        text = text.replace(k, v)
    leftover = set(re.findall(r"\{\{[a-z_]+\}\}", text))
    assert not leftover, f"unfilled placeholders in {rel}: {leftover}"
    p.write_text(text)
    print(f"finalized {rel}")
