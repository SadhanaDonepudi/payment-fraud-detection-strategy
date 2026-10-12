# Payment fraud data pipeline - test suite
#
# Runs the FULL pipeline end-to-end at small scale (FRAUD_SCALE=small ->
# 100,000 transactions, exactly 130 frauds) inside a tmp dir (FRAUD_ROOT),
# then asserts the pipeline's exactness guarantees plus README consistency.
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole pipeline once at small scale; share outputs across tests."""
    import runpy
    workdir = tmp_path_factory.mktemp("fraud_small")
    env = dict(os.environ, FRAUD_SCALE="small", FRAUD_ROOT=str(workdir),
               PYTHONPATH=str(SRC))
    # runpy in-process with patched env
    old = dict(os.environ)
    os.environ.update(env)
    try:
        for mod in ["config", "generate_transactions", "feature_engineering",
                    "train_model", "make_figures"]:
            sys.modules.pop(mod, None)
        runpy.run_module("generate_transactions", run_name="__main__")
        runpy.run_module("feature_engineering", run_name="__main__")
        runpy.run_module("train_model", run_name="__main__")
        runpy.run_module("make_figures", run_name="__main__")
    finally:
        os.environ.clear()
        os.environ.update(old)
        for mod in ["config", "generate_transactions", "feature_engineering",
                    "train_model", "make_figures"]:
            sys.modules.pop(mod, None)
    return workdir


def test_exact_transaction_count(pipeline):
    import pyarrow.parquet as pq
    total = sum(pq.read_table(p).num_rows
                for p in (pipeline / "data" / "raw").glob("part-*.parquet"))
    assert total == 100_000, f"expected exactly 100,000 rows, got {total:,}"


def test_exact_fraud_prevalence(pipeline):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    fraud = sum(pc.sum(pq.read_table(p, columns=["is_fraud"])
                       .column("is_fraud")).as_py()
                for p in (pipeline / "data" / "raw").glob("part-*.parquet"))
    assert fraud == 130, f"expected exactly 130 frauds, got {fraud}"
    assert abs(fraud / 100_000 - 0.0013) < 1e-12


def test_exact_feature_count(pipeline):
    import pyarrow.parquet as pq
    t = pq.read_table(str(pipeline / "data" / "features" / "feat-000.parquet"))
    assert t.num_columns == 35, f"expected 35 features, got {t.num_columns}"
    assert len(set(t.schema.names)) == 35, "feature names must be unique"


def test_no_feature_leakage_columns(pipeline):
    import pyarrow.parquet as pq
    t = pq.read_table(str(pipeline / "data" / "features" / "feat-000.parquet"))
    assert "is_fraud" not in t.schema.names, "label must not leak into features"


def test_model_metric_sanity(pipeline):
    m = json.loads((pipeline / "metrics.json").read_text())
    assert m["valid_auc"] > 0.85, f"AUC too low: {m['valid_auc']}"
    assert m["valid_avg_precision"] > 0.30
    assert 0 <= m["fraud_dollars_caught_pct_at_1pct_review"] <= 100
    assert m["annual_fraud_loss_savings_usd"] > 0
    assert m["n_features"] == 35
    assert len(m["feature_names"]) == 35
    assert m["synthetic_data"] is True


def test_readme_number_consistency():
    """Every number the README reports must equal metrics.json exactly."""
    readme = (ROOT / "README.md").read_text()
    metrics = json.loads((ROOT / "metrics.json").read_text())
    for text in ["6,300,000", "8,190", "0.13%", "35"]:
        assert text in readme, f"README missing {text}"
    # same formatting as src/finalize_docs.py
    assert f"{metrics['fraud_dollars_caught_pct_at_1pct_review']:.2f}%" in readme
    assert f"{metrics['annual_fraud_loss_savings_usd']:,.0f}" in readme
    assert f"{metrics['valid_auc']:.4f}" in readme
    assert f"{metrics['valid_avg_precision']:.4f}" in readme


def test_figures_exist(pipeline):
    figs = pipeline / "docs" / "figures"
    for name in ["score_distribution.png", "alert_volumes.png", "model_drift.png"]:
        assert (figs / name).exists(), f"missing figure {name}"
