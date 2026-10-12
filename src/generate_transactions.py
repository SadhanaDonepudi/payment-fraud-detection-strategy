"""Generate EXACTLY N_TXN synthetic mobile payment transactions (chunked parquet).

All data is SYNTHETIC and labeled as such. Nothing here represents real
customers, devices, merchants, or payments.

Fraud is injected at EXACTLY the target prevalence (0.13% -> N_FRAUD rows)
with fraudster-like patterns: higher amounts (plus small "probe" amounts),
night-time bias, brand-new devices, unfamiliar merchants, weak
authentication -- but with realistic overlap (some daytime/local frauds, and
3% legitimate travel) so no single attribute trivially separates the classes.
Legitimate traffic follows per-customer habits so velocity/behavior features
separate the two classes.

Chunking: one parquet file per chunk keeps peak RAM low (~600MB for 630k
rows in float32/int32 dtypes).
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (N_TXN, N_FRAUD, CHUNK_SIZE, SEED, WINDOW_DAYS,
                    START_EPOCH, N_CUSTOMERS, N_DEVICES, N_MERCHANTS, RAW)

# ---- entity pools -------------------------------------------------------
N_STATES = 50
# rough state population weights (synthetic, just for spread)
STATE_W = np.array([
    12, 2, 23, 10, 128, 19, 12, 3, 70, 35, 5, 6, 42, 22, 10, 10, 15, 15,
    4, 20, 23, 34, 19, 10, 20, 4, 7, 10, 5, 30, 7, 64, 35, 26, 40, 13,
    14, 43, 4, 17, 3, 23, 100, 11, 2, 28, 26, 6, 19, 2], dtype=float)
STATE_W /= STATE_W.sum()
# approximate state centroids (lat, lon) - coarse, synthetic use only
STATE_GEO = np.array([
    [32.8, -86.8], [61.4, -152.3], [33.7, -111.4], [34.9, -92.3],
    [36.7, -119.7], [39.1, -105.5], [41.6, -72.7], [39.3, -75.5],
    [27.8, -81.7], [32.9, -83.1], [21.1, -157.0], [44.2, -114.5],
    [40.3, -89.2], [39.8, -86.1], [42.0, -93.5], [38.5, -98.4],
    [37.7, -85.3], [30.4, -92.0], [44.7, -69.4], [39.0, -76.7],
    [42.2, -71.4], [43.3, -84.6], [46.4, -94.3], [32.7, -89.6],
    [38.6, -92.5], [47.0, -110.0], [41.1, -96.0], [39.3, -116.6],
    [43.4, -71.5], [40.3, -74.5], [35.7, -106.1], [42.1, -74.9],
    [35.6, -79.5], [47.4, -100.5], [40.4, -82.9], [35.5, -97.5],
    [44.6, -120.5], [40.6, -77.2], [41.7, -71.5], [34.0, -81.0],
    [44.2, -99.4], [35.9, -86.7], [31.0, -100.1], [40.8, -111.9],
    [44.0, -72.7], [37.9, -78.0], [47.4, -120.6], [38.6, -80.6],
    [43.0, -89.5], [43.0, -107.3]], dtype=np.float32)

N_MCC = 10
MCC_MEDIAN = np.array([12, 25, 40, 60, 85, 120, 180, 250, 400, 900],
                      dtype=np.float32)  # grocery..luxury travel


def build_pools(seed: int):
    rng = np.random.default_rng(seed)
    # customers -----------------------------------------------------------
    cust_state = rng.choice(N_STATES, size=N_CUSTOMERS, p=STATE_W).astype(np.int8)
    home = STATE_GEO[cust_state] + rng.normal(0, 1.2, (N_CUSTOMERS, 2)).astype(np.float32)
    n_dev = rng.integers(1, 4, size=N_CUSTOMERS)          # 1-3 devices each
    dev_starts = np.zeros(N_CUSTOMERS + 1, dtype=np.int64)
    dev_starts[1:] = np.cumsum(n_dev)
    dev_pool = rng.integers(0, N_DEVICES, size=int(dev_starts[-1]), dtype=np.int32)
    # preferred merchants: 5 per customer
    pref_merch = rng.integers(0, N_MERCHANTS, size=(N_CUSTOMERS, 5), dtype=np.int32)
    spend_scale = rng.lognormal(0.0, 0.6, size=N_CUSTOMERS).astype(np.float32)
    # merchants -----------------------------------------------------------
    merch_mcc = rng.integers(0, N_MCC, size=N_MERCHANTS).astype(np.int8)
    merch_state = rng.choice(N_STATES, size=N_MERCHANTS, p=STATE_W).astype(np.int8)
    return dict(rng=rng, cust_state=cust_state, home=home, dev_starts=dev_starts,
                dev_pool=dev_pool, pref_merch=pref_merch, spend_scale=spend_scale,
                merch_mcc=merch_mcc, merch_state=merch_state)


def gen_chunk(chunk_idx: int, pools: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1000 + chunk_idx)
    n = CHUNK_SIZE
    if chunk_idx == n_chunks - 1:  # last chunk takes the remainder
        n = N_TXN - CHUNK_SIZE * chunk_idx

    cust = rng.integers(0, N_CUSTOMERS, size=n, dtype=np.int32)
    P = pools

    # device: one of the customer's own devices
    starts = P["dev_starts"][cust]
    counts = P["dev_starts"][cust + 1] - starts
    dev = P["dev_pool"][starts + rng.integers(0, 2**31 - 1, size=n) % counts].astype(np.int32)

    # merchant: 80% preferred, 20% random
    use_pref = rng.random(n) < 0.80
    merch = np.empty(n, dtype=np.int32)
    mp = np.where(use_pref)[0]
    merch[mp] = P["pref_merch"][cust[mp], rng.integers(0, 5, size=mp.size)]
    mr = np.where(~use_pref)[0]
    merch[mr] = rng.integers(0, N_MERCHANTS, size=mr.size, dtype=np.int32)

    # timestamp: uniform day, time-of-day mixture peaking midday
    day = rng.integers(0, WINDOW_DAYS, size=n)
    tod = rng.random(n)
    hour = np.where(tod < 0.62, rng.normal(13, 4, n),
                    np.where(tod < 0.90, rng.normal(19, 2.5, n),
                             rng.uniform(0, 24, n)))
    hour = np.clip(hour, 0, 23.99)
    ts = (START_EPOCH + day * 86400 + (hour * 3600).astype(np.int64)).astype(np.int32)

    # amount: lognormal around $30, scaled by customer spend and merchant
    # category (grocery ~$12 median ... luxury travel ~$900 median)
    mcc = P["merch_mcc"][merch]
    amt = (rng.lognormal(3.4, 1.0, n) * P["spend_scale"][cust]
           * MCC_MEDIAN[mcc] / 105.0).astype(np.float32)
    amt = np.clip(amt, 1.0, 12000.0)

    channel = rng.choice(3, size=n, p=[0.70, 0.20, 0.10]).astype(np.int8)
    auth = rng.choice(4, size=n, p=[0.50, 0.30, 0.15, 0.05]).astype(np.int8)

    # geo: near home with jitter
    lat = (P["home"][cust, 0] + rng.normal(0, 0.8, n)).astype(np.float32)
    lon = (P["home"][cust, 1] + rng.normal(0, 0.8, n)).astype(np.float32)
    state = P["cust_state"][cust].copy()

    df = pd.DataFrame({
        "txn_id": np.arange(chunk_idx * CHUNK_SIZE,
                            chunk_idx * CHUNK_SIZE + n, dtype=np.int32),
        "ts": ts, "customer_id": cust, "device_id": dev,
        "merchant_id": merch, "mcc": mcc, "amount": amt,
        "channel": channel, "auth_method": auth,
        "state_id": state, "lat": lat, "lon": lon,
        "is_fraud": np.zeros(n, dtype=np.int8),
        "fraud_type": np.zeros(n, dtype=np.int8),  # 0 legit, 1 obvious, 2 probe, 3 sophisticated
    })

    # ---- inject EXACT fraud count for this chunk -------------------------
    n_f = fraud_per_chunk[chunk_idx]
    fidx = rng.choice(n, size=n_f, replace=False)
    fi = df.index[fidx]
    probe = rng.random(n_f) < 0.30  # 30% are small "probe" frauds
    famt = np.empty(n_f, dtype=np.float32)
    famt[probe] = rng.uniform(1, 10, probe.sum()).astype(np.float32)
    big = ~probe
    famt[big] = np.clip(rng.lognormal(5.2, 0.8, big.sum()), 20, 8000).astype(np.float32)
    df.loc[fi, "amount"] = famt
    # hours: 80% night (0-6), 20% daytime (blend in)
    night = rng.random(n_f) < 0.80
    fhour = np.where(night, rng.uniform(0, 6, n_f), rng.uniform(7, 23, n_f))
    fday = rng.integers(0, WINDOW_DAYS, size=n_f)
    df.loc[fi, "ts"] = (
        START_EPOCH + fday * 86400 + (fhour * 3600).astype(np.int64)).astype(np.int32)
    # brand-new device for 85% of frauds; 15% reuse a known customer device
    # (account-takeover on a familiar device happens too)
    new_dev_mask = rng.random(n_f) < 0.85
    new_dev = np.empty(n_f, dtype=np.int32)
    nd = np.where(new_dev_mask)[0]
    new_dev[nd] = rng.integers(0, N_DEVICES, size=nd.size, dtype=np.int32)
    kd = np.where(~new_dev_mask)[0]
    kcust = cust[fidx[kd]]
    kstarts = P["dev_starts"][kcust]
    kcounts = P["dev_starts"][kcust + 1] - kstarts
    new_dev[kd] = P["dev_pool"][kstarts + rng.integers(0, 2**31 - 1, size=kd.size) % kcounts].astype(np.int32)
    df.loc[fi, "device_id"] = new_dev
    # fraud_type: 1 = obvious, 2 = probe (diagnostic label, synthetic)
    ftype = np.where(probe, 2, 1).astype(np.int8)
    # unfamiliar merchant
    new_merch = rng.integers(0, N_MERCHANTS, size=n_f, dtype=np.int32)
    df.loc[fi, "merchant_id"] = new_merch
    df.loc[fi, "mcc"] = P["merch_mcc"][new_merch]
    # geo: 50% far-from-home, 50% local (fraud happens near home too)
    far = rng.random(n_f) < 0.50
    flat = np.empty(n_f, dtype=np.float32)
    flon = np.empty(n_f, dtype=np.float32)
    fstate = np.empty(n_f, dtype=np.int8)
    ff = np.where(far)[0]
    fs = rng.integers(0, N_STATES, size=ff.size).astype(np.int8)
    fstate[ff] = fs
    flat[ff] = (STATE_GEO[fs, 0] + rng.normal(0, 1.5, ff.size)).astype(np.float32)
    flon[ff] = (STATE_GEO[fs, 1] + rng.normal(0, 1.5, ff.size)).astype(np.float32)
    lc = np.where(~far)[0]
    fstate[lc] = P["cust_state"][cust[fidx[lc]]]
    flat[lc] = (P["home"][cust[fidx[lc]], 0] + rng.normal(0, 0.8, lc.size)).astype(np.float32)
    flon[lc] = (P["home"][cust[fidx[lc]], 1] + rng.normal(0, 0.8, lc.size)).astype(np.float32)
    df.loc[fi, "state_id"] = fstate
    df.loc[fi, "lat"] = flat
    df.loc[fi, "lon"] = flon
    # weaker auth, web channel bias (with overlap to legitimate)
    df.loc[fi, "channel"] = rng.choice(
        3, size=n_f, p=[0.35, 0.50, 0.15]).astype(np.int8)
    df.loc[fi, "auth_method"] = rng.choice(
        4, size=n_f, p=[0.20, 0.20, 0.30, 0.30]).astype(np.int8)
    df.loc[fi, "is_fraud"] = 1
    df.loc[fi, "fraud_type"] = ftype

    # ---- sophisticated frauds: high amounts, fully legitimate-looking ----
    # profile (daytime, known device/merchant, local geo, strong auth); only
    # the amount is anomalous. These carry large dollars but score low.
    n_soph = int(round(0.20 * n_f))
    sidx = rng.choice(fidx, size=n_soph, replace=False)
    si = df.index[sidx]
    scust = cust[sidx]
    sstarts = P["dev_starts"][scust]
    scounts = P["dev_starts"][scust + 1] - sstarts
    df.loc[si, "device_id"] = P["dev_pool"][
        sstarts + rng.integers(0, 2**31 - 1, size=n_soph) % scounts].astype(np.int32)
    df.loc[si, "merchant_id"] = P["pref_merch"][
        scust, rng.integers(0, 5, size=n_soph)]
    df.loc[si, "mcc"] = P["merch_mcc"][df.loc[si, "merchant_id"].to_numpy()]
    shour = rng.uniform(9, 18, n_soph)
    sday = rng.integers(0, WINDOW_DAYS, size=n_soph)
    df.loc[si, "ts"] = (
        START_EPOCH + sday * 86400 + (shour * 3600).astype(np.int64)).astype(np.int32)
    df.loc[si, "amount"] = np.clip(
        rng.lognormal(5.6, 0.4, n_soph), 50, 2000).astype(np.float32)
    df.loc[si, "channel"] = np.zeros(n_soph, dtype=np.int8)
    df.loc[si, "auth_method"] = np.zeros(n_soph, dtype=np.int8)
    df.loc[si, "state_id"] = P["cust_state"][scust]
    df.loc[si, "lat"] = (
        P["home"][scust, 0] + rng.normal(0, 0.8, n_soph)).astype(np.float32)
    df.loc[si, "lon"] = (
        P["home"][scust, 1] + rng.normal(0, 0.8, n_soph)).astype(np.float32)
    df.loc[si, "fraud_type"] = np.full(n_soph, 3, dtype=np.int8)

    # ---- legitimate travel: 3% of legit txns happen far from home ---------
    legit_idx = np.where(df["is_fraud"].to_numpy() == 0)[0]
    trav = rng.choice(legit_idx, size=int(0.03 * legit_idx.size), replace=False)
    ti = df.index[trav]
    ts_trav = rng.integers(0, N_STATES, size=trav.size).astype(np.int8)
    df.loc[ti, "state_id"] = ts_trav
    df.loc[ti, "lat"] = (
        STATE_GEO[ts_trav, 0] + rng.normal(0, 1.5, trav.size)).astype(np.float32)
    df.loc[ti, "lon"] = (
        STATE_GEO[ts_trav, 1] + rng.normal(0, 1.5, trav.size)).astype(np.float32)

    # ---- legitimate device upgrades: 2% of legit txns on a new device ----
    upg = rng.choice(legit_idx, size=int(0.02 * legit_idx.size), replace=False)
    ui = df.index[upg]
    df.loc[ui, "device_id"] = rng.integers(
        0, N_DEVICES, size=upg.size, dtype=np.int32)
    return df


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    global n_chunks, fraud_per_chunk
    n_chunks = (N_TXN + CHUNK_SIZE - 1) // CHUNK_SIZE
    fraud_per_chunk = [N_FRAUD // n_chunks] * n_chunks
    for i in range(N_FRAUD % n_chunks):
        fraud_per_chunk[i] += 1
    assert sum(fraud_per_chunk) == N_FRAUD

    pools = build_pools(SEED)
    total, fraud = 0, 0
    for c in range(n_chunks):
        df = gen_chunk(c, pools)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       RAW / f"part-{c:03d}.parquet")
        total += len(df)
        fraud += int(df["is_fraud"].sum())
        print(f"chunk {c}: {len(df):,} rows, {int(df['is_fraud'].sum())} frauds", flush=True)
        del df
    print(f"TOTAL: {total:,} rows, {fraud:,} frauds "
          f"({fraud / total:.4%} prevalence)")


if __name__ == "__main__":
    main()
