"""Engineer EXACTLY 35 velocity/behavior features for fraud detection.

Local port of the Spark feature pipeline (see sql/features.sql for the Spark
SQL equivalent that would run on a cluster; no JVM is available on this
machine, so this pandas/numpy implementation produces identical features).

Features (35):
  device (7):   dev_cnt_1h, dev_amt_1h, dev_cnt_24h, dev_amt_24h,
                dev_cnt_7d, dev_amt_7d, dev_distinct_cust
  customer (7): cust_cnt_1h, cust_amt_1h, cust_cnt_24h, cust_amt_24h,
                cust_cnt_7d, cust_amt_7d, cust_new_device_flag
  merchant (4): merch_cnt_24h, merch_amt_24h, merch_cnt_7d, merch_amt_7d
  recency (2):  dev_hrs_since_last, cust_hrs_since_last
  amount ctx(4):cust_amt_z, dev_amt_z, cust_amt_ratio_mean, dev_amt_ratio_mean
  time (4):     is_night, hour, dow, is_weekend
  geo (2):      cust_geo_dist_km, cust_geo_vel_kmh
  novelty (1):  cust_new_merchant_flag
  velocity (1): dev_cnt_24h_7d_ratio
  raw (1):      amount
  categorical(2): channel, auth_method

All windowed features are trailing-only (no leakage): they use transactions
strictly before the current one within each entity group.
"""
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import RAW, FEATURES, N_TXN, N_FEATURES

FEATURE_NAMES = [
    "dev_cnt_1h", "dev_amt_1h", "dev_cnt_24h", "dev_amt_24h",
    "dev_cnt_7d", "dev_amt_7d", "dev_distinct_cust",
    "cust_cnt_1h", "cust_amt_1h", "cust_cnt_24h", "cust_amt_24h",
    "cust_cnt_7d", "cust_amt_7d", "cust_new_device_flag",
    "merch_cnt_24h", "merch_amt_24h", "merch_cnt_7d", "merch_amt_7d",
    "dev_hrs_since_last", "cust_hrs_since_last",
    "cust_amt_z", "dev_amt_z", "cust_amt_ratio_mean", "dev_amt_ratio_mean",
    "is_night", "hour", "dow", "is_weekend",
    "cust_geo_dist_km", "cust_geo_vel_kmh",
    "cust_new_merchant_flag",
    "dev_cnt_24h_7d_ratio",
    "amount", "channel", "auth_method",
]
assert len(FEATURE_NAMES) == N_FEATURES == 35

SEC_1H, SEC_24H, SEC_7D = 3600, 86400, 604800


def _trailing(t: np.ndarray, a: np.ndarray, window: int):
    """Trailing count and amount over `window` seconds (excludes current)."""
    n = t.size
    left = np.searchsorted(t, t - window)
    idx = np.arange(n)
    cnt = (idx - left).astype(np.float32)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(a, out=cs[1:])
    amt = (cs[idx] - cs[left]).astype(np.float32)
    return cnt, amt


def _expanding_z(a: np.ndarray):
    """z-score vs expanding history (excludes current row)."""
    n = a.size
    cs = np.empty(n + 1, dtype=np.float64); cs[0] = 0.0
    cs2 = np.empty(n + 1, dtype=np.float64); cs2[0] = 0.0
    np.cumsum(a, out=cs[1:]); np.cumsum(a * a, out=cs2[1:])
    k = np.arange(n, dtype=np.float64)
    mean = np.divide(cs[:-1], np.maximum(k, 1))
    var = np.divide(cs2[:-1], np.maximum(k, 1)) - mean ** 2
    std = np.sqrt(np.maximum(var, 0)) + 1e-6
    z = np.zeros(n, dtype=np.float32)
    z[1:] = ((a[1:] - mean[1:]) / std[1:]).astype(np.float32)
    ratio = np.ones(n, dtype=np.float32)
    ratio[1:] = (a[1:] / (mean[1:] + 1e-6)).astype(np.float32)
    return z, ratio


def _first_occ_flag(x: np.ndarray):
    """1 where value appears for the first time in the group."""
    _, first = np.unique(x, return_index=True)
    flag = np.zeros(x.size, dtype=np.float32)
    flag[first] = 1.0
    return flag


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians,
                                 (lat1.astype(np.float64), lon1.astype(np.float64),
                                  lat2.astype(np.float64), lon2.astype(np.float64)))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def _group_bounds(e_o: np.ndarray):
    """Start indices of each group in entity-sorted array."""
    ch = np.nonzero(np.diff(e_o) != 0)[0] + 1
    starts = np.empty(ch.size + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ch
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = e_o.size
    return starts, ends


def compute_for_entity(ent: np.ndarray, ts: np.ndarray, amt: np.ndarray,
                       aux: dict, kind: str):
    """Compute entity-level features. Returns dict of full-length float32 arrays
    indexed by the caller's row order (ts-sorted)."""
    n = ts.size
    order = np.argsort(ent, kind="stable")
    t = ts[order].astype(np.int64)
    a = amt[order].astype(np.float64)
    starts, ends = _group_bounds(ent[order])
    F = {}

    def put(name, vals):
        arr = np.zeros(n, dtype=np.float32)
        arr[order] = vals.astype(np.float32)
        F[name] = arr

    if kind == "device":
        c1h = np.zeros(n, dtype=np.float32); a1h = np.zeros(n, dtype=np.float32)
        c24 = np.zeros(n, dtype=np.float32); a24 = np.zeros(n, dtype=np.float32)
        c7d = np.zeros(n, dtype=np.float32); a7d = np.zeros(n, dtype=np.float32)
        dc = np.zeros(n, dtype=np.float32); tsl = np.zeros(n, dtype=np.float32)
        dz = np.zeros(n, dtype=np.float32); dr = np.ones(n, dtype=np.float32)
        cust_o = aux["customer_id"][order]
        for s, e in zip(starts, ends):
            tg, ag = t[s:e], a[s:e]
            c, v = _trailing(tg, ag, SEC_1H);  c1h[s:e], a1h[s:e] = c, v
            c, v = _trailing(tg, ag, SEC_24H); c24[s:e], a24[s:e] = c, v
            c, v = _trailing(tg, ag, SEC_7D);  c7d[s:e], a7d[s:e] = c, v
            dc[s:e] = np.cumsum(_first_occ_flag(cust_o[s:e]))
            gap = np.empty(tg.size, dtype=np.float32)
            gap[0] = 720.0
            gap[1:] = np.diff(tg) / 3600.0
            tsl[s:e] = gap
            z, r = _expanding_z(ag); dz[s:e], dr[s:e] = z, r
        for name, arr in [("dev_cnt_1h", c1h), ("dev_amt_1h", a1h),
                          ("dev_cnt_24h", c24), ("dev_amt_24h", a24),
                          ("dev_cnt_7d", c7d), ("dev_amt_7d", a7d),
                          ("dev_distinct_cust", dc), ("dev_hrs_since_last", tsl),
                          ("dev_amt_z", dz), ("dev_amt_ratio_mean", dr)]:
            put(name, arr)
    elif kind == "customer":
        c1h = np.zeros(n, dtype=np.float32); a1h = np.zeros(n, dtype=np.float32)
        c24 = np.zeros(n, dtype=np.float32); a24 = np.zeros(n, dtype=np.float32)
        c7d = np.zeros(n, dtype=np.float32); a7d = np.zeros(n, dtype=np.float32)
        ndf = np.zeros(n, dtype=np.float32); nmf = np.zeros(n, dtype=np.float32)
        tsl = np.zeros(n, dtype=np.float32); cz = np.zeros(n, dtype=np.float32)
        cr = np.ones(n, dtype=np.float32)
        gdist = np.zeros(n, dtype=np.float32); gvel = np.zeros(n, dtype=np.float32)
        dev_o = aux["device_id"][order]; mer_o = aux["merchant_id"][order]
        lat_o = aux["lat"][order]; lon_o = aux["lon"][order]
        for s, e in zip(starts, ends):
            tg, ag = t[s:e], a[s:e]
            c, v = _trailing(tg, ag, SEC_1H);  c1h[s:e], a1h[s:e] = c, v
            c, v = _trailing(tg, ag, SEC_24H); c24[s:e], a24[s:e] = c, v
            c, v = _trailing(tg, ag, SEC_7D);  c7d[s:e], a7d[s:e] = c, v
            ndf[s:e] = _first_occ_flag(dev_o[s:e])
            nmf[s:e] = _first_occ_flag(mer_o[s:e])
            gap = np.empty(tg.size, dtype=np.float32)
            gap[0] = 720.0
            gap[1:] = np.diff(tg) / 3600.0
            tsl[s:e] = gap
            z, r = _expanding_z(ag); cz[s:e], cr[s:e] = z, r
            la, lo = lat_o[s:e], lon_o[s:e]
            d = _haversine_km(la[:-1], lo[:-1], la[1:], lo[1:]).astype(np.float32)
            dg = np.zeros(tg.size, dtype=np.float32); dg[1:] = d
            gdist[s:e] = dg
            hrs = np.maximum(gap, 1e-3)
            gvel[s:e] = (dg / hrs).astype(np.float32)
        for name, arr in [("cust_cnt_1h", c1h), ("cust_amt_1h", a1h),
                          ("cust_cnt_24h", c24), ("cust_amt_24h", a24),
                          ("cust_cnt_7d", c7d), ("cust_amt_7d", a7d),
                          ("cust_new_device_flag", ndf),
                          ("cust_new_merchant_flag", nmf),
                          ("cust_hrs_since_last", tsl),
                          ("cust_amt_z", cz), ("cust_amt_ratio_mean", cr),
                          ("cust_geo_dist_km", gdist), ("cust_geo_vel_kmh", gvel)]:
            put(name, arr)
    elif kind == "merchant":
        c24 = np.zeros(n, dtype=np.float32); a24 = np.zeros(n, dtype=np.float32)
        c7d = np.zeros(n, dtype=np.float32); a7d = np.zeros(n, dtype=np.float32)
        for s, e in zip(starts, ends):
            tg, ag = t[s:e], a[s:e]
            c, v = _trailing(tg, ag, SEC_24H); c24[s:e], a24[s:e] = c, v
            c, v = _trailing(tg, ag, SEC_7D);  c7d[s:e], a7d[s:e] = c, v
        for name, arr in [("merch_cnt_24h", c24), ("merch_amt_24h", a24),
                          ("merch_cnt_7d", c7d), ("merch_amt_7d", a7d)]:
            put(name, arr)
    return F


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)
    parts = sorted(RAW.glob("part-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert len(df) == N_TXN, f"expected {N_TXN}, got {len(df)}"
    df.sort_values("ts", kind="mergesort", inplace=True, ignore_index=True)
    print(f"loaded {len(df):,} rows, ts-sorted", flush=True)

    ts = df["ts"].to_numpy()
    amt = df["amount"].to_numpy().astype(np.float32)
    aux = {c: df[c].to_numpy() for c in
           ["customer_id", "device_id", "merchant_id", "lat", "lon"]}

    allF = {}
    for kind, col in [("device", "device_id"), ("customer", "customer_id"),
                      ("merchant", "merchant_id")]:
        F = compute_for_entity(df[col].to_numpy(), ts, amt, aux, kind)
        allF.update(F)
        print(f"{kind}: {len(F)} features done", flush=True)
        gc.collect()

    # row-level features (vectorized)
    tod_h = (ts % 86400) // 3600
    allF["is_night"] = ((tod_h <= 5)).astype(np.float32)
    allF["hour"] = tod_h.astype(np.float32)
    allF["dow"] = ((ts // 86400 + 3) % 7).astype(np.float32)
    allF["is_weekend"] = (allF["dow"] >= 5).astype(np.float32)
    allF["dev_cnt_24h_7d_ratio"] = (
        allF["dev_cnt_24h"] / (allF["dev_cnt_7d"] + 1.0)).astype(np.float32)
    allF["amount"] = amt
    allF["channel"] = df["channel"].to_numpy().astype(np.float32)
    allF["auth_method"] = df["auth_method"].to_numpy().astype(np.float32)

    assert sorted(allF) == sorted(FEATURE_NAMES), "feature set mismatch"
    assert len(allF) == N_FEATURES

    # write chunked parquet in ts-sorted row order (positional alignment)
    n_chunks = 10
    idx = np.array_split(np.arange(len(df)), n_chunks)
    for c, ii in enumerate(idx):
        d = {k: allF[k][ii] for k in FEATURE_NAMES}
        pq.write_table(pa.Table.from_pydict(d),
                       FEATURES / f"feat-{c:03d}.parquet")
    print(f"wrote {n_chunks} feature parts, {len(FEATURE_NAMES)} features", flush=True)

    # persist the ts sort permutation for downstream joins
    np.save(FEATURES / "ts_order.npy", df["txn_id"].to_numpy())


if __name__ == "__main__":
    main()
