-- =====================================================================
-- Feature pipeline: Spark SQL port of src/feature_engineering.py
-- =====================================================================
-- The local build (src/feature_engineering.py, pandas/numpy) implements the
-- exact logic below. On a cluster this Spark SQL would run unchanged against
-- the `mobile_payments` Hive table (no JVM is available on the build
-- machine, so the local port is what produced the repo's features).
--
-- Conventions: every windowed aggregate is TRAILING-ONLY -- the frame
-- `ROWS BETWEEN ... PRECEDING AND 1 PRECEDING` excludes the current row, so
-- no label leakage. `ts` is epoch seconds.
-- =====================================================================

-- 1) entity-level trailing velocity features ----------------------------
CREATE OR REPLACE TEMP VIEW entity_features AS
SELECT
  txn_id, ts, customer_id, device_id, merchant_id, amount,

  -- device velocity: counts / amounts over 1h, 24h, 7d
  COUNT(*) OVER w_dev_1h  AS dev_cnt_1h,
  SUM(amount) OVER w_dev_1h  AS dev_amt_1h,
  COUNT(*) OVER w_dev_24h AS dev_cnt_24h,
  SUM(amount) OVER w_dev_24h AS dev_amt_24h,
  COUNT(*) OVER w_dev_7d  AS dev_cnt_7d,
  SUM(amount) OVER w_dev_7d  AS dev_amt_7d,
  -- distinct customers ever seen on this device (device-sharing signal)
  SIZE(COLLECT_SET(customer_id) OVER w_dev_all) AS dev_distinct_cust,

  -- customer velocity
  COUNT(*) OVER w_cust_1h  AS cust_cnt_1h,
  SUM(amount) OVER w_cust_1h  AS cust_amt_1h,
  COUNT(*) OVER w_cust_24h AS cust_cnt_24h,
  SUM(amount) OVER w_cust_24h AS cust_amt_24h,
  COUNT(*) OVER w_cust_7d  AS cust_cnt_7d,
  SUM(amount) OVER w_cust_7d  AS cust_amt_7d,

  -- merchant velocity
  COUNT(*) OVER w_merch_24h AS merch_cnt_24h,
  SUM(amount) OVER w_merch_24h AS merch_amt_24h,
  COUNT(*) OVER w_merch_7d  AS merch_cnt_7d,
  SUM(amount) OVER w_merch_7d  AS merch_amt_7d,

  -- recency
  (ts - LAG(ts) OVER w_dev_time) / 3600.0  AS dev_hrs_since_last,
  (ts - LAG(ts) OVER w_cust_time) / 3600.0 AS cust_hrs_since_last,

  -- novelty flags: first time this customer uses the device / merchant
  CASE WHEN ROW_NUMBER() OVER (PARTITION BY customer_id, device_id
                               ORDER BY ts) = 1 THEN 1.0 ELSE 0.0 END
    AS cust_new_device_flag,
  CASE WHEN ROW_NUMBER() OVER (PARTITION BY customer_id, merchant_id
                               ORDER BY ts) = 1 THEN 1.0 ELSE 0.0 END
    AS cust_new_merchant_flag
FROM mobile_payments
WINDOW
  w_dev_1h  AS (PARTITION BY device_id   ORDER BY ts RANGE BETWEEN 3600   PRECEDING AND 1 PRECEDING),
  w_dev_24h AS (PARTITION BY device_id   ORDER BY ts RANGE BETWEEN 86400  PRECEDING AND 1 PRECEDING),
  w_dev_7d  AS (PARTITION BY device_id   ORDER BY ts RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
  w_dev_all AS (PARTITION BY device_id   ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
  w_cust_1h  AS (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN 3600   PRECEDING AND 1 PRECEDING),
  w_cust_24h AS (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN 86400  PRECEDING AND 1 PRECEDING),
  w_cust_7d  AS (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
  w_merch_24h AS (PARTITION BY merchant_id ORDER BY ts RANGE BETWEEN 86400  PRECEDING AND 1 PRECEDING),
  w_merch_7d  AS (PARTITION BY merchant_id ORDER BY ts RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
  w_dev_time  AS (PARTITION BY device_id   ORDER BY ts),
  w_cust_time AS (PARTITION BY customer_id ORDER BY ts);

-- 2) amount context: expanding z-score and ratio-to-mean ------------------
-- (expanding window, excludes current row -- mirrors _expanding_z)
CREATE OR REPLACE TEMP VIEW amount_context AS
SELECT
  txn_id,
  (amount - AVG(amount) OVER w_cust_exp)
    / (STDDEV_SAMP(amount) OVER w_cust_exp + 1e-6) AS cust_amt_z,
  amount / NULLIF(AVG(amount) OVER w_cust_exp, 0)   AS cust_amt_ratio_mean,
  (amount - AVG(amount) OVER w_dev_exp)
    / (STDDEV_SAMP(amount) OVER w_dev_exp + 1e-6)  AS dev_amt_z,
  amount / NULLIF(AVG(amount) OVER w_dev_exp, 0)   AS dev_amt_ratio_mean
FROM mobile_payments
WINDOW
  w_cust_exp AS (PARTITION BY customer_id ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
  w_dev_exp  AS (PARTITION BY device_id   ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- 3) geo velocity: distance from previous txn / elapsed hours -------------
CREATE OR REPLACE TEMP VIEW geo_features AS
SELECT
  txn_id,
  6371.0 * 2 * ASIN(SQRT(
      POWER(SIN(RADIANS(lat - LAG(lat) OVER w) / 2), 2)
    + COS(RADIANS(LAG(lat) OVER w)) * COS(RADIANS(lat))
      * POWER(SIN(RADIANS(lon - LAG(lon) OVER w) / 2), 2)
  )) AS cust_geo_dist_km,
  6371.0 * 2 * ASIN(SQRT(
      POWER(SIN(RADIANS(lat - LAG(lat) OVER w) / 2), 2)
    + COS(RADIANS(LAG(lat) OVER w)) * COS(RADIANS(lat))
      * POWER(SIN(RADIANS(lon - LAG(lon) OVER w) / 2), 2)
  )) / GREATEST((ts - LAG(ts) OVER w) / 3600.0, 1e-3) AS cust_geo_vel_kmh
FROM mobile_payments
WINDOW w AS (PARTITION BY customer_id ORDER BY ts);

-- 4) final feature mart: 35 features ---------------------------------------
CREATE OR REPLACE TABLE fraud_feature_mart AS
SELECT
  p.txn_id, p.ts, p.is_fraud,
  e.dev_cnt_1h, e.dev_amt_1h, e.dev_cnt_24h, e.dev_amt_24h,
  e.dev_cnt_7d, e.dev_amt_7d, e.dev_distinct_cust,
  e.cust_cnt_1h, e.cust_amt_1h, e.cust_cnt_24h, e.cust_amt_24h,
  e.cust_cnt_7d, e.cust_amt_7d, e.cust_new_device_flag,
  e.merch_cnt_24h, e.merch_amt_24h, e.merch_cnt_7d, e.merch_amt_7d,
  COALESCE(e.dev_hrs_since_last, 720.0)  AS dev_hrs_since_last,
  COALESCE(e.cust_hrs_since_last, 720.0) AS cust_hrs_since_last,
  COALESCE(a.cust_amt_z, 0.0)          AS cust_amt_z,
  COALESCE(a.dev_amt_z, 0.0)           AS dev_amt_z,
  COALESCE(a.cust_amt_ratio_mean, 1.0) AS cust_amt_ratio_mean,
  COALESCE(a.dev_amt_ratio_mean, 1.0)  AS dev_amt_ratio_mean,
  CASE WHEN HOUR(FROM_UNIXTIME(p.ts)) BETWEEN 0 AND 5 THEN 1.0 ELSE 0.0 END AS is_night,
  HOUR(FROM_UNIXTIME(p.ts))            AS hour,
  (DAYOFWEEK(FROM_UNIXTIME(p.ts)) + 5) % 7 AS dow,          -- Monday = 0
  CASE WHEN (DAYOFWEEK(FROM_UNIXTIME(p.ts)) + 5) % 7 >= 5 THEN 1.0 ELSE 0.0 END AS is_weekend,
  COALESCE(g.cust_geo_dist_km, 0.0)  AS cust_geo_dist_km,
  COALESCE(g.cust_geo_vel_kmh, 0.0)  AS cust_geo_vel_kmh,
  e.cust_new_merchant_flag,
  e.dev_cnt_24h / (e.dev_cnt_7d + 1.0) AS dev_cnt_24h_7d_ratio,
  p.amount, p.channel, p.auth_method
FROM mobile_payments p
LEFT JOIN entity_features e USING (txn_id)
LEFT JOIN amount_context a USING (txn_id)
LEFT JOIN geo_features g   USING (txn_id);
