-- =====================================================================
-- InsureGuard | 02_features.sql
-- Builds insureguard.txn_features: one row per transaction with
-- behavioral risk features computed ONLY from that customer's history
-- up to and including the transaction (no future data leaks in).
--
-- Feature families
--   1. Rolling spend      7d / 30d average of PRIOR amounts, deviation ratio
--   2. Velocity           txn count and spend in 1h / 24h sliding windows
--   3. Geographic jump    km and minutes since previous txn -> implied km/h
--   4. Device & context   device age on the account, distance from home, time of day
--
-- Written for PostgreSQL 14+. Porting to Snowflake:
--   * replace the named WINDOW clause with inline OVER (...) definitions
--   * EXTRACT(EPOCH FROM a - b)  ->  DATEDIFF('second', b, a)
--   * DROP + CREATE TABLE AS     ->  CREATE OR REPLACE TABLE ... AS
-- RANGE ... INTERVAL window frames work the same on both.
-- =====================================================================

DROP TABLE IF EXISTS insureguard.txn_features CASCADE;

CREATE TABLE insureguard.txn_features AS
WITH windowed AS (
    SELECT
        t.transaction_id,
        t.customer_id,
        t.device_id,
        t.txn_timestamp,
        t.amount,
        t.merchant_category,
        t.channel,
        t.latitude,
        t.longitude,
        t.is_fraud,

        -- Frames include the current row; it is subtracted out below so
        -- averages describe the customer's behavior BEFORE this transaction.
        SUM(t.amount) OVER w_7d   AS sum_7d,
        COUNT(*)      OVER w_7d   AS cnt_7d,
        SUM(t.amount) OVER w_30d  AS sum_30d,
        COUNT(*)      OVER w_30d  AS cnt_30d,

        COUNT(*)      OVER w_1h   AS cnt_1h,
        SUM(t.amount) OVER w_1h   AS sum_1h,
        COUNT(*)      OVER w_24h  AS cnt_24h,
        SUM(t.amount) OVER w_24h  AS sum_24h,

        ROW_NUMBER()  OVER w_seq  AS customer_txn_seq,
        LAG(t.txn_timestamp) OVER w_seq AS prev_timestamp,
        LAG(t.latitude)      OVER w_seq AS prev_latitude,
        LAG(t.longitude)     OVER w_seq AS prev_longitude,

        -- first time this device ever transacted on this account (always <= now)
        MIN(t.txn_timestamp) OVER (PARTITION BY t.customer_id, t.device_id) AS device_first_seen
    FROM insureguard.fact_transaction AS t
    WINDOW
        w_seq AS (PARTITION BY t.customer_id ORDER BY t.txn_timestamp, t.transaction_id),
        w_1h  AS (PARTITION BY t.customer_id ORDER BY t.txn_timestamp RANGE BETWEEN INTERVAL '1 hour'  PRECEDING AND CURRENT ROW),
        w_24h AS (PARTITION BY t.customer_id ORDER BY t.txn_timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW),
        w_7d  AS (PARTITION BY t.customer_id ORDER BY t.txn_timestamp RANGE BETWEEN INTERVAL '7 days'  PRECEDING AND CURRENT ROW),
        w_30d AS (PARTITION BY t.customer_id ORDER BY t.txn_timestamp RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW)
),

deltas AS (
    SELECT
        w.*,
        (w.sum_7d  - w.amount) / NULLIF(w.cnt_7d  - 1, 0) AS prior_avg_7d,
        (w.sum_30d - w.amount) / NULLIF(w.cnt_30d - 1, 0) AS prior_avg_30d,
        EXTRACT(EPOCH FROM (w.txn_timestamp - w.prev_timestamp)) / 3600.0 AS hours_since_prev,

        -- haversine great-circle distance (km) from the previous transaction
        2 * 6371 * ASIN(SQRT(
              POWER(SIN(RADIANS(w.latitude - w.prev_latitude) / 2), 2)
            + COS(RADIANS(w.prev_latitude)) * COS(RADIANS(w.latitude))
            * POWER(SIN(RADIANS(w.longitude - w.prev_longitude) / 2), 2)
        )) AS km_from_prev,

        -- haversine distance (km) from the customer's home address
        2 * 6371 * ASIN(SQRT(
              POWER(SIN(RADIANS(w.latitude - c.home_latitude) / 2), 2)
            + COS(RADIANS(c.home_latitude)) * COS(RADIANS(w.latitude))
            * POWER(SIN(RADIANS(w.longitude - c.home_longitude) / 2), 2)
        )) AS km_from_home
    FROM windowed AS w
    JOIN insureguard.dim_customer AS c USING (customer_id)
)

SELECT
    transaction_id,
    customer_id,
    txn_timestamp,
    amount,
    merchant_category,
    channel,
    customer_txn_seq,

    -- 1. rolling spend
    ROUND(prior_avg_7d, 2)                               AS avg_amount_7d,
    ROUND(prior_avg_30d, 2)                              AS avg_amount_30d,
    ROUND(amount / NULLIF(prior_avg_7d, 0), 3)           AS amount_to_avg_7d_ratio,
    ROUND(amount / NULLIF(prior_avg_30d, 0), 3)          AS amount_to_avg_30d_ratio,

    -- 2. velocity (windows include the current transaction)
    cnt_1h                                               AS txn_count_1h,
    cnt_24h                                              AS txn_count_24h,
    sum_1h                                               AS amount_sum_1h,
    sum_24h                                              AS amount_sum_24h,

    -- 3. geographic jump
    ROUND((hours_since_prev * 60)::NUMERIC, 1)           AS minutes_since_prev_txn,
    ROUND(km_from_prev::NUMERIC, 1)                      AS km_from_prev_txn,
    -- floor elapsed time at 1 minute so simultaneous txns do not divide by zero
    ROUND((km_from_prev / GREATEST(hours_since_prev, 1.0 / 60))::NUMERIC, 1) AS geo_velocity_kmh,
    ROUND(km_from_home::NUMERIC, 1)                      AS km_from_home,

    -- 4. device & context
    ROUND((EXTRACT(EPOCH FROM (txn_timestamp - device_first_seen)) / 3600.0)::NUMERIC, 2) AS device_age_hours,
    EXTRACT(HOUR FROM txn_timestamp)::SMALLINT           AS hour_of_day,
    EXTRACT(DOW FROM txn_timestamp)::SMALLINT            AS day_of_week,
    (EXTRACT(HOUR FROM txn_timestamp) < 5)               AS is_night,
    (channel = 'online')                                 AS is_online,

    is_fraud                                             -- label, for training/evaluation only
FROM deltas;

ALTER TABLE insureguard.txn_features ADD PRIMARY KEY (transaction_id);
CREATE INDEX ix_features_customer_time ON insureguard.txn_features (customer_id, txn_timestamp);
ANALYZE insureguard.txn_features;
