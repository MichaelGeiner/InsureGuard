-- =====================================================================
-- InsureGuard | 04_bi_views.sql
-- Reporting layer for Power BI / Tableau: a clean star schema in its own
-- schema (bi) with business-friendly labels, so the dashboard does no
-- data wrangling. Rebuild after every model run.
--
--   bi.fact_transactions   one row per transaction, scored and labeled
--   bi.dim_customer        customer attributes and age bands
--   bi.dim_date            calendar table for time intelligence
--   bi.model_performance   test-period metrics per model (table, from Python)
--   bi.scenario_recall     test-period recall per fraud scenario (table, from Python)
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS bi;

CREATE OR REPLACE VIEW bi.fact_transactions AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.device_id,
    t.txn_timestamp,
    t.txn_timestamp::DATE                                             AS txn_date,
    EXTRACT(HOUR FROM t.txn_timestamp)::SMALLINT                      AS txn_hour,
    t.amount,
    INITCAP(REPLACE(t.merchant_category, '_', ' '))                   AS merchant_category,
    CASE t.channel WHEN 'online' THEN 'Online' ELSE 'In-store' END    AS channel,
    t.city,
    t.state,
    t.latitude,
    t.longitude,
    INITCAP(REPLACE(d.device_type, '_', ' '))                         AS device_type,

    -- ground truth
    t.is_fraud,
    COALESCE(INITCAP(REPLACE(t.fraud_scenario, '_', ' ')), 'Legitimate') AS fraud_type,

    -- model output
    s.anomaly_risk_score,
    s.risk_tier,
    CASE s.risk_tier WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 ELSE 4 END AS risk_tier_order,
    (s.anomaly_risk_score / 10) * 10                                  AS score_band,  -- 0, 10, ... 100
    s.xgb_fraud_probability,
    s.isolation_anomaly_score,
    s.is_flagged,
    CASE
        WHEN s.is_flagged AND t.is_fraud       THEN 'Fraud Prevented'
        WHEN NOT s.is_flagged AND t.is_fraud   THEN 'Fraud Missed'
        WHEN s.is_flagged AND NOT t.is_fraud   THEN 'False Alarm'
        ELSE 'Cleared'
    END                                                               AS outcome,
    CASE WHEN s.dataset_split = 'test' THEN 'Live (May to Jun)' ELSE 'Training (Jan to Apr)' END AS model_period,
    s.model_version,

    -- risk drivers shown to analysts on the alert queue
    f.txn_count_1h,
    f.amount_to_avg_30d_ratio,
    f.geo_velocity_kmh,
    f.km_from_home,
    f.device_age_hours
FROM insureguard.fact_transaction AS t
JOIN insureguard.txn_risk_scores  AS s USING (transaction_id)
JOIN insureguard.txn_features     AS f USING (transaction_id)
JOIN insureguard.dim_device       AS d USING (device_id);

CREATE OR REPLACE VIEW bi.dim_customer AS
SELECT
    customer_id,
    home_city,
    home_state,
    INITCAP(customer_segment)                                         AS customer_segment,
    age,
    CASE
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 50 THEN '35-49'
        WHEN age < 65 THEN '50-64'
        ELSE '65+'
    END                                                               AS age_band,
    signup_date
FROM insureguard.dim_customer;

CREATE OR REPLACE VIEW bi.dim_date AS
SELECT
    d::DATE                                                           AS date,
    EXTRACT(YEAR FROM d)::SMALLINT                                    AS year,
    EXTRACT(MONTH FROM d)::SMALLINT                                   AS month_number,
    TO_CHAR(d, 'Mon')                                                 AS month_short,
    TO_CHAR(d, 'YYYY-MM')                                             AS year_month,
    DATE_TRUNC('week', d)::DATE                                       AS week_start,
    EXTRACT(ISODOW FROM d)::SMALLINT                                  AS day_of_week_number,
    TO_CHAR(d, 'Dy')                                                  AS day_name,
    EXTRACT(ISODOW FROM d) IN (6, 7)                                  AS is_weekend
FROM GENERATE_SERIES(DATE '2026-01-01', DATE '2026-06-30', INTERVAL '1 day') AS d;
