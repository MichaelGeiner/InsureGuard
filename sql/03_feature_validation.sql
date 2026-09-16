-- =====================================================================
-- InsureGuard | 03_feature_validation.sql
-- Sanity check: does each fraud scenario stand out on the feature
-- designed to catch it? Medians per scenario vs. legitimate traffic.
-- =====================================================================

SELECT
    COALESCE(t.fraud_scenario, 'legit')                                           AS scenario,
    COUNT(*)                                                                      AS txns,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.amount)                         AS med_amount,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.amount_to_avg_30d_ratio)        AS med_ratio_30d,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.txn_count_1h)                   AS med_count_1h,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.txn_count_24h)                  AS med_count_24h,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.geo_velocity_kmh)               AS med_geo_kmh,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.km_from_home)                   AS med_km_home,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.device_age_hours)               AS med_device_age_h,
    ROUND(AVG(CASE WHEN f.is_night THEN 1 ELSE 0 END), 3)                         AS night_share
FROM insureguard.txn_features AS f
JOIN insureguard.fact_transaction AS t USING (transaction_id)
GROUP BY 1
ORDER BY 2 DESC;
