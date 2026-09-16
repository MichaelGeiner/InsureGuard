# InsureGuard Executive Dashboard Spec

Three-page Power BI report built on the `bi` schema in PostgreSQL.

## Data model

| Table | Type | Relationship |
|---|---|---|
| `fact_transactions` | fact, 100,000 rows | many-to-one to both dimensions |
| `dim_customer` | dimension | `dim_customer[customer_id]` 1 to * `fact_transactions[customer_id]` |
| `dim_date` | dimension (mark as date table) | `dim_date[date]` 1 to * `fact_transactions[txn_date]` |
| `model_performance` | standalone | none (static test-period metrics) |
| `scenario_recall` | standalone | none |
| `_Measures` | measure container | none |

Sort-by-column settings: `risk_tier` by `risk_tier_order`, `month_short` by `month_number`, `day_name` by `day_of_week_number`, `model` by `sort_order`.

Color semantics (used everywhere): Fraud Prevented `#13837A` green, Fraud Missed `#D64545` red, False Alarm `#E3A33B` amber, Cleared `#9AA5B1` gray. Risk tiers: Low gray, Medium amber, High orange `#E07B39`, Critical red.

## Page 1: Executive Overview

Page filter: `model_period` = Live (May to Jun) (executives see out-of-sample results only).

| Zone | Visual | Fields |
|---|---|---|
| Header | Text box + card | "InsureGuard Fraud Risk Overview", `[Report Subtitle]` |
| KPI row | 6 cards | `[Fraud Exposure]`, `[Fraud Prevented]`, `[Prevention Rate]`, `[Flagged Alerts]`, `[Alert Precision]`, `[False Positive Rate]` |
| Left middle | Stacked column | X `dim_date[week_start]`, Y `[Fraud Prevented]` + `[Fraud Missed]` ("Weekly fraud dollars: prevented vs missed") |
| Right middle | Stacked bar | Y `fraud_type` (exclude Legitimate), X `[Fraud Exposure]`, legend `outcome` ("Fraud exposure by attack type") |
| Left bottom | Filled map or bar | `state`, `[Fraud Exposure]` ("Where fraud happens") |
| Right bottom | Donut | legend `risk_tier`, values `[Flagged Alerts]`, filter tier in High, Critical |
| Slicer panel | Slicers | `month_short`, `channel`, `dim_customer[customer_segment]`, `merchant_category` |

## Page 2: Alert Investigation

Page filter: `risk_tier` in Medium, High, Critical. Drillthrough target on `fraud_type` and `state`.

| Zone | Visual | Fields |
|---|---|---|
| KPI row | 4 cards | `[Flagged Alerts]`, `[Critical Alerts]`, `[Alert Precision]`, `[Fraud Prevented per Alert]` |
| Top left | Scatter | X `amount`, Y `anomaly_risk_score`, legend `outcome`, details `transaction_id` |
| Top right | Column | X `txn_hour`, Y `[Flagged Alerts]` ("When alerts fire") |
| Bottom | Table (alert queue) | `transaction_id`, `txn_timestamp`, `customer_id`, `amount`, `merchant_category`, `city`, `anomaly_risk_score` (data bars), `risk_tier` (background color rules), `outcome`, `txn_count_1h`, `amount_to_avg_30d_ratio`, `geo_velocity_kmh`, `device_age_hours`; sorted by score descending |
| Slicers | Slicers | `risk_tier` (buttons), `anomaly_risk_score` (range), `model_period`, `state` |

## Page 3: Model Performance

| Zone | Visual | Fields |
|---|---|---|
| Top left | Clustered bar | `model_performance[model]`, `dollar_recall`, `precision_rate` ("Model vs simple rule") |
| Top right | Matrix | rows `scenario_recall[fraud_type]`, columns `model`, values `recall_rate` (color scale red to green) |
| Bottom left | Matrix (confusion matrix) | rows `is_fraud`, columns `is_flagged`, values `[Total Transactions]` |
| Bottom middle | Clustered column | X `score_band`, Y `[Total Transactions]`, legend `is_fraud`, log scale Y ("Risk score separates fraud") |
| Bottom right | Text box | Method notes: out-of-time validation (train Jan to Mar, validate Apr, test May to Jun), class weighting over SMOTE, friendly fraud limitation |

## Expected headline numbers (Live period)

| Measure | Value |
|---|---:|
| Fraud Exposure | $41,998 |
| Fraud Prevented | $31,711 |
| Prevention Rate | 75.5% |
| Flagged Alerts | 361 |
| Alert Precision | 85.0% |
| False Positive Rate | 0.16% |

If your cards show these numbers, the model and relationships are correct.
