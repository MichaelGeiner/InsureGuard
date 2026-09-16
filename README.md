# InsureGuard

**End-to-end financial fraud and anomaly detection pipeline:** synthetic transaction data, PostgreSQL feature engineering, machine learning risk scoring, and an executive fraud exposure dashboard.

> Status: Step 2 of 5 complete (data, database, SQL features). Built in public, step by step.

## Pipeline

```
generate_data.py  ->  PostgreSQL (star schema)  ->  SQL feature engineering  ->  ML risk scoring  ->  Power BI dashboard
     Step 1              Step 1                        Step 2                       Step 3              Step 4
```

## Step 1: Data and schema

Public fraud datasets either anonymize away the fields fraud analysts actually use (Kaggle's credit card set is PCA components only) or lack a reliable customer key. InsureGuard instead simulates **100,000 transactions across 5,000 customers and 8,470 devices** over six months, with per-customer behavioral baselines (home city, spend level, diurnal pattern, primary device).

Fraud (**1.2% of rows, $165K**) is injected as incidents with distinct, realistic signatures:

| Scenario | Signature | Rows | Avg amount |
|---|---|---:|---:|
| Card testing | 5 to 12 micro-charges under $5 within 25 minutes, new device | 660 | $2.70 |
| Account takeover | 3 to 6 high-value online purchases, new device, distant city | 429 | $262.93 |
| High-value night | 10 to 25x normal spend between 1 and 4am, customer's own device | 65 | $681.00 |
| Impossible travel | In-store purchase 1,500+ km from a home purchase minutes earlier | 47 | $131.89 |

Key insight: the median fraudulent transaction ($4.46) is *smaller* than the median legitimate one ($39.72). Amount thresholds alone miss most fraud, which is why the pipeline relies on velocity and behavioral features.

### Schema

| Table | Grain | Key columns |
|---|---|---|
| `insureguard.dim_customer` | one row per account holder | `customer_id`, home location, segment, signup date |
| `insureguard.dim_device` | one row per device fingerprint | `device_id`, `customer_id`, device type, first seen |
| `insureguard.fact_transaction` | one row per transaction | `transaction_id`, `customer_id`, `device_id`, `txn_timestamp`, `amount`, category, channel, lat/lon, `is_fraud` |

DDL is PostgreSQL and Snowflake compatible. A composite index on `(customer_id, txn_timestamp)` supports the window functions used for feature engineering.

## Step 2: SQL feature engineering

[`sql/02_features.sql`](sql/02_features.sql) builds `insureguard.txn_features` (100,000 rows in about 2.5 seconds) using PostgreSQL window functions with time-based `RANGE` frames. Every feature uses only the customer's history up to that transaction, so nothing from the future leaks into the model.

| Family | Features | Technique |
|---|---|---|
| Rolling spend | `avg_amount_7d`, `avg_amount_30d`, `amount_to_avg_7d_ratio`, `amount_to_avg_30d_ratio` | `RANGE BETWEEN INTERVAL '30 days' PRECEDING`, current row excluded from the baseline |
| Velocity | `txn_count_1h`, `txn_count_24h`, `amount_sum_1h`, `amount_sum_24h` | sliding time windows per customer |
| Geographic jump | `km_from_prev_txn`, `minutes_since_prev_txn`, `geo_velocity_kmh`, `km_from_home` | `LAG()` + haversine distance in SQL |
| Device and context | `device_age_hours`, `hour_of_day`, `day_of_week`, `is_night`, `is_online` | first-seen device timestamp per account |

### Validation: each fraud pattern separates on its target feature

Medians per scenario from [`sql/03_feature_validation.sql`](sql/03_feature_validation.sql):

| Scenario | Spend vs 30d avg | Txns in 1h | Geo velocity (km/h) | Device age (hours) | Night share |
|---|---:|---:|---:|---:|---:|
| Legitimate | 0.75x | 1 | 0.1 | 1,824 | 2% |
| Card testing | 0.14x | **5** | 158.5 | **0.16** | 3% |
| Account takeover | 1.64x | 2 | 24.6 | **0.36** | 1% |
| High-value night | **6.38x** | 1 | 0.1 | 993 | **92%** |
| Impossible travel | 2.43x | 1 | **2,603** | **0.00** | 6% |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # then set your PostgreSQL password
python src/generate_data.py     # Step 1: writes data/raw/*.csv
python src/load_to_postgres.py  # Step 1: schema + bulk load
python src/build_features.py    # Step 2: feature table + validation report
```

## Project structure

```
InsureGuard/
├── sql/
│   ├── 01_schema.sql              star schema DDL
│   ├── 02_features.sql            window-function feature engineering
│   └── 03_feature_validation.sql  per-scenario feature medians
└── src/
    ├── db.py                      shared connection helpers
    ├── generate_data.py           synthetic data generator
    ├── load_to_postgres.py        COPY-based bulk loader
    └── build_features.py          runs feature SQL + prints validation
```

## Tech stack

Python (pandas, NumPy), PostgreSQL 18, SQL window functions, scikit-learn, Power BI
