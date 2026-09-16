# InsureGuard

**End-to-end financial fraud and anomaly detection pipeline:** synthetic transaction data, PostgreSQL feature engineering, machine learning risk scoring, and an executive fraud exposure dashboard.

> Status: Step 1 of 5 complete (data + database). Built in public, step by step.

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

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then set your PostgreSQL password
python src/generate_data.py   # writes data/raw/*.csv
python src/load_to_postgres.py
```

## Tech stack

Python (pandas, NumPy), PostgreSQL 18, SQL window functions, scikit-learn, Power BI
