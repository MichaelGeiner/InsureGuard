"""
InsureGuard | Step 5: export a compact data file for the web dashboard (docs/).

Writes docs/data/insureguard.json:
  * live-period transactions (May to Jun) as dictionary-encoded columns, so the
    browser can re-aggregate every chart when filters change
  * the reviewable alert queue (risk score 30+) with the drivers behind each flag
  * test-period model metrics from reports/model_metrics.json

Usage:
  python src/export_web_data.py
"""
from __future__ import annotations

import json

from db import ROOT, connect  # must come first: makes libpq findable

import pandas as pd  # noqa: E402

OUT = ROOT / "docs" / "data" / "insureguard.json"

QUERY = """
    SELECT f.transaction_id, f.customer_id, f.txn_timestamp, f.txn_date, f.txn_hour, f.amount,
           f.merchant_category, f.channel, f.state, f.city, c.customer_segment,
           f.fraud_type, f.outcome, f.risk_tier, f.anomaly_risk_score,
           f.is_fraud, f.is_flagged, d.week_start,
           f.txn_count_1h, f.amount_to_avg_30d_ratio, f.geo_velocity_kmh, f.device_age_hours
    FROM bi.fact_transactions AS f
    JOIN bi.dim_customer AS c USING (customer_id)
    JOIN bi.dim_date AS d ON d.date = f.txn_date
    WHERE f.model_period = 'Live (May to Jun)'
    ORDER BY f.txn_timestamp, f.transaction_id
"""

DIMENSIONS = ["merchant_category", "channel", "state", "customer_segment", "fraud_type", "outcome", "risk_tier"]


def encode(series: pd.Series) -> tuple[list[str], list[int]]:
    values = sorted(series.unique())
    index = {v: i for i, v in enumerate(values)}
    return values, [index[v] for v in series]


def main() -> None:
    with connect() as conn:
        cur = conn.execute(QUERY)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

    weeks = sorted(df["week_start"].unique())
    columns = {
        "week": [weeks.index(w) for w in df["week_start"]],
        "hour": df["txn_hour"].astype(int).tolist(),
        "cents": (df["amount"].astype(float) * 100).round().astype(int).tolist(),
        "score": df["anomaly_risk_score"].astype(int).tolist(),
        "fraud": df["is_fraud"].astype(int).tolist(),
        "flagged": df["is_flagged"].astype(int).tolist(),
    }
    dictionaries = {}
    for dim in DIMENSIONS:
        dictionaries[dim], columns[dim] = encode(df[dim])

    alerts = df[df["anomaly_risk_score"] >= 30].sort_values("anomaly_risk_score", ascending=False)
    alert_rows = [[
        r.transaction_id, r.customer_id, r.txn_timestamp.strftime("%Y-%m-%d %H:%M"), r.city,
        None if pd.isna(r.txn_count_1h) else int(r.txn_count_1h),
        None if pd.isna(r.amount_to_avg_30d_ratio) else round(float(r.amount_to_avg_30d_ratio), 1),
        None if pd.isna(r.geo_velocity_kmh) else round(float(r.geo_velocity_kmh)),
        None if pd.isna(r.device_age_hours) else round(float(r.device_age_hours), 1),
        int(i),
    ] for i, r in zip(alerts.index, alerts.itertuples())]

    metrics = json.loads((ROOT / "reports" / "model_metrics.json").read_text())
    payload = {
        "generated_from": "bi.fact_transactions (Live period, May to Jun 2026)",
        "rows": len(df),
        "weeks": [str(w) for w in weeks],
        "dictionaries": dictionaries,
        "columns": columns,
        "alert_fields": ["transaction_id", "customer_id", "timestamp", "city", "txn_count_1h",
                         "spend_vs_30d_avg", "geo_speed_kmh", "device_age_hours", "row"],
        "alerts": alert_rows,
        "model": {k: metrics[k] for k in ("model_version", "split", "flag_threshold", "blend", "test_results")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} transactions, {len(alert_rows):,} alerts, "
          f"{OUT.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
