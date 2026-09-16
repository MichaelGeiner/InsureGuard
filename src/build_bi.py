"""
InsureGuard | Step 4: build the BI reporting layer (schema `bi`).

Creates the dashboard views from sql/04_bi_views.sql and loads the Step 3
test-period metrics (reports/model_metrics.json) into two small tables.

Usage:
  python src/build_bi.py
"""
from __future__ import annotations

import json

from db import ROOT, connect, run_sql_file

MODEL_LABELS = {
    "rule: amount >= $500": ("Rule: amount >= $500", 1),
    "isolation_forest": ("Isolation Forest", 2),
    "xgboost": ("XGBoost", 3),
    "insureguard_blend": ("InsureGuard (blend)", 4),
}


def main() -> None:
    metrics = json.loads((ROOT / "reports" / "model_metrics.json").read_text())

    with connect() as conn:
        run_sql_file(conn, "04_bi_views.sql")
        conn.execute("""
            DROP TABLE IF EXISTS bi.model_performance, bi.scenario_recall;
            CREATE TABLE bi.model_performance (
                model                 VARCHAR(30) PRIMARY KEY,
                sort_order            SMALLINT,
                roc_auc               NUMERIC(6, 4),
                pr_auc                NUMERIC(6, 4),
                flagged_txns          INTEGER,
                precision_rate        NUMERIC(6, 4),
                recall_rate           NUMERIC(6, 4),
                false_positive_rate   NUMERIC(6, 4),
                fraud_dollars         NUMERIC(12, 2),
                fraud_dollars_caught  NUMERIC(12, 2),
                dollar_recall         NUMERIC(6, 4)
            );
            CREATE TABLE bi.scenario_recall (
                model        VARCHAR(30),
                sort_order   SMALLINT,
                fraud_type   VARCHAR(30),
                recall_rate  NUMERIC(6, 4),
                PRIMARY KEY (model, fraud_type)
            );
        """)
        for r in metrics["test_results"]:
            label, order = MODEL_LABELS[r["model"]]
            conn.execute(
                "INSERT INTO bi.model_performance VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (label, order, r["roc_auc"], r["pr_auc"], r["flagged_txns"], r["precision"], r["recall"],
                 r["false_positive_rate"], r["fraud_dollars"], r["fraud_dollars_caught"], r["dollar_recall"]))
            for scenario, recall in r["recall_by_scenario"].items():
                conn.execute("INSERT INTO bi.scenario_recall VALUES (%s,%s,%s,%s)",
                             (label, order, scenario.replace("_", " ").title(), recall))

        summary = conn.execute("""
            SELECT model_period, outcome, COUNT(*), SUM(amount)
            FROM bi.fact_transactions GROUP BY 1, 2 ORDER BY 1, 2
        """).fetchall()

    print("bi schema ready: fact_transactions, dim_customer, dim_date, model_performance, scenario_recall\n")
    print(f"{'period':<22}{'outcome':<17}{'txns':>8}{'amount':>14}")
    for period, outcome, n, amount in summary:
        print(f"{period:<22}{outcome:<17}{n:>8,}{amount:>14,.2f}")


if __name__ == "__main__":
    main()
