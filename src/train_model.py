"""
InsureGuard | Step 3: anomaly detection + fraud risk scoring.

Pipeline
  1. load       read insureguard.txn_features from PostgreSQL
  2. prepare    impute cold-start gaps, one-hot merchant category
  3. split      out-of-time: train Jan-Mar, validate Apr, test May-Jun
  4. models     Isolation Forest (unsupervised, no labels)
                XGBoost (supervised, class-weighted for 1.2% fraud rate)
  5. score      Anomaly Risk Score 0-100 = 80% XGBoost probability + 20% Isolation Forest
  6. evaluate   test-period precision / recall / dollars caught vs a simple rule baseline
  7. publish    insureguard.txn_risk_scores table + reports/ + models/

Usage:
  python src/train_model.py
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

from db import ROOT, connect  # must come first: makes libpq findable

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

MODEL_VERSION = "v1.0"
SEED = 42
VALID_START = pd.Timestamp("2026-04-01")
TEST_START = pd.Timestamp("2026-05-01")

# risk tiers used by the dashboard; "flagged" = routed to an analyst
TIERS = [(80, "Critical"), (60, "High"), (30, "Medium"), (0, "Low")]
FLAG_THRESHOLD = 60
BLEND_XGB_WEIGHT = 0.8

NUMERIC_FEATURES = [
    "amount", "customer_txn_seq",
    "avg_amount_7d", "avg_amount_30d", "amount_to_avg_7d_ratio", "amount_to_avg_30d_ratio",
    "txn_count_1h", "txn_count_24h", "amount_sum_1h", "amount_sum_24h",
    "minutes_since_prev_txn", "km_from_prev_txn", "geo_velocity_kmh", "km_from_home",
    "device_age_hours", "hour_of_day", "day_of_week", "is_night", "is_online",
]

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


# ---------------------------------------------------------------- 1. load
def load_features() -> pd.DataFrame:
    query = """
        SELECT f.*, t.fraud_scenario
        FROM insureguard.txn_features AS f
        JOIN insureguard.fact_transaction AS t USING (transaction_id)
        ORDER BY f.txn_timestamp, f.transaction_id
    """
    with connect() as conn:
        cur = conn.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    return df


# ------------------------------------------------------------- 2. prepare
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric model matrix. Missing values only occur when a customer has no prior history."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    X["has_history"] = X["avg_amount_30d"].notna().astype(float)

    X["avg_amount_7d"] = X["avg_amount_7d"].fillna(X["amount"])
    X["avg_amount_30d"] = X["avg_amount_30d"].fillna(X["amount"])
    X[["amount_to_avg_7d_ratio", "amount_to_avg_30d_ratio"]] = (
        X[["amount_to_avg_7d_ratio", "amount_to_avg_30d_ratio"]].fillna(1.0))
    X["minutes_since_prev_txn"] = X["minutes_since_prev_txn"].fillna(60 * 24 * 30)
    X[["km_from_prev_txn", "geo_velocity_kmh"]] = X[["km_from_prev_txn", "geo_velocity_kmh"]].fillna(0.0)

    categories = pd.get_dummies(df["merchant_category"], prefix="cat", dtype=float)
    return pd.concat([X, categories], axis=1)


# --------------------------------------------------------------- 3. split
def split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    ts = df["txn_timestamp"]
    return {
        "train": ts < VALID_START,
        "valid": (ts >= VALID_START) & (ts < TEST_START),
        "test": ts >= TEST_START,
    }


# -------------------------------------------------------------- 4. models
def train_isolation_forest(X_train: pd.DataFrame) -> tuple[IsolationForest, dict]:
    """Unsupervised: never sees labels, learns what 'normal' looks like."""
    model = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto",
                            random_state=SEED, n_jobs=-1).fit(X_train)
    raw = -model.score_samples(X_train)  # higher = more anomalous
    # scale so a typical transaction ~0 and the most extreme training point = 100
    scaling = {"median": float(np.median(raw)), "max": float(raw.max())}
    return model, scaling


def isolation_score(model: IsolationForest, scaling: dict, X: pd.DataFrame) -> np.ndarray:
    raw = -model.score_samples(X)
    return np.clip((raw - scaling["median"]) / (scaling["max"] - scaling["median"]), 0, 1) * 100


def train_xgboost(X_train, y_train, X_valid, y_valid) -> XGBClassifier:
    """Supervised. Class imbalance handled with scale_pos_weight (= legit / fraud ratio)
    instead of SMOTE: oversampling interpolates between fraud rows, which invents
    velocity/geo combinations that cannot exist, while weighting keeps the data real."""
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=5, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=pos_weight, eval_metric="aucpr", early_stopping_rounds=50,
        tree_method="hist", random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


# --------------------------------------------------------------- 5. score
def risk_tier(score: pd.Series) -> pd.Series:
    return pd.Series(np.select([score >= t for t, _ in TIERS], [name for _, name in TIERS], "Low"),
                     index=score.index)


# ------------------------------------------------------------ 6. evaluate
def evaluate(name: str, y: pd.Series, amount: pd.Series, score: np.ndarray, flagged: np.ndarray,
             scenario: pd.Series) -> dict:
    y, flagged = y.to_numpy().astype(bool), np.asarray(flagged, dtype=bool)
    amount = amount.to_numpy(dtype=float)
    tp, fp = (flagged & y).sum(), (flagged & ~y).sum()
    fraud_dollars = amount[y].sum()
    caught_dollars = amount[flagged & y].sum()
    by_scenario = (pd.DataFrame({"scenario": scenario.to_numpy(), "caught": flagged})[y]
                   .groupby("scenario")["caught"].mean().round(3).to_dict())
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, score), 4),
        "pr_auc": round(average_precision_score(y, score), 4),
        "flagged_txns": int(flagged.sum()),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(y.sum(), 1), 3),
        "false_positive_rate": round(fp / max((~y).sum(), 1), 4),
        "fraud_dollars": round(fraud_dollars, 2),
        "fraud_dollars_caught": round(caught_dollars, 2),
        "dollar_recall": round(caught_dollars / max(fraud_dollars, 1e-9), 3),
        "recall_by_scenario": by_scenario,
    }


# ------------------------------------------------------------- 7. publish
def publish_scores(df: pd.DataFrame) -> None:
    ddl = """
        DROP TABLE IF EXISTS insureguard.txn_risk_scores CASCADE;  -- bi views are rebuilt by build_bi.py
        CREATE TABLE insureguard.txn_risk_scores (
            transaction_id           VARCHAR(12)   NOT NULL PRIMARY KEY
                                     REFERENCES insureguard.fact_transaction (transaction_id),
            model_version            VARCHAR(10)   NOT NULL,
            scored_at                TIMESTAMP     NOT NULL,
            dataset_split            VARCHAR(5)    NOT NULL,   -- train | valid | test
            xgb_fraud_probability    NUMERIC(6, 5) NOT NULL,
            isolation_anomaly_score  NUMERIC(5, 2) NOT NULL,   -- 0-100
            anomaly_risk_score       SMALLINT      NOT NULL,   -- 0-100, blended
            risk_tier                VARCHAR(8)    NOT NULL,   -- Low | Medium | High | Critical
            is_flagged               BOOLEAN       NOT NULL
        );
    """
    cols = ["transaction_id", "model_version", "scored_at", "dataset_split", "xgb_fraud_probability",
            "isolation_anomaly_score", "anomaly_risk_score", "risk_tier", "is_flagged"]
    buffer = io.StringIO()
    df[cols].to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    with connect() as conn:
        conn.execute(ddl)
        with conn.cursor().copy(f"COPY insureguard.txn_risk_scores ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buffer.read())
        conn.execute("CREATE INDEX ix_scores_tier ON insureguard.txn_risk_scores (risk_tier)")
        conn.execute("ANALYZE insureguard.txn_risk_scores")


def print_report(results: list[dict]) -> None:
    table = pd.DataFrame(results).drop(columns="recall_by_scenario").set_index("model")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(table.to_string())
    print("\nrecall by fraud scenario (test period):")
    print(pd.DataFrame({r["model"]: r["recall_by_scenario"] for r in results}).to_string())


def main() -> None:
    df = load_features()
    X, y = prepare(df), df["is_fraud"].astype(int)
    masks = split_masks(df)
    for name, m in masks.items():
        print(f"{name:<5} {int(m.sum()):>7,} txns  {y[m].mean():.2%} fraud  "
              f"{df.loc[m, 'txn_timestamp'].min():%b %d} to {df.loc[m, 'txn_timestamp'].max():%b %d}")

    iso, iso_scaling = train_isolation_forest(X[masks["train"]])
    xgb = train_xgboost(X[masks["train"]], y[masks["train"]], X[masks["valid"]], y[masks["valid"]])
    print(f"\nxgboost early-stopped at {xgb.best_iteration + 1} trees\n")

    df["dataset_split"] = np.select([masks["train"], masks["valid"]], ["train", "valid"], "test")
    df["xgb_fraud_probability"] = xgb.predict_proba(X)[:, 1].round(5)
    df["isolation_anomaly_score"] = isolation_score(iso, iso_scaling, X).round(2)
    df["anomaly_risk_score"] = np.rint(
        100 * BLEND_XGB_WEIGHT * df["xgb_fraud_probability"]
        + (1 - BLEND_XGB_WEIGHT) * df["isolation_anomaly_score"]).astype(int)
    df["risk_tier"] = risk_tier(df["anomaly_risk_score"])
    df["is_flagged"] = df["anomaly_risk_score"] >= FLAG_THRESHOLD
    df["model_version"] = MODEL_VERSION
    df["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

    # ---- evaluate on the unseen test months only
    t = df[masks["test"]]
    args = (t["is_fraud"], t["amount"])
    iso_cut = np.quantile(t["isolation_anomaly_score"], 1 - t["is_flagged"].mean())  # same alert volume
    results = [
        evaluate("rule: amount >= $500", *args, t["amount"].astype(float), t["amount"] >= 500, t["fraud_scenario"]),
        evaluate("isolation_forest", *args, t["isolation_anomaly_score"], t["isolation_anomaly_score"] >= iso_cut,
                 t["fraud_scenario"]),
        evaluate("xgboost", *args, t["xgb_fraud_probability"], t["xgb_fraud_probability"] >= 0.5, t["fraud_scenario"]),
        evaluate("insureguard_blend", *args, t["anomaly_risk_score"], t["is_flagged"], t["fraud_scenario"]),
    ]
    print_report(results)

    importance = (pd.Series(xgb.feature_importances_, index=X.columns)
                  .sort_values(ascending=False).round(4))
    print("\ntop xgboost features (gain share):")
    print(importance.head(10).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    xgb.save_model(MODELS_DIR / "xgboost_fraud.json")
    joblib.dump({"model": iso, "scaling": iso_scaling, "columns": list(X.columns)}, MODELS_DIR / "isolation_forest.joblib")
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps({
        "model_version": MODEL_VERSION,
        "split": {"train": "2026-01-01 to 2026-03-31", "valid": "2026-04", "test": "2026-05-01 to 2026-06-30"},
        "flag_threshold": FLAG_THRESHOLD,
        "blend": {"xgboost": BLEND_XGB_WEIGHT, "isolation_forest": round(1 - BLEND_XGB_WEIGHT, 2)},
        "xgboost_trees": int(xgb.best_iteration + 1),
        "test_results": results,
        "feature_importance": importance.to_dict(),
    }, indent=2))

    publish_scores(df)
    print(f"\nwrote insureguard.txn_risk_scores ({len(df):,} rows), models/, reports/model_metrics.json")
    print(df.loc[masks["test"], "risk_tier"].value_counts().reindex([n for _, n in TIERS][::-1]).to_string())


if __name__ == "__main__":
    main()
