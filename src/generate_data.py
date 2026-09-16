"""
InsureGuard | Step 1: synthetic financial transaction generator.

Produces three relational CSVs in data/raw/:
  customers.csv     -> insureguard.dim_customer
  devices.csv       -> insureguard.dim_device
  transactions.csv  -> insureguard.fact_transaction

Legitimate behavior is modeled per customer (home city, spend level, diurnal
pattern, primary device). Fraud is injected as *incidents* with realistic
signatures so the Step 2 features (rolling spend, 1h/24h velocity, geo-jump
speed) actually have signal to find:

  account_takeover   burst of 3-6 high-value online txns, new device, distant city
  card_testing       burst of 5-12 micro-charges (<$5) within ~25 min, new device
  impossible_travel  in-store txn far from home minutes after a legit home txn
  high_value_night   1-2 txns at 10-25x normal spend, 1-4am, customer's own device

Usage:
  python src/generate_data.py --rows 100000 --fraud-rate 0.012 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-01-01")
DAYS = 180

# city, state, lat, lon, relative population weight
CITIES = [
    ("New York", "NY", 40.7128, -74.0060, 8.3),
    ("Los Angeles", "CA", 34.0522, -118.2437, 3.9),
    ("Chicago", "IL", 41.8781, -87.6298, 2.7),
    ("Houston", "TX", 29.7604, -95.3698, 2.3),
    ("Phoenix", "AZ", 33.4484, -112.0740, 1.6),
    ("Philadelphia", "PA", 39.9526, -75.1652, 1.6),
    ("San Antonio", "TX", 29.4241, -98.4936, 1.5),
    ("San Diego", "CA", 32.7157, -117.1611, 1.4),
    ("Dallas", "TX", 32.7767, -96.7970, 1.3),
    ("Austin", "TX", 30.2672, -97.7431, 1.0),
    ("Jacksonville", "FL", 30.3322, -81.6557, 0.95),
    ("Columbus", "OH", 39.9612, -82.9988, 0.9),
    ("Charlotte", "NC", 35.2271, -80.8431, 0.9),
    ("San Francisco", "CA", 37.7749, -122.4194, 0.8),
    ("Seattle", "WA", 47.6062, -122.3321, 0.75),
    ("Denver", "CO", 39.7392, -104.9903, 0.72),
    ("Nashville", "TN", 36.1627, -86.7816, 0.69),
    ("Boston", "MA", 42.3601, -71.0589, 0.65),
    ("Portland", "OR", 45.5152, -122.6784, 0.65),
    ("Las Vegas", "NV", 36.1699, -115.1398, 0.65),
    ("Detroit", "MI", 42.3314, -83.0458, 0.63),
    ("Atlanta", "GA", 33.7490, -84.3880, 0.5),
    ("Miami", "FL", 25.7617, -80.1918, 0.45),
    ("Minneapolis", "MN", 44.9778, -93.2650, 0.43),
    ("Salt Lake City", "UT", 40.7608, -111.8910, 0.2),
]
CITY_NAME = np.array([c[0] for c in CITIES])
CITY_STATE = np.array([c[1] for c in CITIES])
CITY_LAT = np.array([c[2] for c in CITIES])
CITY_LON = np.array([c[3] for c in CITIES])
CITY_P = np.array([c[4] for c in CITIES]) / sum(c[4] for c in CITIES)

# category: (frequency weight, amount multiplier, probability online)
CATEGORIES = {
    "grocery": (0.22, 1.0, 0.10),
    "restaurants": (0.18, 0.8, 0.15),
    "gas": (0.12, 1.1, 0.00),
    "retail": (0.14, 1.5, 0.45),
    "entertainment": (0.08, 1.0, 0.60),
    "electronics": (0.06, 4.0, 0.60),
    "utilities": (0.05, 2.5, 0.90),
    "health": (0.05, 1.8, 0.20),
    "travel": (0.04, 6.0, 0.80),
    "gift_cards": (0.02, 2.0, 0.70),
}
CAT_NAME = np.array(list(CATEGORIES))
CAT_P = np.array([v[0] for v in CATEGORIES.values()])
CAT_P = CAT_P / CAT_P.sum()
CAT_MULT = np.array([v[1] for v in CATEGORIES.values()])
CAT_ONLINE = np.array([v[2] for v in CATEGORIES.values()])

# legitimate activity by hour of day (quiet overnight, peaks at lunch and evening)
HOUR_W = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.3, 0.8, 1.5, 2.2, 2.6, 2.9, 3.4,
                   3.8, 3.4, 3.0, 3.0, 3.3, 3.8, 4.0, 3.6, 3.0, 2.2, 1.4, 0.7])
HOUR_P = HOUR_W / HOUR_W.sum()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


CITY_DIST = haversine_km(CITY_LAT[:, None], CITY_LON[:, None], CITY_LAT[None, :], CITY_LON[None, :])


class Generator:
    def __init__(self, n_rows: int, n_customers: int, fraud_rate: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n_rows = n_rows
        self.n_customers = n_customers
        self.fraud_rate = fraud_rate
        self.fraud_device_seq = 0

    # ---------- dimensions ----------
    def build_customers(self) -> pd.DataFrame:
        rng, n = self.rng, self.n_customers
        self.home_idx = rng.choice(len(CITIES), n, p=CITY_P)
        segment = rng.choice(["standard", "premium", "business"], n, p=[0.75, 0.20, 0.05])
        # hidden behavioral traits (not written to the database)
        self.activity = rng.lognormal(0.0, 0.7, n)
        self.spend_mu = rng.normal(3.3, 0.45, n) + np.select(
            [segment == "premium", segment == "business"], [0.4, 0.8], 0.0)

        signup = START - pd.to_timedelta(rng.integers(30, 365 * 8, n), unit="D")
        return pd.DataFrame({
            "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
            "home_city": CITY_NAME[self.home_idx],
            "home_state": CITY_STATE[self.home_idx],
            "home_latitude": CITY_LAT[self.home_idx],
            "home_longitude": CITY_LON[self.home_idx],
            "age": np.clip(rng.normal(42, 14, n), 18, 90).astype(int),
            "customer_segment": segment,
            "signup_date": signup.date,
        })

    def build_devices(self, customers: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        self.n_dev = rng.choice([1, 2, 3], self.n_customers, p=[0.5, 0.35, 0.15])
        self.dev_offset = np.concatenate([[0], np.cumsum(self.n_dev)[:-1]])
        cust = np.repeat(customers["customer_id"].values, self.n_dev)
        signup = np.repeat(pd.to_datetime(customers["signup_date"]).values, self.n_dev)
        total = int(self.n_dev.sum())
        self.devices = pd.DataFrame({
            "device_id": [f"D{i:07d}" for i in range(1, total + 1)],
            "customer_id": cust,
            "device_type": rng.choice(["mobile_ios", "mobile_android", "desktop_web"], total, p=[0.45, 0.35, 0.20]),
            "first_seen_at": pd.to_datetime(signup) + pd.to_timedelta(rng.integers(0, 86400 * 30, total), unit="s"),
            "is_trusted": True,
        })
        return self.devices

    def new_fraud_device(self, customer_id: str, seen_at: pd.Timestamp) -> str:
        """Fraudster's device gets attached to the victim's account at takeover time."""
        self.fraud_device_seq += 1
        device_id = f"X{self.fraud_device_seq:07d}"
        self.extra_devices.append({
            "device_id": device_id, "customer_id": customer_id,
            "device_type": self.rng.choice(["desktop_web", "mobile_android"]),
            "first_seen_at": seen_at, "is_trusted": False,
        })
        return device_id

    # ---------- transactions ----------
    def random_time(self, night: bool = False) -> pd.Timestamp:
        rng = self.rng
        hour = rng.integers(1, 5) if night else rng.choice(24, p=HOUR_P)
        return START + pd.Timedelta(days=int(rng.integers(0, DAYS - 1)), hours=int(hour),
                                    seconds=int(rng.integers(0, 3600)))

    def far_city(self, home: int, min_km: float = 800) -> int:
        candidates = np.flatnonzero(CITY_DIST[home] >= min_km)
        return int(self.rng.choice(candidates))

    def row(self, c, device, ts, amount, category, channel, city, fraud, scenario):
        jitter = self.rng.normal(0, 0.04, 2)
        return {
            "customer_id": f"C{c + 1:06d}", "device_id": device, "txn_timestamp": ts,
            "amount": round(max(float(amount), 0.5), 2), "merchant_category": category,
            "channel": channel, "city": CITY_NAME[city], "state": CITY_STATE[city],
            "latitude": round(CITY_LAT[city] + jitter[0], 5),
            "longitude": round(CITY_LON[city] + jitter[1], 5),
            "is_fraud": fraud, "fraud_scenario": scenario,
        }

    def primary_device(self, c: int) -> str:
        return self.devices["device_id"].values[self.dev_offset[c]]

    def build_fraud(self) -> list[dict]:
        rng, rows = self.rng, []
        target = int(self.n_rows * self.fraud_rate)
        scenarios = ["account_takeover", "card_testing", "impossible_travel", "high_value_night"]
        while sum(r["is_fraud"] for r in rows) < target:
            c = int(rng.integers(self.n_customers))
            home = int(self.home_idx[c])
            cid = f"C{c + 1:06d}"
            scenario = rng.choice(scenarios, p=[0.35, 0.25, 0.20, 0.20])

            if scenario == "account_takeover":
                t0 = self.random_time()
                dev, city = self.new_fraud_device(cid, t0), self.far_city(home)
                for m in np.sort(rng.uniform(0, 90, rng.integers(3, 7))):
                    cat = rng.choice(["electronics", "gift_cards", "retail"], p=[0.5, 0.3, 0.2])
                    rows.append(self.row(c, dev, t0 + pd.Timedelta(minutes=float(m)),
                                         rng.lognormal(5.4, 0.6), cat, "online", city, True, scenario))
            elif scenario == "card_testing":
                t0 = self.random_time()
                dev, city = self.new_fraud_device(cid, t0), int(rng.choice(len(CITIES)))
                for m in np.sort(rng.uniform(0, 25, rng.integers(5, 13))):
                    rows.append(self.row(c, dev, t0 + pd.Timedelta(minutes=float(m)),
                                         rng.uniform(0.5, 4.99), rng.choice(["entertainment", "retail", "gift_cards"]),
                                         "online", city, True, scenario))
            elif scenario == "impossible_travel":
                t0 = self.random_time()
                rows.append(self.row(c, self.primary_device(c), t0, rng.lognormal(self.spend_mu[c], 0.5),
                                     "grocery", "in_store", home, False, None))
                dev = self.new_fraud_device(cid, t0)
                rows.append(self.row(c, dev, t0 + pd.Timedelta(minutes=float(rng.uniform(20, 120))),
                                     rng.lognormal(4.9, 0.6), rng.choice(["electronics", "retail"]),
                                     "in_store", self.far_city(home, 1500), True, scenario))
            else:  # high_value_night: compromised session on the customer's own device
                t0 = self.random_time(night=True)
                for m in np.sort(rng.uniform(0, 40, rng.integers(1, 3))):
                    rows.append(self.row(c, self.primary_device(c), t0 + pd.Timedelta(minutes=float(m)),
                                         np.exp(self.spend_mu[c]) * rng.uniform(10, 25),
                                         rng.choice(["electronics", "travel"]), "online", home, True, scenario))
        return rows

    def build_legit(self, n: int) -> pd.DataFrame:
        rng = self.rng
        c = rng.choice(self.n_customers, n, p=self.activity / self.activity.sum())
        cat = rng.choice(len(CAT_NAME), n, p=CAT_P)

        ts = (START + pd.to_timedelta(rng.integers(0, DAYS, n), unit="D")
              + pd.to_timedelta(rng.choice(24, n, p=HOUR_P), unit="h")
              + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))

        # 94% of spend happens in the home city, the rest is ordinary travel
        traveling = rng.random(n) < 0.06
        city = np.where(traveling, rng.choice(len(CITIES), n, p=CITY_P), self.home_idx[c])

        n_dev = self.n_dev[c]
        other = 1 + np.floor(rng.random(n) * np.maximum(n_dev - 1, 1)).astype(int)
        dev_idx = np.where((n_dev == 1) | (rng.random(n) < 0.7), 0, other)
        device = self.devices["device_id"].values[self.dev_offset[c] + dev_idx]

        amount = rng.lognormal(self.spend_mu[c] + np.log(CAT_MULT[cat]), 0.7)
        return pd.DataFrame({
            "customer_id": np.char.add("C", np.char.zfill((c + 1).astype(str), 6)),
            "device_id": device,
            "txn_timestamp": ts,
            "amount": np.round(np.maximum(amount, 1.0), 2),
            "merchant_category": CAT_NAME[cat],
            "channel": np.where(rng.random(n) < CAT_ONLINE[cat], "online", "in_store"),
            "city": CITY_NAME[city],
            "state": CITY_STATE[city],
            "latitude": np.round(CITY_LAT[city] + rng.normal(0, 0.04, n), 5),
            "longitude": np.round(CITY_LON[city] + rng.normal(0, 0.04, n), 5),
            "is_fraud": False,
            "fraud_scenario": None,
        })

    def run(self):
        customers = self.build_customers()
        devices = self.build_devices(customers)
        self.extra_devices = []

        fraud = pd.DataFrame(self.build_fraud())
        legit = self.build_legit(self.n_rows - len(fraud))
        txns = (pd.concat([legit, fraud], ignore_index=True)
                .sort_values(["txn_timestamp", "customer_id"], kind="stable")
                .reset_index(drop=True))
        txns["txn_timestamp"] = txns["txn_timestamp"].dt.floor("s")
        txns.insert(0, "transaction_id", [f"TXN{i:08d}" for i in range(1, len(txns) + 1)])

        devices = pd.concat([devices, pd.DataFrame(self.extra_devices)], ignore_index=True)
        devices["first_seen_at"] = pd.to_datetime(devices["first_seen_at"]).dt.floor("s")
        return customers, devices, txns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()

    customers, devices, txns = Generator(args.rows, args.customers, args.fraud_rate, args.seed).run()
    args.out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(args.out / "customers.csv", index=False)
    devices.to_csv(args.out / "devices.csv", index=False)
    txns.to_csv(args.out / "transactions.csv", index=False)

    fraud = txns[txns["is_fraud"]]
    print(f"customers:    {len(customers):>8,}")
    print(f"devices:      {len(devices):>8,}  ({(~devices['is_trusted']).sum():,} fraudster devices)")
    print(f"transactions: {len(txns):>8,}  ({txns['txn_timestamp'].min()} to {txns['txn_timestamp'].max()})")
    print(f"fraud rows:   {len(fraud):>8,}  ({len(fraud) / len(txns):.2%})")
    print(f"total volume: ${txns['amount'].sum():>14,.2f}")
    print(f"fraud volume: ${fraud['amount'].sum():>14,.2f}")
    print("\nfraud rows by scenario:")
    print(fraud["fraud_scenario"].value_counts().to_string())


if __name__ == "__main__":
    main()
