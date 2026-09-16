"""
InsureGuard | Step 1: synthetic financial transaction generator.

Produces three relational CSVs in data/raw/:
  customers.csv     -> insureguard.dim_customer
  devices.csv       -> insureguard.dim_device
  transactions.csv  -> insureguard.fact_transaction

Legitimate behavior is modeled per customer and deliberately includes the
"innocent anomalies" that make real fraud detection hard:
  * multi-day trips to distant cities (legit geo jumps after a flight)
  * new phones activated mid-period (legit brand-new devices)
  * short bursts of small purchases (coffee, transit, app stores)
  * night-owl customers who routinely shop after midnight

Fraud is injected as incidents with overlapping, imperfect signatures:
  account_takeover   2-6 high-value purchases over a few hours, usually new device, often distant city
  card_testing       3-10 small charges within ~90 min, usually new device
  impossible_travel  purchase far from a home purchase made shortly before
  high_value_spend   4-20x normal spend on the customer's own device, often at night
  friendly_fraud     customer disputes their own ordinary purchases (by design ~undetectable)

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

# legitimate activity by hour of day
HOUR_W = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.3, 0.8, 1.5, 2.2, 2.6, 2.9, 3.4,
                   3.8, 3.4, 3.0, 3.0, 3.3, 3.8, 4.0, 3.6, 3.0, 2.2, 1.4, 0.7])
HOUR_P = HOUR_W / HOUR_W.sum()
# night owls (shift workers, students): activity shifted about 7 hours later
HOUR_P_OWL = np.roll(HOUR_P, 7)


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
        self.extra_devices: list[dict] = []

    # ---------- dimensions ----------
    def build_customers(self) -> pd.DataFrame:
        rng, n = self.rng, self.n_customers
        self.home_idx = rng.choice(len(CITIES), n, p=CITY_P)
        segment = rng.choice(["standard", "premium", "business"], n, p=[0.75, 0.20, 0.05])

        # hidden behavioral traits (not written to the database)
        self.activity = rng.lognormal(0.0, 0.7, n)
        self.spend_mu = rng.normal(3.3, 0.45, n) + np.select(
            [segment == "premium", segment == "business"], [0.4, 0.8], 0.0)
        self.night_owl = rng.random(n) < 0.06

        # one multi-day trip to a distant city for about a third of customers
        self.has_trip = rng.random(n) < 0.35
        self.trip_start = rng.integers(0, DAYS - 8, n)
        self.trip_end = self.trip_start + rng.integers(2, 8, n)
        self.trip_city = np.array([self.far_city(h) for h in self.home_idx])

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
        total = int(self.n_dev.sum())

        # about half of multi-device customers get their newest device (new phone) mid-period
        last_dev = self.dev_offset + self.n_dev - 1
        upgrades = last_dev[(self.n_dev > 1) & (rng.random(self.n_customers) < 0.5)]
        self.dev_active_day = np.zeros(total, dtype=int)
        self.dev_active_day[upgrades] = rng.integers(10, DAYS - 10, len(upgrades))

        signup = pd.to_datetime(np.repeat(pd.to_datetime(customers["signup_date"]).values, self.n_dev))
        first_seen = signup + pd.to_timedelta(rng.integers(0, 86400 * 30, total), unit="s")
        activated = START + pd.to_timedelta(self.dev_active_day, unit="D") + pd.to_timedelta(rng.integers(0, 36000, total), unit="s")
        first_seen = pd.Series(first_seen).where(self.dev_active_day == 0, pd.Series(activated))

        self.devices = pd.DataFrame({
            "device_id": [f"D{i:07d}" for i in range(1, total + 1)],
            "customer_id": np.repeat(customers["customer_id"].values, self.n_dev),
            "device_type": rng.choice(["mobile_ios", "mobile_android", "desktop_web"], total, p=[0.45, 0.35, 0.20]),
            "first_seen_at": first_seen,
            "is_trusted": True,
        })
        return self.devices

    def new_fraud_device(self, customer_id: str, seen_at: pd.Timestamp) -> str:
        """Fraudster's device gets attached to the victim's account at takeover time."""
        self.fraud_device_seq += 1
        device_id = f"X{self.fraud_device_seq:07d}"
        self.extra_devices.append({
            "device_id": device_id, "customer_id": customer_id,
            "device_type": self.rng.choice(["desktop_web", "mobile_android", "mobile_ios"]),
            "first_seen_at": seen_at, "is_trusted": False,
        })
        return device_id

    # ---------- helpers ----------
    def random_time(self, night: bool = False) -> pd.Timestamp:
        rng = self.rng
        hour = rng.integers(0, 5) if night else rng.choice(24, p=HOUR_P)
        return START + pd.Timedelta(days=int(rng.integers(0, DAYS - 1)), hours=int(hour),
                                    seconds=int(rng.integers(0, 3600)))

    def far_city(self, home: int, min_km: float = 800) -> int:
        return int(self.rng.choice(np.flatnonzero(CITY_DIST[home] >= min_km)))

    def primary_device(self, c: int) -> str:
        return self.devices["device_id"].values[self.dev_offset[c]]

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

    # ---------- fraud ----------
    def build_fraud(self) -> list[dict]:
        rng, rows = self.rng, []
        target = int(self.n_rows * self.fraud_rate)
        scenarios = ["account_takeover", "card_testing", "impossible_travel", "high_value_spend", "friendly_fraud"]
        while sum(r["is_fraud"] for r in rows) < target:
            c = int(rng.integers(self.n_customers))
            home, cid = int(self.home_idx[c]), f"C{c + 1:06d}"
            scenario = rng.choice(scenarios, p=[0.26, 0.20, 0.14, 0.15, 0.25])
            t0 = self.random_time(night=(scenario == "high_value_spend" and rng.random() < 0.6))

            if scenario == "account_takeover":
                dev = self.new_fraud_device(cid, t0) if rng.random() < 0.85 else self.primary_device(c)
                city = self.far_city(home) if rng.random() < 0.55 else home  # proxies/local fraud rings
                for m in np.sort(rng.uniform(0, 180, rng.integers(2, 7))):
                    rows.append(self.row(c, dev, t0 + pd.Timedelta(minutes=float(m)), rng.lognormal(5.0, 0.8),
                                         rng.choice(["electronics", "gift_cards", "retail", "travel"]),
                                         "online", city, True, scenario))
            elif scenario == "card_testing":
                dev = self.new_fraud_device(cid, t0) if rng.random() < 0.9 else self.primary_device(c)
                city = home if rng.random() < 0.5 else int(rng.choice(len(CITIES), p=CITY_P))
                span = rng.uniform(10, 90)
                for m in np.sort(rng.uniform(0, span, rng.integers(3, 11))):
                    rows.append(self.row(c, dev, t0 + pd.Timedelta(minutes=float(m)), rng.uniform(0.5, 15),
                                         rng.choice(["entertainment", "retail", "gift_cards", "restaurants"]),
                                         "online", city, True, scenario))
            elif scenario == "impossible_travel":
                rows.append(self.row(c, self.primary_device(c), t0, rng.lognormal(self.spend_mu[c], 0.5),
                                     "grocery", "in_store", home, False, None))
                dev = self.new_fraud_device(cid, t0) if rng.random() < 0.7 else self.primary_device(c)
                rows.append(self.row(c, dev, t0 + pd.Timedelta(minutes=float(rng.uniform(30, 240))),
                                     rng.lognormal(4.7, 0.7), rng.choice(["electronics", "retail", "restaurants"]),
                                     "in_store", self.far_city(home), True, scenario))
            elif scenario == "high_value_spend":  # compromised session on the customer's own device
                for m in np.sort(rng.uniform(0, 60, rng.integers(1, 3))):
                    rows.append(self.row(c, self.primary_device(c), t0 + pd.Timedelta(minutes=float(m)),
                                         np.exp(self.spend_mu[c]) * rng.uniform(4, 20),
                                         rng.choice(["electronics", "travel", "retail"]), "online", home, True, scenario))
            else:  # friendly_fraud: customer later disputes an ordinary purchase
                for _ in range(rng.integers(1, 3)):
                    rows.append(self.row(c, self.primary_device(c), self.random_time(),
                                         rng.lognormal(self.spend_mu[c] + np.log(2.0), 0.6),
                                         rng.choice(["retail", "electronics", "travel", "entertainment"]),
                                         rng.choice(["online", "in_store"]), home, True, scenario))
        return rows

    # ---------- legitimate ----------
    def build_legit_bursts(self, n_rows: int) -> list[dict]:
        """Innocent velocity: a coffee run, transit taps, several app-store buys in one sitting."""
        rng, rows = self.rng, []
        while len(rows) < n_rows:
            c = int(rng.choice(self.n_customers, p=self.activity / self.activity.sum()))
            t0 = self.random_time()
            channel = rng.choice(["online", "in_store"])
            for m in np.sort(rng.uniform(0, 45, rng.integers(3, 9))):
                rows.append(self.row(c, self.primary_device(c), t0 + pd.Timedelta(minutes=float(m)),
                                     rng.lognormal(2.3, 0.6), rng.choice(["restaurants", "entertainment", "retail"]),
                                     channel, int(self.home_idx[c]), False, None))
        return rows[:n_rows]

    def build_legit(self, n: int) -> pd.DataFrame:
        rng = self.rng
        c = rng.choice(self.n_customers, n, p=self.activity / self.activity.sum())
        cat = rng.choice(len(CAT_NAME), n, p=CAT_P)

        day = rng.integers(0, DAYS, n)
        hour = np.where(self.night_owl[c], rng.choice(24, n, p=HOUR_P_OWL), rng.choice(24, n, p=HOUR_P))
        ts = (START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h")
              + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))

        on_trip = self.has_trip[c] & (day >= self.trip_start[c]) & (day < self.trip_end[c])
        wandering = rng.random(n) < 0.03  # one-off purchases elsewhere (online merchants, day trips)
        city = np.where(on_trip, self.trip_city[c],
                        np.where(wandering, rng.choice(len(CITIES), n, p=CITY_P), self.home_idx[c]))

        n_dev = self.n_dev[c]
        other = 1 + np.floor(rng.random(n) * np.maximum(n_dev - 1, 1)).astype(int)
        dev_pos = self.dev_offset[c] + np.where((n_dev == 1) | (rng.random(n) < 0.6), 0, other)
        # a device can only be used once it has been activated; fall back to the primary device
        dev_pos = np.where(day >= self.dev_active_day[dev_pos], dev_pos, self.dev_offset[c])
        device = self.devices["device_id"].values[dev_pos]

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

        fraud = pd.DataFrame(self.build_fraud())
        bursts = pd.DataFrame(self.build_legit_bursts(int(self.n_rows * 0.03)))
        legit = self.build_legit(self.n_rows - len(fraud) - len(bursts))
        txns = (pd.concat([legit, bursts, fraud], ignore_index=True)
                .sort_values(["txn_timestamp", "customer_id"], kind="stable")
                .reset_index(drop=True))
        txns["txn_timestamp"] = pd.to_datetime(txns["txn_timestamp"]).dt.floor("s")
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
    print(fraud.groupby("fraud_scenario")["amount"].agg(rows="size", avg_amount="mean", total="sum").round(2).to_string())


if __name__ == "__main__":
    main()
