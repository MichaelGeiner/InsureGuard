-- =====================================================================
-- InsureGuard | 01_schema.sql
-- Star-style schema: two dimensions + one transaction fact table.
-- Core DDL runs unchanged on PostgreSQL 14+ and Snowflake.
-- (Snowflake accepts PK/FK declarations but does not enforce them.)
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS insureguard;

-- ---------------------------------------------------------------------
-- dim_customer: one row per account holder
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insureguard.dim_customer (
    customer_id        VARCHAR(10)      NOT NULL PRIMARY KEY,   -- C000001
    home_city          VARCHAR(50)      NOT NULL,
    home_state         CHAR(2)          NOT NULL,
    home_latitude      DOUBLE PRECISION NOT NULL,
    home_longitude     DOUBLE PRECISION NOT NULL,
    age                SMALLINT         NOT NULL,
    customer_segment   VARCHAR(20)      NOT NULL,               -- standard | premium | business
    signup_date        DATE             NOT NULL
);

-- ---------------------------------------------------------------------
-- dim_device: every device fingerprint ever seen on an account.
-- is_trusted = FALSE marks devices first seen during a fraud incident
-- (kept for analysis only, it is label-derived and must NOT be a model feature).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insureguard.dim_device (
    device_id          VARCHAR(10)      NOT NULL PRIMARY KEY,   -- D0000001 / X0000001
    customer_id        VARCHAR(10)      NOT NULL REFERENCES insureguard.dim_customer (customer_id),
    device_type        VARCHAR(20)      NOT NULL,               -- mobile_ios | mobile_android | desktop_web
    first_seen_at      TIMESTAMP        NOT NULL,
    is_trusted         BOOLEAN          NOT NULL
);

-- ---------------------------------------------------------------------
-- fact_transaction: grain = one card/account transaction
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS insureguard.fact_transaction (
    transaction_id     VARCHAR(12)      NOT NULL PRIMARY KEY,   -- TXN00000001
    customer_id        VARCHAR(10)      NOT NULL REFERENCES insureguard.dim_customer (customer_id),
    device_id          VARCHAR(10)      NOT NULL REFERENCES insureguard.dim_device (device_id),
    txn_timestamp      TIMESTAMP        NOT NULL,
    amount             NUMERIC(12, 2)   NOT NULL,
    merchant_category  VARCHAR(30)      NOT NULL,
    channel            VARCHAR(10)      NOT NULL,               -- online | in_store
    city               VARCHAR(50)      NOT NULL,
    state              CHAR(2)          NOT NULL,
    latitude           DOUBLE PRECISION NOT NULL,
    longitude          DOUBLE PRECISION NOT NULL,
    is_fraud           BOOLEAN          NOT NULL,               -- ground-truth label
    fraud_scenario     VARCHAR(30)                              -- label metadata, never a feature
);

-- =====================================================================
-- PostgreSQL only (skip on Snowflake: no secondary indexes / CHECKs;
-- use CLUSTER BY (customer_id, txn_timestamp) there instead).
-- =====================================================================
ALTER TABLE insureguard.fact_transaction
    DROP CONSTRAINT IF EXISTS chk_amount_positive,
    DROP CONSTRAINT IF EXISTS chk_channel;
ALTER TABLE insureguard.fact_transaction
    ADD CONSTRAINT chk_amount_positive CHECK (amount > 0),
    ADD CONSTRAINT chk_channel         CHECK (channel IN ('online', 'in_store'));

-- Drives every per-customer window function in Step 2 (rolling spend, velocity, geo-jump).
CREATE INDEX IF NOT EXISTS ix_txn_customer_time ON insureguard.fact_transaction (customer_id, txn_timestamp);
CREATE INDEX IF NOT EXISTS ix_txn_timestamp     ON insureguard.fact_transaction (txn_timestamp);
CREATE INDEX IF NOT EXISTS ix_device_customer   ON insureguard.dim_device (customer_id);
