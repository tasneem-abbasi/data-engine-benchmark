"""The six benchmark operations, implemented per engine."""

import duckdb
import pandas as pd
import polars as pl

PARQUET = "data/klines.parquet"


# ---------------- pandas ----------------

def pandas_read():
    return pd.read_parquet(PARQUET)


def pandas_filter(df):
    return df[(df["volume"] > 10) & (df["open_time"] >= "2024-06-01")]


def pandas_groupby(df):
    return (df.set_index("open_time")
              .groupby("symbol")
              .resample("1D")
              .agg(open=("open", "first"), high=("high", "max"),
                   low=("low", "min"), close=("close", "last"),
                   volume=("volume", "sum")))


def pandas_join(df):
    daily = (df.groupby([df["symbol"], df["open_time"].dt.date])["volume"]
               .mean().reset_index(name="avg_daily_volume"))
    daily.columns = ["symbol", "day", "avg_daily_volume"]
    left = df.assign(day=df["open_time"].dt.date)
    return left.merge(daily, on=["symbol", "day"], how="left")


def pandas_rolling(df):
    out = df.sort_values("open_time").copy()
    out["ma20"] = out.groupby("symbol")["close"].transform(
        lambda s: s.rolling(20).mean())
    return out


def pandas_sort(df):
    return df.sort_values(["symbol", "open_time"])
# ---------------- polars ----------------

def polars_read():
    return pl.read_parquet(PARQUET)


def polars_filter(df):
    return df.filter(
        (pl.col("volume") > 10) & (pl.col("open_time") >= pl.datetime(2024, 6, 1))
    )


def polars_groupby(df):
    return (df.sort("open_time")
              .group_by_dynamic("open_time", every="1d", group_by="symbol")
              .agg(
                  pl.col("open").first().alias("open"),
                  pl.col("high").max().alias("high"),
                  pl.col("low").min().alias("low"),
                  pl.col("close").last().alias("close"),
                  pl.col("volume").sum().alias("volume"),
              ))


def polars_join(df):
    daily = (df.with_columns(pl.col("open_time").dt.date().alias("day"))
               .group_by(["symbol", "day"])
               .agg(pl.col("volume").mean().alias("avg_daily_volume")))
    return (df.with_columns(pl.col("open_time").dt.date().alias("day"))
              .join(daily, on=["symbol", "day"], how="left"))


def polars_rolling(df):
    return (df.sort(["symbol", "open_time"])
              .with_columns(
                  pl.col("close").rolling_mean(window_size=20).over("symbol").alias("ma20")
              ))


def polars_sort(df):
    return df.sort(["symbol", "open_time"])


# ---------------- duckdb ----------------

def duckdb_read(con):
    return con.execute("SELECT * FROM read_parquet('data/klines.parquet')").df()


def duckdb_filter(con):
    return con.execute("""
        SELECT * FROM read_parquet('data/klines.parquet')
        WHERE volume > 10 AND open_time >= '2024-06-01'
    """).df()


def duckdb_groupby(con):
    return con.execute("""
        SELECT symbol,
               date_trunc('day', open_time) AS day,
               first(open ORDER BY open_time) AS open,
               max(high) AS high,
               min(low) AS low,
               last(close ORDER BY open_time) AS close,
               sum(volume) AS volume
        FROM read_parquet('data/klines.parquet')
        GROUP BY symbol, day
        ORDER BY symbol, day
    """).df()


def duckdb_join(con):
    return con.execute("""
        WITH daily AS (
            SELECT symbol,
                   date_trunc('day', open_time) AS day,
                   avg(volume) AS avg_daily_volume
            FROM read_parquet('data/klines.parquet')
            GROUP BY symbol, day
        )
        SELECT k.*, d.avg_daily_volume
        FROM read_parquet('data/klines.parquet') k
        LEFT JOIN daily d
          ON k.symbol = d.symbol
         AND date_trunc('day', k.open_time) = d.day
    """).df()


def duckdb_rolling(con):
    return con.execute("""
        SELECT *,
               avg(close) OVER (
                   PARTITION BY symbol ORDER BY open_time
                   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
               ) AS ma20
        FROM read_parquet('data/klines.parquet')
    """).df()


def duckdb_sort(con):
    return con.execute("""
        SELECT * FROM read_parquet('data/klines.parquet')
        ORDER BY symbol, open_time
    """).df()
