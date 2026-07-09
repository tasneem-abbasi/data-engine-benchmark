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
