"""Download Binance 1-minute kline data and combine into a single Parquet file."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
MONTHS = [f"2024-{m:02d}" for m in range(1, 13)]


def fetch_month(symbol: str, month: str) -> pd.DataFrame | None:
    """Download one symbol-month zip and return it as a DataFrame."""
    url = f"{BASE_URL}/{symbol}/1m/{symbol}-1m-{month}.zip"
    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        print(f"  skipped {symbol} {month} (HTTP {response.status_code})")
        return None

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file, header=None, names=COLUMNS)

    # Some newer files ship with a header row; drop it if present.
    if not str(df.iloc[0]["open_time"]).isdigit():
        df = df.iloc[1:]

    df["symbol"] = symbol
    return df


def main() -> None:
    frames = []
    for symbol in SYMBOLS:
        for month in MONTHS:
            print(f"fetching {symbol} {month}")
            df = fetch_month(symbol, month)
            if df is not None:
                frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Timestamps arrive as integers; convert to real datetimes.
    for col in ["open_time", "close_time"]:
        combined[col] = pd.to_numeric(combined[col])
        unit = "us" if combined[col].iloc[0] > 1e14 else "ms"
        combined[col] = pd.to_datetime(combined[col], unit=unit)

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote",
    ]
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric)
    combined = combined.drop(columns=["ignore"])

    out_path = Path("data/klines.parquet")
    out_path.parent.mkdir(exist_ok=True)
    combined.to_parquet(out_path, index=False)

    print(f"\nrows: {len(combined):,}")
    print(f"symbols: {combined['symbol'].nunique()}")
    print(f"date range: {combined['open_time'].min()} to {combined['open_time'].max()}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()

