"""Time and memory-profile each engine/operation pair, across data sizes."""

import os
import threading
import time

import duckdb
import pandas as pd
import polars as pl
import psutil

import operations as ops

PARQUET = "data/klines.parquet"
SIZES = [100_000, 500_000, 1_000_000, None]  # None = full dataset


def measure(fn, *args, repeats=3):
    proc = psutil.Process(os.getpid())
    times, peak_mem = [], 0
    stop = threading.Event()

    def sample():
        nonlocal peak_mem
        while not stop.is_set():
            peak_mem = max(peak_mem, proc.memory_info().rss)
            time.sleep(0.01)

    for _ in range(repeats):
        stop.clear()
        watcher = threading.Thread(target=sample)
        watcher.start()
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
        stop.set()
        watcher.join()

    times.sort()
    return times[len(times) // 2], peak_mem / 1e6


def make_slice(n):
    full = pd.read_parquet(PARQUET)
    if n is None:
        return PARQUET, len(full)
    sliced = full.head(n)
    path = f"data/_slice_{n}.parquet"
    sliced.to_parquet(path, index=False)
    return path, len(sliced)


PANDAS_OPS = {
    "filter": ops.pandas_filter, "groupby": ops.pandas_groupby,
    "join": ops.pandas_join, "rolling": ops.pandas_rolling, "sort": ops.pandas_sort,
}
POLARS_OPS = {
    "filter": ops.polars_filter, "groupby": ops.polars_groupby,
    "join": ops.polars_join, "rolling": ops.polars_rolling, "sort": ops.polars_sort,
}
DUCKDB_OPS = {
    "filter": ops.duckdb_filter, "groupby": ops.duckdb_groupby,
    "join": ops.duckdb_join, "rolling": ops.duckdb_rolling, "sort": ops.duckdb_sort,
}


def main():
    rows = []

    for size in SIZES:
        path, n = make_slice(size)
        print(f"\n=== size: {n:,} rows ===")

        # --- read is now TIMED, once per engine ---
        print("running pandas read")
        t, m = measure(ops.pandas_read, path)
        rows.append({"engine": "pandas", "operation": "read",
                     "rows": n, "median_time_s": t, "peak_mem_mb": m})

        print("running polars read")
        t, m = measure(ops.polars_read, path)
        rows.append({"engine": "polars", "operation": "read",
                     "rows": n, "median_time_s": t, "peak_mem_mb": m})

        con = duckdb.connect()
        print("running duckdb read")
        t, m = measure(ops.duckdb_read, con, path)
        rows.append({"engine": "duckdb", "operation": "read",
                     "rows": n, "median_time_s": t, "peak_mem_mb": m})

        # --- then the five operations, on already-loaded data ---
        df_pd = pd.read_parquet(path)
        for name, fn in PANDAS_OPS.items():
            print(f"running pandas {name}")
            t, m = measure(fn, df_pd)
            rows.append({"engine": "pandas", "operation": name,
                         "rows": n, "median_time_s": t, "peak_mem_mb": m})

        df_pl = pl.read_parquet(path)
        for name, fn in POLARS_OPS.items():
            print(f"running polars {name}")
            t, m = measure(fn, df_pl)
            rows.append({"engine": "polars", "operation": name,
                         "rows": n, "median_time_s": t, "peak_mem_mb": m})

        for name, fn in DUCKDB_OPS.items():
            print(f"running duckdb {name}")
            t, m = measure(fn, con, path)
            rows.append({"engine": "duckdb", "operation": name,
                         "rows": n, "median_time_s": t, "peak_mem_mb": m})

    results = pd.DataFrame(rows)
    results.to_csv("results/results.csv", index=False)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
