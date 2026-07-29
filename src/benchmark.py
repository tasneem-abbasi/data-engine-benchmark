"""Time and memory-profile each engine/operation pair."""

import os
import threading
import time

import pandas as pd
import psutil

import operations as ops


def measure(fn, *args, repeats=3):
    """Run fn several times; return median wall time and peak process memory."""
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


ENGINES = {
    "pandas": {
        "read": ops.pandas_read,
        "ops": {
            "filter": ops.pandas_filter,
            "groupby": ops.pandas_groupby,
            "join": ops.pandas_join,
            "rolling": ops.pandas_rolling,
            "sort": ops.pandas_sort,
        },
    },
    "polars": {
        "read": ops.polars_read,
        "ops": {
            "filter": ops.polars_filter,
            "groupby": ops.polars_groupby,
            "join": ops.polars_join,
            "rolling": ops.polars_rolling,
            "sort": ops.polars_sort,
        },
    },
}


def main():
    rows = []

    for engine, cfg in ENGINES.items():
        print(f"running {engine} read")
        t, m = measure(cfg["read"])
        rows.append({"engine": engine, "operation": "read",
                     "median_time_s": t, "peak_mem_mb": m})

        df = cfg["read"]()
        for name, fn in cfg["ops"].items():
            print(f"running {engine} {name}")
            t, m = measure(fn, df)
            rows.append({"engine": engine, "operation": name,
                         "median_time_s": t, "peak_mem_mb": m})

    results = pd.DataFrame(rows)
    results.to_csv("results/results.csv", index=False)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
