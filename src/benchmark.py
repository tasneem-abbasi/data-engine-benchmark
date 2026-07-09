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


def main():
    rows = []

    # 'read' takes no dataframe; the rest operate on one already in memory.
    t, m = measure(ops.pandas_read)
    rows.append({"engine": "pandas", "operation": "read",
                 "median_time_s": t, "peak_mem_mb": m})

    df = ops.pandas_read()

    for name, fn in [
        ("filter", ops.pandas_filter),
        ("groupby", ops.pandas_groupby),
        ("join", ops.pandas_join),
        ("rolling", ops.pandas_rolling),
        ("sort", ops.pandas_sort),
    ]:
        print(f"running pandas {name}")
        t, m = measure(fn, df)
        rows.append({"engine": "pandas", "operation": name,
                     "median_time_s": t, "peak_mem_mb": m})

    results = pd.DataFrame(rows)
    results.to_csv("results/results.csv", index=False)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
