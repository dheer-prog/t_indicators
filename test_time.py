from pathlib import Path
import time

import pandas as pd
import pandas_ta as ta
import t_indicators as t


DATA_PATH = Path("data.csv")
WINDOW = 20
RUNS = 10
WARMUPS = 3


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run get_data.py first to download and cache the data."
        )

    data = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=["Date"])
    return data.astype("float32")


def my_ema(data: pd.DataFrame) -> pd.DataFrame:
    return t.ema(data, WINDOW)


def pandas_ema(data: pd.DataFrame) -> pd.DataFrame:
    return data.ewm(span=WINDOW, adjust=False, min_periods=WINDOW).mean()


def pandas_ta_ema(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(lambda column: ta.ema(column, length=WINDOW))


def benchmark(name: str, fn, data: pd.DataFrame) -> None:
    for _ in range(WARMUPS):
        result = fn(data)
        _ = result.shape

    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        result = fn(data)
        _ = result.shape
        timings.append(time.perf_counter() - start)

    timings.sort()
    print(
        f"{name}: min={timings[0]:.6f}s "
        f"avg={sum(timings) / len(timings):.6f}s "
        f"med={timings[len(timings) // 2]:.6f}s"
    )


def main() -> None:
    data = load_data()
    print(f"Rows: {len(data)}, Columns: {len(data.columns)}, Window: {WINDOW}")
    print("Note: pandas ewm and pandas_ta/t_indicators use different EMA seeding rules.")

    benchmark("My Library EMA", my_ema, data)
    benchmark("Pandas EMA", pandas_ema, data)
    benchmark("Pandas TA EMA", pandas_ta_ema, data)


if __name__ == "__main__":
    main()
