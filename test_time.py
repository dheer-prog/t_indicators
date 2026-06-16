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


def my_sma(data: pd.DataFrame) -> pd.DataFrame:
    return t.rolling_sma_dataframe(data, WINDOW)


def pandas_sma(data: pd.DataFrame) -> pd.DataFrame:
    return data.rolling(WINDOW).mean()


def pandas_ta_sma(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(lambda column: ta.sma(column, length=WINDOW))


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

    benchmark("My Library", my_sma, data)
    benchmark("Pandas", pandas_sma, data)
    benchmark("Pandas TA", pandas_ta_sma, data)


if __name__ == "__main__":
    main()
