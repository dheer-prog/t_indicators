from pathlib import Path
from functools import lru_cache
import time

import pandas as pd
import pandas_ta as ta
import t_indicators as t
import yfinance as yf


DATA_PATH = Path("data.csv")
WINDOW = 14
RUNS = 10
WARMUPS = 3


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run get_data.py first to download and cache the data."
        )

    data = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=["Date"])
    return data.astype("float32")


@lru_cache(maxsize=1)
def load_williams_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_only = load_data()
    tickers = close_only.columns.tolist()
    start = close_only.index.min()
    end = close_only.index.max() + pd.Timedelta(days=1)

    ohlc = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if ohlc.empty:
        raise RuntimeError("yfinance returned no data for Williams %R benchmark.")

    open_data = ohlc["Open"].astype("float32")
    high_data = ohlc["High"].astype("float32")
    low_data = ohlc["Low"].astype("float32")
    close_data = ohlc["Close"].astype("float32")

    common_columns = sorted(
        set(open_data.columns) & set(high_data.columns) & set(low_data.columns) & set(close_data.columns)
    )
    open_data = open_data[common_columns]
    high_data = high_data[common_columns]
    low_data = low_data[common_columns]
    close_data = close_data[common_columns]

    valid_rows = ~(
        open_data.isna().any(axis=1)
        | high_data.isna().any(axis=1)
        | low_data.isna().any(axis=1)
        | close_data.isna().any(axis=1)
    )
    open_data = open_data.loc[valid_rows]
    high_data = high_data.loc[valid_rows]
    low_data = low_data.loc[valid_rows]
    close_data = close_data.loc[valid_rows]

    if open_data.empty or not common_columns:
        raise RuntimeError("No complete OHLC rows were available for Williams %R benchmark.")

    return open_data, high_data, low_data, close_data


def prepare_williams_inputs(
    data: pd.DataFrame, window: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if window <= 0:
        raise ValueError("window must be positive")

    return load_williams_data()


def my_williams(high_data: pd.DataFrame, low_data: pd.DataFrame, close_data: pd.DataFrame) -> pd.DataFrame:
    return t.williams_r_DataFrame(high_data, low_data, close_data, WINDOW)


def pandas_ta_williams(high_data: pd.DataFrame, low_data: pd.DataFrame, close_data: pd.DataFrame) -> pd.DataFrame:
    result = {
        column: ta.willr(high_data[column], low_data[column], close_data[column], length=WINDOW)
        for column in close_data.columns
    }
    return pd.DataFrame(result, index=close_data.index)


def benchmark(name: str, fn) -> None:
    for _ in range(WARMUPS):
        result = fn()
        _ = result.shape

    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        result = fn()
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
    _, high_data, low_data, close_data = prepare_williams_inputs(data, WINDOW)

    print(
        f"Williams %R benchmark rows={len(close_data)}, columns={len(close_data.columns)}, "
        f"window={WINDOW}"
    )

    benchmark("My Library Williams %R", lambda: my_williams(high_data, low_data, close_data))
    benchmark("Pandas TA Williams %R", lambda: pandas_ta_williams(high_data, low_data, close_data))


if __name__ == "__main__":
    main()
