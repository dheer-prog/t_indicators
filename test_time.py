from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas_ta as ta
import t_indicators as t


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


def pandas_rsi_series(series: pd.Series) -> pd.Series:
    values = series.astype("float64").to_numpy(copy=False)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)

    if WINDOW <= 0 or WINDOW >= values.shape[0]:
        return pd.Series(out, index=series.index, name=series.name)

    deltas = np.diff(values)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    avg_gain = gains[:WINDOW].mean()
    avg_loss = losses[:WINDOW].mean()

    if avg_loss == 0.0:
        out[WINDOW] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[WINDOW] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(WINDOW + 1, values.shape[0]):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = ((avg_gain * (WINDOW - 1)) + gain) / WINDOW
        avg_loss = ((avg_loss * (WINDOW - 1)) + loss) / WINDOW

        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(out, index=series.index, name=series.name)


def my_rsi(data: pd.DataFrame) -> pd.DataFrame:
    return t.RSI_DataFrame(data, WINDOW)


def pandas_rsi(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(pandas_rsi_series)


def pandas_ta_rsi(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(lambda column: ta.rsi(column, length=WINDOW))


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

    benchmark("My Library RSI", my_rsi, data)
    benchmark("Pandas RSI", pandas_rsi, data)
    benchmark("Pandas TA RSI", pandas_ta_rsi, data)


if __name__ == "__main__":
    main()
