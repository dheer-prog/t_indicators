from pathlib import Path
import time

import pandas as pd
import pandas_ta as ta
import t_indicators as t


DATA_PATH = Path("data.csv")
WINDOW = 14
MACD_SLOW = 26
MACD_FAST = 12
MACD_SIGNAL = 9
RUNS = 10
WARMUPS = 3


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run get_data.py first to download and cache the data."
        )

    data = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=["Date"])
    return data.astype("float32")


def my_macd(data: pd.DataFrame) -> pd.DataFrame:
    return t.macd_dataframe(data, MACD_SLOW, MACD_FAST)


def pandas_ta_macd(data: pd.DataFrame) -> pd.DataFrame:
    result = {}
    macd_column = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    for column in data.columns:
        macd_frame = ta.macd(data[column], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd_frame is None or macd_column not in macd_frame:
            raise RuntimeError(f"pandas_ta.macd did not return {macd_column} for column {column}.")
        result[column] = macd_frame[macd_column]
    return pd.DataFrame(result, index=data.index)


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

    print(
        f"MACD benchmark rows={len(data)}, columns={len(data.columns)}, "
        f"slow={MACD_SLOW}, fast={MACD_FAST}, signal={MACD_SIGNAL}"
    )

    benchmark("My Library MACD", lambda: my_macd(data))
    benchmark("Pandas TA MACD", lambda: pandas_ta_macd(data))


if __name__ == "__main__":
    main()
