from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import t_indicators as t


DEFAULT_DATA_PATH = Path("data.csv")
DEFAULT_WINDOW = 14
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-5
INDICATOR_TO_COMPARE = "ema"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run get_data.py first.")

    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return data.astype("float32")


def pandas_sma(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return data.rolling(window=window, min_periods=window).mean()


def pandas_ta_sma(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return data.apply(lambda column: ta.sma(column, length=window))


def my_sma(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return t.rolling_sma_dataframe(data, window)


def pandas_ema(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return data.ewm(span=window, adjust=False, min_periods=window).mean()


def pandas_ta_ema(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return data.apply(lambda column: ta.ema(column, length=window))


def my_ema(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return t.ema(data, window)


def pandas_rsi_series(series: pd.Series, window: int) -> pd.Series:
    values = series.astype("float64").to_numpy(copy=False)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)

    if window <= 0 or window >= values.shape[0]:
        return pd.Series(out, index=series.index, name=series.name)

    deltas = np.diff(values)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    if avg_loss == 0.0:
        out[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[window] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(window + 1, values.shape[0]):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = ((avg_gain * (window - 1)) + gain) / window
        avg_loss = ((avg_loss * (window - 1)) + loss) / window

        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(out, index=series.index, name=series.name)


def pandas_rsi(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return data.apply(lambda column: pandas_rsi_series(column, window))


def pandas_ta_rsi(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return data.apply(lambda column: ta.rsi(column, length=window))


def my_rsi(data: pd.DataFrame, window: int) -> pd.DataFrame:
    return t.RSI_DataFrame(data, window)


INDICATORS = {
    "sma": {
        "pandas": pandas_sma,
        "pandas_ta": pandas_ta_sma,
        "framework": my_sma,
    },
    "ema": {
        "pandas": pandas_ema,
        "pandas_ta": pandas_ta_ema,
        "framework": my_ema,
    },
    "rsi": {
        "pandas": pandas_rsi,
        "pandas_ta": pandas_ta_rsi,
        "framework": my_rsi,
    },
}


def selected_indicators() -> list[str]:
    if INDICATOR_TO_COMPARE == "all":
        return list(INDICATORS.keys())

    if INDICATOR_TO_COMPARE not in INDICATORS:
        raise ValueError(
            f"Invalid INDICATOR_TO_COMPARE={INDICATOR_TO_COMPARE!r}. "
            f"Choose from {', '.join(sorted(INDICATORS))}, or 'all'."
        )

    return [INDICATOR_TO_COMPARE]


def compare_frames(
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    atol: float,
    rtol: float,
) -> None:
    left_frame = left.astype("float64")
    right_frame = right.astype("float64").reindex(index=left_frame.index, columns=left_frame.columns)

    left_values = left_frame.to_numpy()
    right_values = right_frame.to_numpy()
    finite_mask = np.isfinite(left_values) & np.isfinite(right_values)

    compared = int(finite_mask.sum())
    skipped = int(left_values.size - compared)
    if compared == 0:
        print(f"  {left_name} vs {right_name}: no overlapping finite values to compare")
        return

    left_comp = left_values[finite_mask]
    right_comp = right_values[finite_mask]
    diffs = np.abs(left_comp - right_comp)
    limits = atol + (rtol * np.abs(right_comp))
    mismatch_mask = diffs > limits
    mismatch_count = int(mismatch_mask.sum())
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())

    if mismatch_count == 0:
        print(
            f"  {left_name} vs {right_name}: PASS "
            f"(compared={compared}, skipped_nan={skipped}, max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g})"
        )
        return

    masked_positions = np.argwhere(finite_mask)
    worst_flat_index = int(np.argmax(diffs))
    row_pos, col_pos = masked_positions[worst_flat_index]
    row_label = left_frame.index[row_pos]
    col_label = left_frame.columns[col_pos]

    print(
        f"  {left_name} vs {right_name}: FAIL "
        f"(compared={compared}, skipped_nan={skipped}, mismatches={mismatch_count}, "
        f"max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}, "
        f"worst_at={row_label}/{col_label})"
    )


def run_indicator(name: str, data: pd.DataFrame, window: int, atol: float, rtol: float) -> None:
    print(f"\n{name.upper()} (window={window})")
    if name == "ema":
        print("  Note: pandas ewm uses a different EMA seed than pandas_ta and the framework.")

    outputs: dict[str, pd.DataFrame] = {}
    for source_name, fn in INDICATORS[name].items():
        try:
            outputs[source_name] = fn(data, window)
        except Exception as exc:
            print(f"  {source_name}: ERROR - {exc}")

    if len(outputs) < 2:
        print("  Not enough successful outputs to compare.")
        return

    pairs = [
        ("pandas", "pandas_ta"),
        ("pandas", "framework"),
        ("pandas_ta", "framework"),
    ]

    for left_name, right_name in pairs:
        if left_name in outputs and right_name in outputs:
            compare_frames(left_name, outputs[left_name], right_name, outputs[right_name], atol, rtol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare indicator values across pandas, pandas_ta, and t_indicators.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file to load. Defaults to data.csv.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Indicator window length.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance for comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance for comparisons.",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=sorted(INDICATORS.keys()),
        default=None,
        help="Indicators to compare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)

    print(
        f"Loaded {len(data)} rows x {len(data.columns)} columns from {args.data} "
        f"(window={args.window}, atol={args.atol}, rtol={args.rtol})"
    )

    indicators = args.indicators if args.indicators is not None else selected_indicators()

    for name in indicators:
        run_indicator(name, data, args.window, args.atol, args.rtol)


if __name__ == "__main__":
    main()
