import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


START = datetime.datetime(1995, 1, 1)
END = datetime.datetime(2025, 1, 1)
OUTPUT_PATH = Path("data.csv")
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "PEP", "COST",
    "AVGO", "ADBE", "CSCO", "NFLX", "INTC", "QCOM", "TXN", "AMD", "SBUX", "AMAT",
    "INTU", "CHTR", "ADI", "LULU", "BKNG", "MU", "ASML", "ADP", "PYPL", "MDLZ",
    "ORLY", "TMUS", "GILD", "CSX", "REGN", "VRTX", "ROST", "PDD", "MRNA", "KLAC",
    "LRCX", "MRVL", "CPRT", "DXCM", "XEL", "SGEN", "VRSK", "CDNS", "SNPS", "ODFL",
    "DDOG", "CRWD", "TEAM", "WDAY", "SPLK", "OKTA", "ZM", "DOCU", "PTON", "ZS",
    "TTD", "PAYX", "CSGP", "WBD", "ABNB", "ARM", "GEHC", "ON", "BKR", "FANG",
    "DASH", "GFS", "HON", "IDXX", "LIN", "MCHP", "MDB", "MNST", "NXPI", "PLTR",
    "PANW", "ROP", "TTWO", "TXN", "VRSK", "WBD", "WDAY", "XEL", "ZM", "ZS",
]


def download_close_data() -> pd.DataFrame:
    data = yf.download(TICKERS, START, END)["Close"]
    data = data.dropna(axis=1, how="any")
    return data.astype("float32")


def main() -> None:
    data = download_close_data()
    data.to_csv(OUTPUT_PATH, index_label="Date")
    print(f"Saved {len(data)} rows x {len(data.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
