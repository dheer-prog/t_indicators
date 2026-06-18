# 📈 Technical Indicators

A Python library for calculating technical indicators on time series data — optimized for stock market data sourced from [yfinance](https://github.com/ranaroussi/yfinance).

---

## Features

- Calculate a wide range of technical indicators (moving averages, oscillators, momentum, volatility, and more)
- Seamlessly integrates with `yfinance` OHLCV data
- Native C++ indicator kernels with Python bindings
- Chunk-aware CSV loading so large datasets do not need to be read into memory all at once
- Clean, intuitive API

---

## Requirements

- Python 3.8+
- `numpy`
- `yfinance`

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/technical-indicators.git
cd technical-indicators
```

### 2. (Recommended) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install via `setup.py`
```bash
python setup.py install
```

Or, for an editable/development install:
```bash
pip install -e .
```

### 4. Install Dependencies

If dependencies aren't pulled in automatically:
```bash
pip install -r requirements.txt
```

---

## Quick Start
```python
import yfinance as yf
import pandas as pd
import t_indicators as ti

# Download stock data
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Calculate indicators
df["EMA_20"] = ti.ema_series(df["Close"], 20)
df["RSI_14"] = ti.RSI(df["Close"], 14)

print(df[["Close", "EMA_20", "RSI_14"]].tail())
```

## Loading Large CSVs Safely
```python
import t_indicators as ti

meta = ti.inspect_csv("data.csv", has_header=True)
print(meta["file_size_mb"], meta["total_rows"], meta["columns"])

# For small files, this loads everything. For files above `auto_limit_mb`,
# it automatically caps the read to `large_file_rows`.
chunk = ti.load_csv(
    "data.csv",
    columns=["Open", "High", "Low", "Close"],
    index_column="Date",
    auto_limit_mb=128,
    large_file_rows=50_000,
)

print(chunk["values"].shape)
print(chunk["truncated"], chunk["next_row"])

# Load the next window explicitly if needed.
if chunk["truncated"]:
    next_chunk = ti.load_csv(
        "data.csv",
        start_row=chunk["next_row"],
        max_rows=50_000,
        columns=["Open", "High", "Low", "Close"],
        index_column="Date",
    )
```

---

## Project Structure
```
technical-indicators/
├── technical_indicators/
│   ├── __init__.py
│   ├── trend.py          # SMA, EMA, DEMA, TEMA, ...
│   ├── momentum.py       # RSI, MACD, Stochastic, ...
│   ├── volatility.py     # Bollinger Bands, ATR, ...
│   └── volume.py         # OBV, VWAP, ...
├── tests/
│   └── test_indicators.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Running Tests
```bash
python -m pytest tests/
```

---

## 🚀 Upcoming

### PyPI Release — `pip install technical-indicators`

We are working on publishing this library to the [Python Package Index (PyPI)](https://pypi.org), which will allow installation without cloning the repository:
```bash
pip install technical-indicators
```

Once published, you will also be able to install a specific version:
```bash
pip install technical-indicators==1.0.0
```

**Planned release checklist:**
- [ ] Finalize public API and versioning (`v1.0.0`)
- [ ] Package and publish to PyPI via `twine`
- [ ] Add PyPI badge to README
- [ ] Set up automated releases via GitHub Actions

Stay tuned — watch or star the repository to get notified when the PyPI release goes live!

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-indicator`
3. Commit your changes: `git commit -m "Add new indicator"`
4. Push to the branch: `git push origin feature/new-indicator`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
