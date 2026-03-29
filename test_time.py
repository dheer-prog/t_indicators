import t_indicators as t
import pandas as pd
import numpy as np
import time
import math 
import pandas_ta as ta 
import yfinance as yf
import datetime
import math
npNan=np.nan

# pdr.get_data_yahoo = pdr.get_data_yahoo
start=datetime.datetime(2000,1,1)
end=datetime.datetime(2025,1,1)
tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "PEP", "COST",
    "AVGO", "ADBE", "CSCO", "NFLX", "INTC", "QCOM", "TXN", "AMD", "SBUX", "AMAT",
    "INTU", "CHTR", "ADI", "LULU", "BKNG", "MU", "ASML", "ADP", "PYPL", "MDLZ",
    "ORLY", "TMUS", "GILD", "CSX","REGN", "VRTX", "ROST", "PDD", "MRNA", "KLAC",
    "LRCX", "MRVL", "CPRT", "DXCM", "XEL", "SGEN", "VRSK", "CDNS", "SNPS", "ODFL",
    "DDOG", "CRWD", "TEAM", "WDAY", "SPLK", "OKTA", "ZM", "DOCU", "PTON", "ZS",
    "TTD", "PAYX", "CSGP", "WBD", "ABNB", "ARM", "GEHC", "ON", "BKR", "FANG",
    "DASH", "GFS", "HON", "IDXX", "LIN", "MCHP", "MDB", "MNST", "NXPI", "PLTR",
    "PANW", "ROP", "TTWO", "TXN", "VRSK", "WBD", "WDAY", "XEL", "ZM", "ZS"
]

data=yf.download(tickers,start,end)['Close']
data.dropna(axis=1,how='any',inplace=True)
start_time=time.time()
rs_calculated=t.rolling_sma_dataframe(data,20)
end_time=time.time()
print(f'My Method: {end_time-start_time}')
start_time=time.time()
ema_df= data.ewm(span=20, adjust=False).mean()
end_time=time.time()
print(f'Pandas Method: {end_time-start_time}')
start_time=time.time()
rsi_ta=ta.sma(data, length=20)
end_time=time.time()
print(f'Pandas TA: {end_time-start_time}')
import pandas as pd

def calculate_df_ema_pure(df, window):
    """
    Calculates EMA for all numeric columns in a DataFrame using iterative logic.
    """
    if df.empty:
        return pd.DataFrame()

    # Define smoothing factor
    alpha = 2 / (window + 1)
    
    # Convert the entire DataFrame to a list of lists for row-by-row iteration
    data_values = df.values.tolist()
    column_names = df.columns
    
    # 1. Initialize EMA results with the first row of data
    ema_matrix = [data_values[0]]
    
    # 2. Iterate through each row starting from the second one
    for row_idx in range(1, len(data_values)):
        prev_ema_row = ema_matrix[-1]
        current_data_row = data_values[row_idx]
        
        new_ema_row = []
        # 3. Calculate EMA for each column in the current row
        for col_idx in range(len(current_data_row)):
            ema_val = (current_data_row[col_idx] * alpha) + (prev_ema_row[col_idx] * (1 - alpha))
            new_ema_row.append(ema_val)
        
        ema_matrix.append(new_ema_row)
    
    # 4. Return as a new DataFrame with original headers and index
    return pd.DataFrame(ema_matrix, columns=column_names, index=df.index)

# --- Example Usage ---


# --- Example Usage ---
start_time = time.time()
ema_results = calculate_df_ema_pure(data,20)
end_time = time.time()
print(f"Pure Python EMA Calculation Time: {end_time - start_time:.4f}")
    
