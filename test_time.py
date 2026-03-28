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
start=datetime.datetime(2010,1,1)
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
rs_calculated=t.RSI_DataFrame(data,5)
end_time=time.time()
print(f'My Method: {end_time-start_time}')
 
 
 
def rolling_rsi(data, window):
    
    # Handle Series input
    if isinstance(data, pd.Series):
        return pd.Series(rolling_rsi_single(data.values, window), index=data.index, name=data.name)
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        result = pd.DataFrame(index=data.index, columns=data.columns)
        for col in data.columns:
            result[col] = rolling_rsi_single(data[col].values, window)
        return result
    
    # Handle array/list input (original behavior)
    return rolling_rsi_single(data, window)


def rolling_rsi_single(data, window):
  
    rsi = [math.nan] * len(data)
    
    if window >= len(data):
        return rsi
    
    average_gain = 0.0
    average_loss = 0.0
    
    # Initial average gain/loss
    for i in range(1, int(window) + 1):
        change = data[i] - data[i - 1]
        if change > 0:
            average_gain += change
        else:
            average_loss -= change
    
    average_gain /= window
    average_loss /= window
    
    if average_loss == 0.0:
        rsi[int(window)] = 100.0
    else:
        rs = average_gain / average_loss
        rsi[int(window)] = 100.0 - (100.0 / (1.0 + rs))
    
    # Rolling calculation
    for i in range(int(window) + 1, len(data)):
        change = data[i] - data[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        
        # Wilder's smoothing
        average_gain = (average_gain * (window - 1) + gain) / window
        average_loss = (average_loss * (window - 1) + loss) / window
        
        if average_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = average_gain / average_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi
start_time=time.time()
rsi=rolling_rsi(data,5)
end_time=time.time()
print(f'Vanilla Python: {end_time-start_time}')
start_time=time.time()
rsi_ta=ta.rsi(data, length=5)
end_time=time.time()
print(f'Pandas TA: {end_time-start_time}')
    
