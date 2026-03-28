import t_indicators as t
import pandas as pd
import numpy as np
import pandas_ta as ta

# Create date index (20 daily dates starting from today)
date_index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=20, freq='D')

# Create random data: 20 rows × 10 columns
data = np.random.randn(20, 10)

# Create column names: A to J
columns = list("ABCDEFGHIJ")

# Create the DataFrame
df = pd.DataFrame(data, index=date_index, columns=columns)
data_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
                  index=date_index)

ans=t.rolling_sma_series(data_series,5)
 
 
print(ans)
print(ta.sma(data_series,length=5))