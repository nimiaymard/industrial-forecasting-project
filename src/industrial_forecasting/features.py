import numpy as np
import pandas as pd

def rolling_features(s: pd.Series, windows=(3, 6, 12)):
    df = pd.DataFrame({"y": s})
    for w in windows:
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_std_{w}"] = s.rolling(w).std()
    df = df.dropna()
    return df

def create_supervised_from_series(s: pd.Series, window: int = 24):
    X, y = [], []
    values = s.values.astype(float)
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X, y
