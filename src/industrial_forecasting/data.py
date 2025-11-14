import pandas as pd

def load_series(path: str, ts_col: str, val_col: str, freq: str = None) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[ts_col])
    df = df.sort_values(ts_col)
    if freq:
        df = df.set_index(ts_col).asfreq(freq)
    else:
        df = df.set_index(ts_col)
    s = df[val_col].astype(float)
    return s

def train_test_split_series(s: pd.Series, train_ratio: float = 0.8):
    n = len(s)
    split = int(n * train_ratio)
    return s.iloc[:split], s.iloc[split:]
