import argparse, numpy as np, pandas as pd, os
from sklearn.ensemble import IsolationForest
from industrial_forecasting.utils.config import load_config
from industrial_forecasting.data import load_series

def zscore_anomaly(y, threshold=3.0):
    m = np.median(y)
    mad = np.median(np.abs(y - m)) + 1e-9
    z = 0.6745*(y - m)/mad
    return (np.abs(z) > threshold).astype(int)

def main(cfg_path):
    cfg = load_config(cfg_path)
    s = load_series(cfg['data']['raw_path'], cfg['data']['datetime_col'], cfg['data']['value_col'], cfg['data']['freq'])

    method = cfg['anomaly']['method']
    if method == 'isolation_forest':
        model = IsolationForest(contamination=cfg['anomaly']['contamination'], random_state=42)
        y = s.values.reshape(-1,1)
        labels = (model.fit_predict(y) == -1).astype(int)
    else:
        labels = zscore_anomaly(s.values, threshold=cfg['anomaly']['zscore_threshold'])

    out = s.to_frame(name='value')
    out['anomaly'] = labels
    os.makedirs('data/processed', exist_ok=True)
    out.to_csv('data/processed/anomalies.csv')
    print("Anomalies sauvegardées dans data/processed/anomalies.csv")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
