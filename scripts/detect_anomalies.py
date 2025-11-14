import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from industrial_forecasting.utils.config import load_config
from industrial_forecasting.data import load_series
from industrial_forecasting.anomaly import detect_anomalies_series

def main(cfg_path: str, override_method: str | None = None):
    cfg = load_config(cfg_path)
    s = load_series(
        cfg["data"]["raw_path"],
        cfg["data"]["datetime_col"],
        cfg["data"]["value_col"],
        cfg["data"].get("freq"),
    )
    method = override_method or cfg.get("anomaly", {}).get("method", "zscore")
    out = detect_anomalies_series(
        s,
        method=method,
        contamination=cfg.get("anomaly", {}).get("contamination", 0.03),
        zscore_threshold=cfg.get("anomaly", {}).get("zscore_threshold", 3.0),
    )
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/anomalies.csv", index=True)
    print("Anomalies sauvegardées dans data/processed/anomalies.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détection d'anomalies")
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["zscore", "isolation_forest"])
    args = ap.parse_args()
    main(args.config, override_method=args.method)
