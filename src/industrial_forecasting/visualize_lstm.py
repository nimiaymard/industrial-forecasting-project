import argparse
import pandas as pd
import matplotlib.pyplot as plt
from industrial_forecasting.utils.config import load_config

def plot_lstm_forecast(cfg):
    df = pd.read_csv(cfg.data.forecast_path_lstm, parse_dates=True, index_col=0)

    plt.figure(figsize=(12, 6))
    plt.plot(df["y_true"], label="Valeurs réelles", linewidth=2)
    plt.plot(df["y_pred"], label="Prédictions LSTM", linestyle="--")
    plt.title("Prédictions vs Réalité (LSTM)")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.legend()
    
    out_path = getattr(cfg.data, "image_png_lstm", "forecast_lstm.png")
    plt.savefig(out_path)
    print(f" Graphique sauvegardé → {out_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plot_lstm_forecast(cfg)
