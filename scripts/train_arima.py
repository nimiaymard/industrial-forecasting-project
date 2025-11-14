
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse
import pandas as pd
from industrial_forecasting.utils.config import load_config
from industrial_forecasting.data import train_test_split_series
from industrial_forecasting.models.arima import ARIMAForecaster
from industrial_forecasting.evaluate import mae, rmse

def main(cfg_path):
    print(" Début exécution MAIN")
    print(f" Chemin config reçu : {cfg_path}")

    # --- Charger la configuration ---
    cfg = load_config(cfg_path)
    print(" Configuration chargée ")

    # --- Charger la série nettoyée ---
    print(f" Chargement série depuis : {cfg.data.processed_path}")
    s = pd.read_csv(
        cfg.data.processed_path,
        parse_dates=[cfg.data.datetime_col],
        index_col=cfg.data.datetime_col
    )[cfg.data.value_col].astype(float).sort_index()
    print(f"Série chargée : {len(s)} lignes")

    # --- Forcer la fréquence (si fournie) ---
    if cfg.data.freq:
        s = s.asfreq(cfg.data.freq.lower())
        print(f"Fréquence forcée à : {cfg.data.freq.lower()}")

    # --- Split train/test ---
    train, test = train_test_split_series(s, cfg.data.train_ratio)
    print(f"Train : {len(train)} obs | Test : {len(test)} obs")
    print("NaN dans train :", train.isna().sum())
    print("NaN dans test :", test.isna().sum())
    train = train.interpolate()
    test = test.interpolate()
    print("NaN dans train :", train.isna().sum())
    print("NaN dans test :", test.isna().sum())
    print(train.describe())


    # --- Entraînement ARIMA ---
    print(f" ARIMA order = {cfg.arima.order} / seasonal = {cfg.arima.seasonal_order}")
    arima = ARIMAForecaster(
        order=cfg.arima.order,
        seasonal_order=cfg.arima.seasonal_order
    ).fit(train.values)
    print(" Entraînement ARIMA terminé ")

    # --- Prévisions ---
    yhat = arima.forecast(steps=len(test))
    yhat = pd.Series(yhat, index=test.index)
    print("Nombre de NaN dans les prévisions :", pd.isna(yhat).sum())
    print(" Prévisions générées")

    # --- Évaluation ---
    m_mae = mae(test.values, yhat.values)
    m_rmse = rmse(test.values, yhat.values)
    print(f"ARIMA - MAE: {m_mae:.3f} | RMSE: {m_rmse:.3f}")
    
    # ANALYSE DE LA VARIABILITÉ ARIMA
    print("\n" + "="*50)
    print(" ANALYSE DE LA VARIABILITÉ ARIMA")
    print("="*50)

    y_true_std = test.std()
    y_pred_std = yhat.std()
    variability_ratio = y_pred_std / y_true_std
    print(f"Valeurs réelles - Min: {test.min():.3f}, Max: {test.max():.3f}, Std: {y_true_std:.3f}")
    print(f"Prédictions     - Min: {yhat.min():.3f}, Max: {yhat.max():.3f}, Std: {y_pred_std:.3f}")
    print(f"Ratio variabilité (prédictions/réel): {variability_ratio:.3f}")

    if variability_ratio < 0.5:
        print("  Le modèle ne capture pas assez la variabilité!")
    elif variability_ratio > 1.5:
        print("  Le modèle est trop variable!")
    elif variability_ratio > 0.75:
        print(" EXCELLENT - Capture >75% de la variabilité réelle")
    elif variability_ratio > 0.6:
        print(" BON - Capture >60% de la variabilité réelle")
    else:
        print(" Bonne capture de la variabilité")
        
    #  RESULTATS FINAUX
    print(f"\n RESULTATS FINAUX ")
    print(f"    ARIMA   - MAE: {m_mae:.3f} | RMSE: {m_rmse:.3f} | Variabilité: {variability_ratio:.3f}")")

    # --- Sauvegardes ---
    os.makedirs(os.path.dirname(cfg.output.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.data.forecast), exist_ok=True)

    arima.save(cfg.output.model_path)
    pd.DataFrame({
        "y_true": test.values,
        "y_pred": yhat.values
    }, index=test.index).to_csv(cfg.data.forecast)

    print(f"Modèle sauvegardé, {cfg.output.model_path}")
    print(f"Prévisions sauvegardées,  {cfg.data.forecast}")
    print(yhat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement ARIMA")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier config.yaml")
    args = parser.parse_args()
    main(args.config)
