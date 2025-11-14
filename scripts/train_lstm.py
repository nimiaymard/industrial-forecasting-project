import argparse
import os
import torch
import pandas as pd
import joblib

from industrial_forecasting.utils.config import load_config
from industrial_forecasting.data import  train_test_split_series
from industrial_forecasting.features import create_supervised_from_series
from industrial_forecasting.models.lstm import train_lstm, predict_lstm
from industrial_forecasting.evaluate import mae, rmse
from sklearn.preprocessing import MinMaxScaler



def main(config_path):
    # --- Chargement de la configuration ---
    cfg = load_config(config_path)
    print(" Configuration chargée")

    # --- Chargement de la série depuis CSV ---
    print(f" Chargement série depuis : {cfg.data.raw_path}")
    s = pd.read_csv(
        cfg.data.raw_path,
        parse_dates=[cfg.data.datetime_col],
        index_col=cfg.data.datetime_col
    )[cfg.data.value_col].astype(float).sort_index()
    print(f"Série chargée : {len(s)} lignes")

    # --- Forcer la fréquence si spécifiée ---
    if cfg.data.freq:
        s = s.asfreq(cfg.data.freq.lower())
        print(f"Fréquence forcée à : {cfg.data.freq.lower()}")

    # --- Split train/test ---
    train, test = train_test_split_series(s, cfg.data.train_ratio)
    print(f"Train : {len(train)} obs | Test : {len(test)} obs")  # DEBUG
    print("NaN dans train :", train.isna().sum())  # Vérifie les valeurs manquantes
    print("NaN dans test :", test.isna().sum())
    train = train.interpolate()  # Interpolation pour remplir les valeurs manquantes
    test = test.interpolate()
    print("NaN dans train :", train.isna().sum())  # Vérifie à nouveau après traitement
    print("NaN dans test :", test.isna().sum())
    print(train.describe())
    
    # Normalisation (fit sur train uniquement pour éviter la fuite)
    scaler = MinMaxScaler()
    train_scaled = pd.Series(scaler.fit_transform(train.values.reshape(-1, 1)).flatten(), index=train.index)
    test_scaled = pd.Series(scaler.transform(test.values.reshape(-1, 1)).flatten(), index=test.index)

    # --- Création des ensembles supervisés ---
    window_size = int(cfg.lstm.window_size)
    X_train, y_train = create_supervised_from_series(train_scaled, window_size)
    
    # --- Fenêtres supervisées ---
    window = int(cfg.lstm.window_size)
    X_train, y_train = create_supervised_from_series(train_scaled, window)

    full_series = pd.concat([train_scaled, test_scaled])
    X_full, y_full = create_supervised_from_series(full_series, window)
    X_test = X_full[-len(test):]
    y_test = y_full[-len(test):]

    # --- Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Utilisation du device : {device}")

    # Récupération du paramètre dropout (avec valeur par défaut 0.2 si absent)
    dropout = getattr(cfg.lstm, "dropout", 0.2)

    # Entraînement du modèle LSTM
    model = train_lstm(
    X_train, y_train,
    X_test, y_test,
    hidden_size=cfg.lstm.hidden_size,
    num_layers=cfg.lstm.num_layers,
    lr=float(cfg.lstm.lr),
    epochs=int(cfg.lstm.epochs),
    batch_size=int(cfg.lstm.batch_size),
    device=device,
    dropout=dropout  #
)

    # --- Prédiction & Inversion ---
    y_pred = predict_lstm(model, X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # --- Évaluation ---
    m_mae, m_rmse = mae(y_test, y_pred), rmse(y_test, y_pred)
    print(f" LSTM - MAE: {m_mae:.3f} | RMSE: {m_rmse:.3f}")
    
    # ANALYSE DE LA VARIABILITÉ
    print("\n" + "="*50)
    print(" ANALYSE DE LA VARIABILITÉ LSTM")
    print("="*50)

    # Calcul du ratio de variabilité
    y_true_std = y_test.std()
    y_pred_std = y_pred.std()
    variability_ratio = y_pred_std / y_true_std

    print(f"Valeurs réelles  - Min: {y_test.min():.3f}, Max: {y_test.max():.3f}, Std: {y_true_std:.3f}")
    print(f"Prédictions LSTM - Min: {y_pred.min():.3f}, Max: {y_pred.max():.3f}, Std: {y_pred_std:.3f}")
    print(f"Ratio variabilité (prédictions/réel): {variability_ratio:.3f}")
    
    # Diagnostic
    if variability_ratio < 0.5:
        print("  Le modèle ne capture pas assez la variabilité des données!")
    elif variability_ratio > 1.5:
        print("  Le modèle est trop variable!")
    elif variability_ratio > 0.75:
       print(" EXCELLENT - Capture >75% de la variabilité réelle")
    elif variability_ratio > 0.6:
        print(" BON - Capture >60% de la variabilité réelle")
    else:
        print(" Bonne capture de la variabilité")
    #  COMPARAISON AVEC PROPHET
    print(f"\n COMPARAISON DES MODÈLES:")
    print(f"   LSTM    - MAE: {m_mae:.3f} | RMSE: {m_rmse:.3f} | Variabilité: {variability_ratio:.3f}")

    # --- Sauvegarde ---
    model_save_path = cfg.output.model_path_lstm  # Utilise le chemin de la config
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Sauvegarde du modèle LSTM
    torch.save(model.state_dict(), model_save_path)
    print(f" Modèle LSTM sauvegardé → {model_save_path}")
    
    # Sauvegarde du scaler (important pour les prédictions futures)
    scaler_save_path = model_save_path.replace('.pkl', '_scaler.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f" Scaler sauvegardé → {scaler_save_path}")

    # Sauvegarde des prédictions dans un CSV
    forecast_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })

    output_path = cfg.data.forecast_path_lstm
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path)
    print(f" Prédictions sauvegardées → {output_path}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Chemin vers le fichier de configuration YAML')
    args = parser.parse_args()
    main(args.config)
    
    