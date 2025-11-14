import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from industrial_forecasting.utils.config import load_config
from industrial_forecasting.data import train_test_split_series
from industrial_forecasting.models.prophet import ProphetForecaster
from industrial_forecasting.evaluate import mae, rmse

def analyze_prophet_components(model, forecast_df, test_df):
    """Analyse détaillée des composantes du modèle Prophet - VERSION CORRIGÉE"""
    print("\n" + "="*60)
    print(" ANALYSE DÉTAILLÉE DES COMPOSANTES PROPHET - CORRIGÉE")
    print("="*60)
    
    # CORRECTION: Utiliser les valeurs centrées pour l'analyse de variabilité
    yhat_mean = forecast_df['yhat'].mean()
    
    print("\n VARIABILITÉ PAR COMPOSANTE (centrée):")
    components = ['trend', 'daily', 'weekly', 'yearly', 'ultra_rapid', 'rapid_hourly']
    total_std = forecast_df['yhat'].std()
    
    print(f"  {'Composante':12} | {'Std':8} | {'% Total':7}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*7}")
    
    for component in components:
        if component in forecast_df.columns:
            # CORRECTION: Centrer les composantes avant calcul
            comp_centered = forecast_df[component] - forecast_df[component].mean()
            comp_std = comp_centered.std()
            percentage = (comp_std / total_std) * 100 if total_std > 0 else 0
            print(f"  {component:12} | {comp_std:7.4f} | {percentage:6.1f}%")
    
    # ANALYSE DES SAISONNALITÉS RÉELLES
    print(f"\n IMPACT RÉEL DES SAISONNALITÉS:")
    seasonal_components = ['daily', 'weekly', 'yearly']
    seasonal_effects = {}
    
    for comp in seasonal_components:
        if comp in forecast_df.columns:
            comp_centered = forecast_df[comp] - forecast_df[comp].mean()
            effect_range = comp_centered.max() - comp_centered.min()
            seasonal_effects[comp] = effect_range
            print(f"  {comp:12} : Amplitude réelle = {effect_range:.4f}")
    
    # IDENTIFICATION DE LA COMPOSANTE DOMINANTE
    print(f"\n COMPOSANTE LA PLUS INFLUENTE:")
    if seasonal_effects:
        dominant_comp = max(seasonal_effects, key=seasonal_effects.get)
        print(f"  → {dominant_comp}: Amplitude = {seasonal_effects[dominant_comp]:.4f}")
    
    # ANALYSE DE LA TREND (CORRIGÉE)
    print(f"\n ANALYSE DE LA TENDANCE RÉELLE:")
    trend_data = forecast_df['trend']
    trend_variation = trend_data.max() - trend_data.min()
    print(f"  Variation trend: {trend_variation:.2f}")
    print(f"  Pente moyenne: {(trend_data.iloc[-1] - trend_data.iloc[0]) / len(trend_data) * 100:.4f} par step")
    
    # QUALITÉ DE L'AJUSTEMENT (DÉJÀ CORRECTE)
    print(f"\n QUALITÉ DE L'AJUSTEMENT:")
    residuals = test_df['y'].values - forecast_df.loc[test_df.index, 'yhat'].values
    print(f"  Résidus - Moyenne: {np.mean(residuals):.4f} (idéal ~0)")
    print(f"  Résidus - Écart-type: {np.std(residuals):.4f}")
    
    # RATIO DE VARIABILITÉ RÉEL
    real_variability_ratio = forecast_df.loc[test_df.index, 'yhat'].std() / test_df['y'].std()
    print(f"\n RATIO DE VARIABILITÉ RÉEL: {real_variability_ratio:.3f}")
    
    # DIAGNOSTIC FINAL
    print(f"\n DIAGNOSTIC FINAL:")
    if real_variability_ratio > 0.75:
        print(f"   EXCELLENT - Ratio: {real_variability_ratio:.3f} (>75%)")
    elif real_variability_ratio > 0.6:
        print(f"   BON - Ratio: {real_variability_ratio:.3f} (>60%)")
    else:
        print(f"    À AMÉLIORER - Ratio: {real_variability_ratio:.3f}")
    
    # ANALYSE DES COMPOSANTES NULLES
    print(f"\n🔍 COMPOSANTES NÉGLIGEABLES:")
    negligible_components = []
    for comp in ['daily', 'weekly', 'ultra_rapid', 'rapid_hourly']:
        if comp in forecast_df.columns:
            comp_centered = forecast_df[comp] - forecast_df[comp].mean()
            if comp_centered.std() < 0.1:  # Seuil d'impact négligeable
                negligible_components.append(comp)
    
    if negligible_components:
        print(f"  Composantes à désactiver: {', '.join(negligible_components)}")

def main(cfg_path):
    cfg = load_config(cfg_path)
    print(" Configuration chargée")

    # Charger la série
    df = pd.read_csv(
        cfg.data.raw_path,
        parse_dates=[cfg.data.datetime_col],
        index_col=cfg.data.datetime_col
    )
    df = df.rename(columns={cfg.data.value_col: "y"})
    df = df.reset_index().rename(columns={cfg.data.datetime_col: "ds"})
    print(f"Série chargée : {len(df)} lignes")
    
    # Train/Test split
    s = pd.Series(df["y"].values, index=pd.DatetimeIndex(df["ds"]))
    print(s.head())
    print(s.index.is_monotonic_increasing)
    print(s.isna().sum())

    train, test = train_test_split_series(s, cfg.data.train_ratio)
    print(f"Train : {len(train)} obs | Test : {len(test)} obs")
    print("NaN dans train :", train.isna().sum())
    print("NaN dans test :", test.isna().sum())
    train = train.interpolate()
    test = test.interpolate()
    print("NaN dans train :", train.isna().sum())
    print("NaN dans test :", test.isna().sum())
    print(train.describe())

    # Préparer DataFrame pour Prophet
    train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
    test_df = pd.DataFrame({'ds': test.index, 'y': test.values})

    # Instanciation et entraînement
    model = ProphetForecaster(
        yearly_seasonality=cfg.prophet.yearly_seasonality,
        weekly_seasonality=cfg.prophet.weekly_seasonality,
        daily_seasonality=cfg.prophet.daily_seasonality,
        seasonality_mode=cfg.prophet.seasonality_mode,
        changepoint_prior_scale=cfg.prophet.changepoint_prior_scale,
        seasonality_prior_scale=cfg.prophet.seasonality_prior_scale,
        holidays_prior_scale=cfg.prophet.holidays_prior_scale,
        changepoint_range=cfg.prophet.changepoint_range
    )

    print("\n DÉBUT DE L'ENTRAÎNEMENT PROPHET...")
    model.fit(train_df)
    print(" Entraînement Prophet terminé")

    # Prévisions
    forecast_df = model.forecast(steps=len(test_df), freq=cfg.data.freq)

    # DEBUG : Afficher les colonnes
    print("\n Colonnes disponibles dans forecast :", forecast_df.columns)
    
    print(" Prévisions générées")
    
    # Extraire uniquement la colonne 'yhat'
    yhat = forecast_df.loc[test_df.index, "yhat"]
    print(" Aperçu des prévisions:")
    print(yhat.head())

    #  ANALYSE DES COMPOSANTES
    analyze_prophet_components(model, forecast_df, test_df)

    #  ANALYSE DE LA VARIABILITÉ
    print("\n" + "="*50)
    print(" ANALYSE DE LA VARIABILITÉ")
    print("="*50)
    
    y_true_std = test_df["y"].std()
    y_pred_std = yhat.std()
    variability_ratio = y_pred_std / y_true_std
    
    print(f"Valeurs réelles - Min: {test_df['y'].min():.3f}, Max: {test_df['y'].max():.3f}, Std: {y_true_std:.3f}")
    print(f"Prédictions    - Min: {yhat.min():.3f}, Max: {yhat.max():.3f}, Std: {y_pred_std:.3f}")
    print(f"Ratio variabilité (prédictions/réel): {variability_ratio:.3f}")
    
    if variability_ratio < 0.5:
        print("  Le modèle ne capture pas assez la variabilité des données!")
    elif variability_ratio > 1.5:
        print("  Le modèle est trop variable!")
    else:
        print(" Bonne capture de la variabilité")

    # Évaluation
    m_mae = mae(test_df["y"].values, yhat.values)
    m_rmse = rmse(test_df["y"].values, yhat.values)
    print(f"\n PERFORMANCE FINALE - Prophet - MAE: {m_mae:.3f} | RMSE: {m_rmse:.3f}")

    # Sauvegarde des prévisions
    os.makedirs(os.path.dirname(cfg.data.forecast_path_prophet), exist_ok=True)
    pd.DataFrame({
        "ds": test_df["ds"].values,
        "y_true": test_df["y"].values,
        "y_pred": yhat.values
    }).to_csv(cfg.data.forecast_path_prophet, index=False)
    print(f" Prévisions sauvegardées : {cfg.data.forecast_path_prophet}")

    # Sauvegarde du modèle
    os.makedirs(os.path.dirname(cfg.output.model_path_prophet), exist_ok=True)
    model.save(cfg.output.model_path_prophet)
    print(f" Modèle sauvegardé : {cfg.output.model_path_prophet}")

    print("\n" + "*" * 20)
    print(" ENTRAÎNEMENT PROPHET TERMINÉ AVEC SUCCÈS!")
    print("\n" + "*" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Prophet")
    parser.add_argument("--config", required=True, help="Chemin vers config.yaml")
    args = parser.parse_args()
    main(args.config)