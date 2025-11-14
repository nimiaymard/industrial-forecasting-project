import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from industrial_forecasting.utils.config import load_config

def plot_sarima_forecast(cfg):
    # Chargement des données
    df = pd.read_csv(cfg.data.forecast, parse_dates=True, index_col=0)
    
    print(f"Type de l'index : {type(df.index)}")
    print(f"Index : {df.index}")
    print(f"Plage temporelle : {df.index.min()} à {df.index.max()}")
    print(f"Nombre de points : {len(df)}")
    
    # Graphique amélioré avec les informations temporelles
    plt.figure(figsize=(14, 8))
    
    # Tracer les séries
    plt.plot(df.index, df["y_true"], 
             label="Valeurs réelles", 
             linewidth=2.5,
             color='#2E86AB',
             marker='o', markersize=2, alpha=0.8)
    
    plt.plot(df.index, df["y_pred"], 
             label="Prédictions SARIMA", 
             linestyle="--", 
             linewidth=2.5,
             color='#A23B72',
             marker='s', markersize=2)
    
    # Personnalisation avancée
    plt.title("Prédictions SARIMA vs Valeurs Réelles\n(Données horaires du 23/03/2014 au 28/05/2014)", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel("Date et Heure", fontsize=12, fontweight='bold')
    plt.ylabel("Valeur", fontsize=12, fontweight='bold')
    
    # Légende et grille
    plt.legend(fontsize=11, framealpha=0.9, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Gestion optimisée des dates pour données horaires
    plt.gcf().autofmt_xdate()
    
    # Calcul et affichage des métriques
    mae = np.mean(np.abs(df["y_true"] - df["y_pred"]))
    rmse = np.sqrt(np.mean((df["y_true"] - df["y_pred"])**2))
    
    # Ajouter les métriques dans une boîte de texte
    textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nN: {len(df)} points'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    out_path = getattr(cfg.data, "image_png_sarimax", "forecast_sarimax.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f" Graphique amélioré sauvegardé → {out_path}")
    
    print(f"\n Métriques de performance :")
    print(f"   MAE: {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plot_sarima_forecast(cfg)