# Analyse & Prévision de Données Industrielles

Projet personnel prêt pour GitHub : **modélisation de séries temporelles (ARIMA, LSTM)** et **détection d’anomalies** pour des flux industriels (capteurs, production, consommation).

## ✨ Contenu
- **ARIMA (statsmodels)** pour la prévision classique
- **LSTM (PyTorch)** pour la prévision deep learning
- **Détection d'anomalies** (IsolationForest + z-score robuste)
- **Pipeline simple**: chargement → prétraitement → entraînement → évaluation → export
- **Données synthétiques** reproductibles (pour tester sans données privées)
- **Scripts CLI** reproductibles + **config YAML**

## 📁 Arborescence
```
industrial-forecasting-project/
├── data/
│   ├── raw/            # Fichiers bruts (synthetic.csv fourni)
│   └── processed/      # Données nettoyées/features
├── models/             # Modèles entraînés (.pkl/.pt)
├── notebooks/          # (Option) analyses exploratoires
├── scripts/            # Scripts CLI (train/eval/anomaly)
├── src/industrial_forecasting/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── evaluate.py
│   ├── visualization.py
│   ├── models/
│   │   ├── arima.py
│   │   └── lstm.py
│   └── utils/
│       ├── config.py
│       └── paths.py
├── config.yaml
├── requirements.txt
├── Makefile
└── README.md
```

## 🚀 Démarrage rapide
```bash
# 1) Créer l'environnement (ex. conda ou venv)
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
#  Windows: .venv\Scripts\activate

# 2) Installer les dépendances
pip install -r requirements.txt

# 3) Vérifier que les données synthétiques existent (déjà incluses)
ls data/raw/real.csv

# 4) Entraîner ARIMA
python scripts/train_arima.py --config config.yaml

# 5) Entraîner LSTM (PyTorch)
python scripts/train_lstm.py --config config.yaml

# 6) Détecter les anomalies
python scripts/detect_anomalies.py --config config.yaml

# 7) Évaluer les prévisions (ARIMA ou LSTM)
python scripts/evaluate_forecasts.py --config config.yaml --model arima
python scripts/evaluate_forecasts.py --config config.yaml --model lstm
```

## ⚙️ Configuration (config.yaml)
- Chemins de fichiers, colonnes des données, fréquence temporelle
- Paramètres ARIMA (p,d,q)
- Hyperparamètres LSTM (fenêtre, hidden_size, lr, epochs)
- Paramètres de détection d’anomalies

## 🧪 Données
Par défaut, **`data/raw/synthetic.csv`** contient un flux industriel synthétique (tendance + saisonnalité + bruit + anomalies injectées) pour tester l’end-to-end.

## 📝 Licence
MIT — libre d’utilisation à des fins d’apprentissage et démonstration.


## 📦 Jeux de données réels (Open)
- **SKAB (Skoltech Anomaly Benchmark)** — capteurs industriels avec anomalies étiquetées. Script: `python scripts/fetch_skab.py` → génère `data/raw/skab_single.csv`.
- **NAB (Numenta Anomaly Benchmark)** — plus de 50 séries réelles/étiquetées. Script: `python scripts/fetch_nab.py` → copies dans `data/raw/nab/`.

### Exemple d'usage (remplacer les chemins dans `config.yaml`)
```yaml
data:
  raw_path: "data/raw/skab_single.csv"
  datetime_col: "timestamp"
  value_col: "value"
  freq: "H"
  train_ratio: 0.8
```

## 📦 Jeux de données réels (sans Kaggle)
- **NAB** (Numenta) – anomalies réelles : `python scripts/fetch_data.py --dataset nab`
- **SKAB** (Skoltech) – capteurs banc d’essai : `python scripts/fetch_data.py --dataset skab`
- **UCI SECOM** (semi-conducteurs) – process industriel tabulaire : `python scripts/fetch_data.py --dataset secom`

> Les fichiers sont enregistrés comme `data/raw/real.csv` (colonnes `timestamp,value`). Mettez ensuite `data.raw_path: data/raw/real.csv` dans `config.yaml`.
