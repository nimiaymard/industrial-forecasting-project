# Analyse & Prévision de Données Industrielles (ARIMA, LSTM, Prophet) + POC Web Frontend en React

Projet complet pour la modélisation de séries temporelles industrielles :
prévision, détection d’anomalies, visualisation web et architecture modulaire.

##   Fonctionnalités principales
- **Prévision** :
 - ARIMA (statsmodels): baseline statistique rapide
 - LSTM (PyTorch): modèle deep learning optimisé (normalisation, dropout, fenêtre temporelle)
 - Prophet (Meta) : prévision avec saisonnalités multiples et robustesse

- **Détection d’anomalies** :
 - IsolationForest
 - Z-score robuste (MAD)

- **Pipeline complet** :
 - Chargement des données
 - Nettoyage et interpolation
 - Feature engineering
 - Entraînement : sauvegarde modèle
 - Prévision : export CSV
 - Visualisation (Python + Frontend React)

- **Frontend React / Vite :**
 - Un prototype web simple permet de :
  - charger les prévisions LSTM (CSV),
  - visualiser les courbes (réel vs prédit),
  - tester le modèle avant déploiement,
  - intégrer facilement un futur backend Flask/FastAPI.

##  Arborescence
```
industrial-forecasting-project/
├── data/
│   ├── raw/               # Données originales
│   └── processed/         # Données nettoyées, prévisions (.csv)
│
├── models/                # Modèles sauvegardés (.pkl)
│
├── notebooks/             # Analyse exploratoire 
│
├── scripts/               # Scripts CLI (train, eval, anomalies)
│
├── src/industrial_forecasting/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── evaluate.py
│   ├── visualize_prophet.py
│   ├── visualize_lstm.py
│   ├── models/
│   │     ├── arima.py
│   │     ├── lstm.py
│   │     └── prophet.py
│   └── utils/
│         ├── config.py
│         └── paths.py
│
├── frontend/                     # Prototype web React
│   ├── public/
│   │     └── index.html
│   ├── src/
│   │     ├── App.jsx
│   │     ├── components/
│   │     │     └── ForecastChart.jsx
│   │     ├── services/
│   │     │     └── api.js
│   │     └── styles/
│   │           └── App.css
│   └── package.json
│
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


##  Configuration (config.yaml)
- Chemins de fichiers, colonnes des données, fréquence temporelle
- Paramètres ARIMA (p,d,q)
- Hyperparamètres LSTM (fenêtre, hidden_size, lr, epochs)
- Paramètres de détection d’anomalies

##  Données
Par défaut, **`data/raw/real.csv`** contient un flux industriel synthétique (tendance + saisonnalité + bruit + anomalies injectées) pour tester l’end-to-end.

## Frontend React — Prototype Web
1. Installer Node.js

Si nécessaire : https://nodejs.org
 (version LTS)

2. Installer le frontend
cd frontend
npm install

3. Lancer le prototype
npm run dev


Frontend accessible sur :  http://localhost:5173

Fonction du frontend :

- Charge le fichier forecast_lstm.csv depuis data/processed/
- Trace le graphe réel vs prévisions LSTM via Chart.js
- Composants React propres :
- ForecastChart.jsx
- api.js pour charger les données

##  Licence
MIT — libre d’utilisation à des fins d’apprentissage et démonstration.


##  Jeux de données réels (Open)
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

## Jeux de données réels (sans Kaggle)
- **NAB** (Numenta) – anomalies réelles : `python scripts/fetch_data.py --dataset nab`
- **SKAB** (Skoltech) – capteurs banc d’essai : `python scripts/fetch_data.py --dataset skab`
- **UCI SECOM** (semi-conducteurs) – process industriel tabulaire : `python scripts/fetch_data.py --dataset secom`

> Les fichiers sont enregistrés comme `data/raw/real.csv` (colonnes `timestamp,value`). Mettez ensuite `data.raw_path: data/raw/real.csv` dans `config.yaml`.
