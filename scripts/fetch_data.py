#!/usr/bin/env python3
"""
Récupérer de vrais jeux de données ouverts (sans authentification Kaggle)
et placer une série temporelle unique dans data/raw/real.csv

Jeux de données pris en charge :

- NAB (Numenta Anomaly Benchmark) : sélectionne un fichier CSV contenant des anomalies étiquetées
- SKAB (Skoltech Anomaly Benchmark) : sélectionne un signal provenant d’un capteur
- UCI SECOM (manufacturing) : charge des données tabulaires de processus et crée un index temporel synthétique (ce n’est donc pas une vraie série temporelle)
"""

# Importation des modules nécessaires
import argparse, os, io, pandas as pd, numpy as np, requests

# Dossier de destination des données téléchargées
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)  # Crée le dossier s’il n’existe pas déjà

# -----------------------------------------------------------
# Fonction utilitaire pour télécharger un fichier depuis une URL
# -----------------------------------------------------------
def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=60)  # Télécharge le contenu avec un timeout de 60 secondes
    resp.raise_for_status()               # Lève une erreur si le téléchargement échoue
    return resp.content                   # Retourne le contenu brut (bytes)

# -----------------------------------------------------------
# Récupération du dataset NAB
# -----------------------------------------------------------
def fetch_nab():
    """
    Télécharge un fichier du dataset NAB :
    'realKnownCause/ambient_temperature_system_failure.csv'
    Source : https://github.com/numenta/NAB
    """
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv"
    csv = _download(url).decode("utf-8")     # Téléchargement + décodage du CSV en texte
    df = pd.read_csv(io.StringIO(csv))       # Lecture du CSV en DataFrame pandas
    # Le fichier NAB contient les colonnes : timestamp, value
    out = os.path.join(DATA_DIR, "real.csv") # Chemin de sortie
    df.to_csv(out, index=False)              # Sauvegarde sous data/raw/real.csv
    print(f"[NAB] Saved {out} ({len(df)} rows)")  # Message de confirmation

# -----------------------------------------------------------
# Récupération du dataset SKAB
# -----------------------------------------------------------
def fetch_skab():
    """
    Télécharge un jeu de données SKAB (un canal de capteur)
    Source : https://github.com/waico/SKAB
    On prend ici un exemple "anomaly-free" (sans anomalies) nommé run1,
    et on crée des étiquettes artificielles (0). L’utilisateur peut ensuite modifier.
    """
    base_url = "https://raw.githubusercontent.com/waico/SKAB/master/data/"
    candidate = "train/1.csv"                   # Choix du fichier d’entraînement
    url = base_url + candidate
    csv = _download(url).decode("utf-8", errors="ignore")  # Téléchargement du CSV
    df = pd.read_csv(io.StringIO(csv))          # Lecture du CSV
    
    # On cherche la première colonne numérique (autre que 'timestamp')
    value_col = None
    for c in df.columns:
        if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break
    if value_col is None:
        raise RuntimeError("Impossible de trouver une colonne numérique dans le fichier SKAB.")
    
    # On garde uniquement 'timestamp' + la colonne numérique choisie
    out_df = df[["timestamp", value_col]].rename(columns={value_col: "value"})
    out = os.path.join(DATA_DIR, "real.csv")
    out_df.to_csv(out, index=False)
    print(f"[SKAB] Saved {out} with column '{value_col}' ({len(out_df)} rows)")

# -----------------------------------------------------------
# Récupération du dataset SECOM
# -----------------------------------------------------------
def fetch_secom():
    """
    Le dataset SECOM de l’UCI est tabulaire (pas de vraie dimension temporelle).
    On va donc créer un index temporel artificiel et choisir une colonne de capteur
    pour en faire une pseudo-série temporelle.

    Source : https://archive.ics.uci.edu/ml/datasets/SECOM
    """
    # URLs du dataset SECOM
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.labels"

    # Téléchargement du fichier principal (les labels ne sont pas utilisés ici)
    data_raw = _download(data_url).decode("utf-8", errors="ignore")

    # Le fichier contient des valeurs séparées par des espaces, parfois 'NaN'
    rows = []
    for line in data_raw.strip().splitlines():
        parts = [p for p in line.strip().split(" ") if p != ""]
        # Conversion des valeurs en float, gestion des NaN
        rows.append([float(x) if x != "NaN" else np.nan for x in parts])
    df = pd.DataFrame(rows)

    # Sélectionne la colonne avec le moins de valeurs manquantes
    nan_counts = df.isna().sum()
    value_col = int(nan_counts.idxmin())

    # Interpolation et remplissage des valeurs manquantes
    s = df[value_col].interpolate().fillna(method="bfill").fillna(method="ffill")

    # Création d’un index temporel artificiel (une heure d’écart entre chaque point)
    ts = pd.date_range("2024-01-01", periods=len(s), freq="H")

    # Construction du DataFrame final : timestamp + value
    out_df = pd.DataFrame({"timestamp": ts, "value": s.values})
    out = os.path.join(DATA_DIR, "real.csv")
    out_df.to_csv(out, index=False)
    print(f"[SECOM] Saved {out} from sensor col {value_col} ({len(out_df)} rows)")

# -----------------------------------------------------------
# Fonction principale qui choisit quel dataset télécharger
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nab","skab","secom"])
    args = ap.parse_args()

    # En fonction du paramètre, appelle la bonne fonction
    if args.dataset == "nab":
        fetch_nab()
    elif args.dataset == "skab":
        fetch_skab()
    elif args.dataset == "secom":
        fetch_secom()

# -----------------------------------------------------------
# Point d’entrée du script
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
