# ...existing code...
import numpy as np                     # numpy pour calculs numériques et manipulation de tableaux
import pandas as pd                    # pandas pour DataFrame / Series
from sklearn.ensemble import IsolationForest  # IsolationForest pour détection d'anomalies non supervisée

def zscore_anomaly(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    # calcule la médiane de la série (robuste aux outliers)
    m = np.median(y)
    # calcule la Median Absolute Deviation (MAD) et ajoute un petit epsilon pour éviter division par zéro
    mad = np.median(np.abs(y - m)) + 1e-9
    # convertit la déviation en un "z-score" robuste comparable à l'écart-type
    z = 0.6745 * (y - m) / mad
    # retourne un tableau d'entiers 0/1 : 1 si la valeur est un outlier (|z| > threshold), 0 sinon
    return (np.abs(z) > threshold).astype(int)

def detect_anomalies_series(
    s: pd.Series,
    method: str = "zscore",
    contamination: float = 0.03,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    # si on choisit la méthode IsolationForest, on entraîne le modèle sur la série
    if method == "isolation_forest":
        # initialise le modèle avec le paramètre contamination (proportion d'anomalies attendue)
        model = IsolationForest(contamination=float(contamination), random_state=42)
        # reshape des données en (n_samples, 1) attendu par scikit-learn
        y = s.values.reshape(-1, 1)
        # fit_predict renvoie -1 pour anomalies, 1 pour normales ; on convertit en 1=anomalie / 0=normal
        labels = (model.fit_predict(y) == -1).astype(int)
    else:
        # sinon on utilise la méthode robuste par score z (MAD) définie ci‑dessus
        labels = zscore_anomaly(s.values, threshold=float(zscore_threshold))
    # construit un DataFrame de sortie avec la valeur d'origine et le label d'anomalie
    out = s.to_frame(name="value")
    out["anomaly"] = labels
    # retourne le DataFrame contenant 'value' et 'anomaly' (0/1)
    return out
# ...existing code...