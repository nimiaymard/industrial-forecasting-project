"""
Télécharge le jeu de données Numenta Anomaly Benchmark (NAB) sous forme de fichiers CSV.

Sources :
- GitHub : https://github.com/numenta/NAB


Ce script télécharge le dépôt complet au format ZIP,
et copie les fichiers CSV du dossier 'data/realKnownCause' 
vers le répertoire local 'data/raw/nab/'.
"""

# Importation des modules nécessaires
import os, io, zipfile, requests, tempfile, shutil, glob

# -----------------------------------------------------------
# Fonction principale exécutée lorsque le script est lancé
# -----------------------------------------------------------
def main():
    # URL du fichier ZIP contenant le dépôt GitHub NAB
    url = 'https://github.com/numenta/NAB/archive/refs/heads/master.zip'
    
    # Téléchargement du contenu du dépôt (en mémoire)
    r = requests.get(url, timeout=60)   # timeout=60 : arrête si le téléchargement prend plus de 60 secondes
    r.raise_for_status()                # Lève une erreur si le téléchargement échoue (ex : 404, 500)
    
    # Création d’un dossier temporaire pour extraire le contenu du ZIP
    with tempfile.TemporaryDirectory() as td:
        # Lecture du contenu ZIP directement depuis la mémoire (sans fichier local)
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        
        # Extraction de tous les fichiers dans le dossier temporaire
        zf.extractall(td)

        # Définition du chemin source contenant les fichiers CSV d'intérêt
        # -> "realKnownCause" contient les séries temporelles avec anomalies connues
        src = os.path.join(td, 'NAB-master', 'data', 'realKnownCause')
        
        # Recherche de tous les fichiers CSV dans ce dossier
        csvs = glob.glob(os.path.join(src, '*.csv'))
        
        # Dossier de destination dans ton projet local
        dst = os.path.join('data', 'raw', 'nab')
        
        # Création du dossier s’il n’existe pas déjà
        os.makedirs(dst, exist_ok=True)
        
        # Copie de chaque fichier CSV trouvé vers le dossier local
        for p in csvs:
            shutil.copy(p, dst)
        
        # Message récapitulatif indiquant le nombre de fichiers copiés
        print(f'Copied {len(csvs)} files to {dst}')

# -----------------------------------------------------------
# Point d’entrée du script : exécute main() si lancé directement
# -----------------------------------------------------------
if __name__ == '__main__':
    main()
