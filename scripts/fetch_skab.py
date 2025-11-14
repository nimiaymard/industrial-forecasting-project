"""
Télécharge le jeu de données **SKAB (Skoltech Anomaly Benchmark)** depuis GitHub.

Source : https://github.com/waico/SKAB  
Licence : GPLv3 — à vérifier avant toute redistribution.

Ce script :
- Télécharge les fichiers CSV du dépôt SKAB ;
- Peut soit :
    - enregistrer **une seule série temporelle** dans `data/raw/skab_single.csv`,  
    - ou fusionner **toutes les séries** en un grand fichier unique `data/raw/skab_all.csv`.
"""

# Importation des modules nécessaires
import argparse, os, io, zipfile, pandas as pd, requests, tempfile, glob

# -----------------------------------------------------------
# Fonction principale : télécharge et traite les données SKAB
# -----------------------------------------------------------
def main(series='single'):
    # URL du dépôt SKAB (téléchargé sous forme d’archive ZIP)
    repo_zip = 'https://github.com/waico/SKAB/archive/refs/heads/master.zip'
    
    # Téléchargement du contenu ZIP depuis GitHub
    r = requests.get(repo_zip, timeout=60)
    r.raise_for_status()  # Lève une erreur si le téléchargement échoue
    
    # Création d’un dossier temporaire pour travailler sans encombrer le projet
    with tempfile.TemporaryDirectory() as td:
        # Lecture du ZIP en mémoire
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        # Extraction du contenu dans le dossier temporaire
        zf.extractall(td)
        
        # Chemin racine des fichiers CSV dans le dépôt SKAB
        root = os.path.join(td, 'SKAB-master', 'data')
        
        # Liste de tous les fichiers CSV trouvés dans /data/
        csvs = sorted(glob.glob(os.path.join(root, '*.csv')))
        
        # Crée le dossier cible dans le projet (data/raw)
        os.makedirs('data/raw', exist_ok=True)
        
        # ---------------------------------------------------
        # CAS 1 : On veut une seule série temporelle ("single")
        # ---------------------------------------------------
        if series == 'single':
            # On prend le premier fichier CSV comme exemple représentatif
            df = pd.read_csv(csvs[0])
            
            # Certains fichiers utilisent la colonne "datetime" → on la renomme en "timestamp"
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            
            # On garde uniquement les colonnes essentielles
            df[['timestamp','anomaly','value']].to_csv('data/raw/skab_single.csv', index=False)
            
            # Message de confirmation
            print('Saved data/raw/skab_single.csv')
        
        # ---------------------------------------------------
        # CAS 2 : On veut fusionner toutes les séries ("all")
        # ---------------------------------------------------
        else:
            frames = []  # liste pour stocker les DataFrames
            
            # Boucle sur chaque fichier CSV
            for path in csvs:
                # Nom du fichier sans extension (sert à identifier la série)
                name = os.path.basename(path).replace('.csv','')
                
                # Lecture du CSV
                df = pd.read_csv(path)
                
                # Renommer 'datetime' en 'timestamp' si présent
                if 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'timestamp'})
                
                # Ajouter une colonne 'series' pour indiquer le nom du fichier d’origine
                df['series'] = name
                
                # Ajouter ce DataFrame à la liste
                frames.append(df)
            
            # Fusionner toutes les séries en un seul grand DataFrame
            out = pd.concat(frames, ignore_index=True)
            
            # Sauvegarder le résultat global
            out.to_csv('data/raw/skab_all.csv', index=False)
            
            print('Saved data/raw/skab_all.csv')

# -----------------------------------------------------------
# Point d’entrée du script (gestion des arguments CLI)
# -----------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # Option --series : permet de choisir entre "single" (une série) ou "all" (toutes les séries)
    ap.add_argument('--series', choices=['single','all'], default='single')
    args = ap.parse_args()
    
    # Exécute la fonction principale avec le paramètre choisi
    main(args.series)
