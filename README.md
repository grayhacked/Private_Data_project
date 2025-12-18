# Introduction 
Ce projet permet de prédire avec précision les Top recherches qui généreront le plus de vues lors des périodes à fort trafic e-commerce (Black Friday, French Days, soldes, etc.).

Étapes pour lancer une prédiction (LightGBM - forecasting)
==========================================================
 
1) Prérequis
    - Python 3.8+
    - Installer dépendances : pandas, numpy, scikit-learn, lightgbm, joblib
    - Fichiers attendus :
      - data/train.csv (historic data)
      - data/new_data.csv (données récentes pour prédiction)
      - scripts/train.py
      - scripts/predict.py
      - dossier models/
 
2) Créer et activer un environnement virtuel (Windows)
    - python -m venv .venv
    - .venv\Scripts\activate
    - pip install -r requirements.txt
 
3) Structure minimale recommandée
    - data/
      - train.csv        (colonnes: date, target, feature_1, feature_2, ...)
      - new_data.csv     (mêmes colonnes features, sans target)
    - scripts/
      - train.py         (entraînement et sauvegarde du modèle)
      - predict.py       (chargement du modèle et génération des prédictions)
    - models/
      - lgbm_model.pkl
 
4) Préparer les données
    - Assurer que la colonne date soit en datetime.
    - Créer les features temporelles (lag, rolling mean, month, dayofweek, etc.).
    - Séparer train / validation selon la date (pas de shuffle pour séries temporelles).
    - Définir le horizon de prédiction (ex: horizon = 7 jours).
 
5) Entraîner et sauvegarder le modèle
    - Exemple de commande :
      python scripts/train.py --data data/train.csv --model models/lgbm_model.pkl --horizon 7
    - Le script doit :
      - charger et prétraiter les données
      - créer features et cibles (déplacer target de horizon pas)
      - entraîner LightGBM
      - sauvegarder le modèle avec joblib.dump(...)
 
6) Lancer une prédiction
    - Exemple de commande :
      python scripts/predict.py --model models/lgbm_model.pkl --input data/new_data.csv --output
      ex (python scripts/predict.py data\raw\data_fresh_bf.parquet -o predictions_bf_2025.csv)
      predictions.csv --horizon 7
    - Le script doit :
      - charger le modèle sauvegardé
      - appliquer le même prétraitement / features que pour l'entraînement
      - générer les prédictions pour l'horizon demandé
      - écrire les résultats (date, prediction) dans predictions.csv
 
7) Évaluation et diagnostic
    - Comparer predictions.csv avec la vérité si disponible.
    - Calculer MAE / RMSE:
      - MAE = mean(|y_true - y_pred|)
      - RMSE = sqrt(mean((y_true - y_pred)^2))
    - Visualiser séries et erreurs (plot)
 
8) Points d'attention
    - Conserver la même pipeline de features entre entraînement et prédiction.
    - Gérer correctement les lags manquants (NaN) pour les premières lignes.
    - Si prévision multi-step, itérer en auto-régressif ou prédire vecteur horizon.
    - Sauvegarder aussi la configuration (features utilisées, paramètres LightGBM).
 
Commandes utiles rapides
    - Création env & install
      python -m venv .venv
      .venv\Scripts\activate
      pip install -r requirements.txt
 
    - Entraînement
      python scripts/train.py --data data/train.csv --model models/lgbm_model.pkl --horizon 7
 
    - Prédiction
      python scripts/predict.py --model models/lgbm_model.pkl --input data/new_data.csv --output predictions.csv --horizon 7