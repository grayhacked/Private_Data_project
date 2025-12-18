"""
Constantes et configurations du projet
"""

# Colonnes à exclure impérativement des features
COLUMNS_TO_EXCLUDE = [
    # Cibles
    'vues_future_28d_log',
    'vues_future_28d',
    
    # Données brutes (on garde les versions log/transformées)
    'vues_lr', 
    'clics_lr', 
    'va_lr', 
    'commandes_lr',
    
    # Identifiants
    'search_id',
    
    # Colonnes temporelles brutes (on a les versions cycliques)
    'mois',
    'jour_de_lannee', 
    'jour_de_la_semaine',
    'semaine_de_lannee',
    'jour_du_mois',
    'trimestre',
    'annee',
]

# Features à conserver natives pour LightGBM
NATIVE_CATEGORICAL_FEATURES = [
    'cat1',
    'segment_popularite',
    'tail_segment',
    'type_solde',
    'type_solde_future',
]

# Features à target encoder
FEATURES_TO_ENCODE = [
    'cat2',
    'cat3',
]

# Fenêtres de lag (jours avant J+28)
LAG_WINDOWS_RELATIVE = [1, 7, 14, 28]

# Features pour les lags
FEATURES_FOR_LAG = ['vues_lr_log', 'clics_lr_log']

# Fenêtres de rolling
ROLLING_WINDOWS = [7, 14, 28, 60]

# Segments de popularité
POPULARITY_SEGMENTS = ['HEAD', 'MID', 'TAIL']

# Métriques d'évaluation
EVALUATION_METRICS = {
    'mae': 'Mean Absolute Error',
    'rmse': 'Root Mean Squared Error',
    'r2': 'R-squared',
    'wape': 'Weighted Absolute Percentage Error'
}

# Coûts business
BUSINESS_COSTS = {
    'false_negative': 300_000,  # Coût rupture stock
    'false_positive': 50_000,   # Coût sur-stockage
}