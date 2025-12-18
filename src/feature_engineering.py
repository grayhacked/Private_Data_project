"""
Module de feature engineering
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe pour créer toutes les features"""
    
    def __init__(self, calendar_events: Dict, lag_windows: List[int], 
                 rolling_windows: List[int]):
        """
        Args:
            calendar_events: Dictionnaire des événements calendaires
            lag_windows: Liste des fenêtres de lag
            rolling_windows: Liste des fenêtres de rolling
        """
        self.calendar_events = calendar_events
        self.lag_windows = lag_windows
        self.rolling_windows = rolling_windows
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée toutes les features"""
        logger.info("=" * 80)
        logger.info("CRÉATION DES FEATURES")
        logger.info("=" * 80)
        
        df = self._create_temporal_features(df)
        df = self._create_event_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_trend_features(df)
        df = self._create_interaction_features(df)
        
        # Nettoyage final
        df = self._clean_features(df)
        
        logger.info(f" Toutes les features créées: {len(df.columns)} colonnes")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features temporelles de base"""
        logger.info("\n[1/6] Features temporelles...")
        
        # Features calendaires
        df['annee'] = df.index.year
        df['mois'] = df.index.month
        df['jour_de_la_semaine'] = df.index.dayofweek
        df['trimestre'] = df.index.quarter
        df['is_sunday'] = (df['jour_de_la_semaine'] == 6).astype(np.int8)
        df['is_weekend'] = (df['jour_de_la_semaine'] >= 5).astype(np.int8)
        
        # Features cycliques
        df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
        df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)
        df['jour_de_la_semaine_sin'] = np.sin(2 * np.pi * df['jour_de_la_semaine'] / 7)
        df['jour_de_la_semaine_cos'] = np.cos(2 * np.pi * df['jour_de_la_semaine'] / 7)
        
        logger.info(f"   Features temporelles créées")
        
        return df
    
    def _create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features d'événements"""
        logger.info("\n[2/6] Features événements...")
        
        # Initialisation
        df['est_en_solde'] = 0
        df['type_solde'] = 'Aucun'
        df['intensite_solde'] = 0
        
        # Événements actuels
        for nom_solde, config in self.calendar_events.items():
            mask = df.index.to_series().apply(lambda x: self._date_in_period(x, config))
            df.loc[mask, 'est_en_solde'] = 1
            df.loc[mask, 'type_solde'] = nom_solde
            df.loc[mask, 'intensite_solde'] = config['intensite']
        
        df['type_solde'] = df['type_solde'].astype('category')
        
        # Événements futurs (J+28)
        df['date_future_temp'] = df.index + pd.Timedelta(days=28)
        df['sera_en_solde'] = 0
        df['type_solde_future'] = 'Aucun'
        df['intensite_solde_future'] = 0
        
        for nom_solde, config in self.calendar_events.items():
            mask = df['date_future_temp'].apply(lambda x: self._date_in_period(x, config))
            df.loc[mask, 'sera_en_solde'] = 1
            df.loc[mask, 'type_solde_future'] = nom_solde
            df.loc[mask, 'intensite_solde_future'] = config['intensite']
        
        df['type_solde_future'] = df['type_solde_future'].astype('category')
        df = df.drop('date_future_temp', axis=1)
        
        # Distance aux événements
        df = self._create_event_distance_features(df)
        
        # Features binaires par événement
        df['in_black_friday'] = (df['type_solde'] == 'black_friday').astype(np.int8)
        df['in_pre_blackfriday'] = (df['type_solde'] == 'pre_blackfriday').astype(np.int8)
        df['in_soldes_hiver'] = (df['type_solde'] == 'soldes_hiver').astype(np.int8)
        df['in_soldes_ete'] = (df['type_solde'] == 'soldes_ete').astype(np.int8)
        df['in_periode_noel'] = (df['type_solde'] == 'periode_noel').astype(np.int8)
        
        df['future_black_friday'] = (df['type_solde_future'] == 'black_friday').astype(np.int8)
        df['future_soldes_hiver'] = (df['type_solde_future'] == 'soldes_hiver').astype(np.int8)
        df['future_soldes_ete'] = (df['type_solde_future'] == 'soldes_ete').astype(np.int8)
        
        # Features d'anticipation
        df['in_pre_event_7d'] = ((df['jours_avant_prochain_solde'] > 0) & 
                                (df['jours_avant_prochain_solde'] <= 7)).astype(np.int8)
        df['in_pre_event_14d'] = ((df['jours_avant_prochain_solde'] > 0) & 
                                   (df['jours_avant_prochain_solde'] <= 14)).astype(np.int8)
        df['in_pre_event_21d'] = ((df['jours_avant_prochain_solde'] > 0) & 
                                   (df['jours_avant_prochain_solde'] <= 21)).astype(np.int8)
        df['semaine_avant_solde_majeur'] = (
            (df['jours_avant_prochain_solde'] <= 7) & 
            (df['intensite_prochain_solde'] >= 3)
        ).astype(np.int8)
        
        logger.info(f"   Features événements créées")
        logger.info(f"    - Jours en événement: {df['est_en_solde'].sum():,}")
        
        return df
    
    def _date_in_period(self, date: pd.Timestamp, config: Dict) -> bool:
        """Vérifie si une date est dans une période"""
        month = date.month
        day = date.day
        
        if config['debut_mois'] == config['fin_mois']:
            return (month == config['debut_mois'] and 
                    config['debut_jour'] <= day <= config['fin_jour'])
        else:
            return ((month == config['debut_mois'] and day >= config['debut_jour']) or
                    (month == config['fin_mois'] and day <= config['fin_jour']))
    
    def _create_event_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features de distance aux événements"""
        
        # Liste des dates d'événements
        all_event_dates = []
        for annee in df['annee'].unique():
            for nom_solde, config in self.calendar_events.items():
                try:
                    debut = pd.Timestamp(f"{int(annee)}-{config['debut_mois']}-{config['debut_jour']}")
                    all_event_dates.append({
                        'date': debut,
                        'event': nom_solde,
                        'intensite': config['intensite']
                    })
                except:
                    continue
        
        event_dates_df = pd.DataFrame(all_event_dates).sort_values('date')
        
        # Calculer distances
        df['jours_avant_prochain_solde'] = 999
        df['intensite_prochain_solde'] = 0
        
        df_reset = df.reset_index()
        distances = df_reset['date'].apply(
            lambda x: self._calculate_days_to_next_event(x, event_dates_df)
        )
        
        df['jours_avant_prochain_solde'] = [d[0] for d in distances]
        df['intensite_prochain_solde'] = [d[1] for d in distances]
        
        # Distance pondérée
        df['distance_ponderee'] = df['jours_avant_prochain_solde'] / (df['intensite_prochain_solde'] + 1)
        
        return df
    
    def _calculate_days_to_next_event(self, date: pd.Timestamp, event_dates_df: pd.DataFrame) -> Tuple[int, int]:
        """Calcule les jours avant le prochain événement"""
        future_events = event_dates_df[event_dates_df['date'] > date]
        if len(future_events) == 0:
            return 999, 0
        
        next_event = future_events.iloc[0]
        days = (next_event['date'] - date).days
        return days, next_event['intensite']
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features de lag"""
        logger.info("\n[3/6] Features lag...")
        
        features_to_lag = ['vues_lr_log', 'clics_lr_log']
        
        # Lags principaux (décalés de 28 pour éviter data leakage)
        for lag_relatif in self.lag_windows[:4]:  # [1, 7, 14, 28]
            lag_absolu = lag_relatif + 28
            
            for feature in features_to_lag:
                col_name = f'{feature}_lag{lag_relatif}d'
                df[col_name] = df.groupby('search_id')[feature].shift(lag_absolu)
        
        # Lags long terme
        for lag_absolu in [56, 365]:
            lag_relatif = lag_absolu - 28
            df[f'vues_lr_log_lag{lag_relatif}d'] = df.groupby('search_id')['vues_lr_log'].shift(lag_absolu)
            df[f'taux_conversion_lag{lag_relatif}d'] = df.groupby('search_id')['taux_conversion'].shift(lag_absolu)
        
        logger.info(f"   Features lag créées")
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features de rolling (moyennes mobiles)"""
        logger.info("\n[4/6] Features rolling...")
        
        for window in self.rolling_windows:
            # Moyenne mobile
            df[f'vues_ma_{window}d'] = df.groupby('search_id')['vues_lr_log'].transform(
                lambda x: x.shift(28).rolling(window=window, min_periods=1).mean()
            )
            
            # Écart-type
            df[f'vues_std_{window}d'] = df.groupby('search_id')['vues_lr_log'].transform(
                lambda x: x.shift(28).rolling(window=window, min_periods=1).std()
            )
            
            # Min et Max pour fenêtres clés
            if window in [7, 28]:
                df[f'vues_min_{window}d'] = df.groupby('search_id')['vues_lr_log'].transform(
                    lambda x: x.shift(28).rolling(window=window, min_periods=1).min()
                )
                df[f'vues_max_{window}d'] = df.groupby('search_id')['vues_lr_log'].transform(
                    lambda x: x.shift(28).rolling(window=window, min_periods=1).max()
                )
        
        logger.info(f"   Features rolling créées")
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features de tendance"""
        logger.info("\n[5/6] Features tendance...")
        
        # Tendances linéaires
        df['trend_7d'] = df.groupby('search_id')['vues_lr_log'].transform(
            lambda x: (x.shift(28) - x.shift(35)) / 7
        ).fillna(0)
        
        df['trend_14d'] = df.groupby('search_id')['vues_lr_log'].transform(
            lambda x: (x.shift(28) - x.shift(42)) / 14
        ).fillna(0)
        
        df['trend_28d'] = df.groupby('search_id')['vues_lr_log'].transform(
            lambda x: (x.shift(28) - x.shift(56)) / 28
        ).fillna(0)
        
        # Accélération
        df['acceleration_7d'] = df.groupby('search_id')['trend_7d'].transform(
            lambda x: x.diff()
        ).fillna(0)
        
        # Momentum
        df['momentum_7_28'] = df['vues_lr_log_lag7d'] / (df['vues_lr_log_lag28d'] + 1e-6)
        df['momentum_ma_7_28'] = df['vues_ma_7d'] / (df['vues_ma_28d'] + 1e-6)
        df['momentum_ma_7_60'] = df['vues_ma_7d'] / (df['vues_ma_60d'] + 1e-6)
        
        # Écarts
        df['ecart_vs_ma28d'] = df.groupby('search_id')['vues_lr_log'].shift(28) - df['vues_ma_28d']
        df['ecart_vs_ma60d'] = df.groupby('search_id')['vues_lr_log'].shift(28) - df['vues_ma_60d']
        
        # Z-score
        df['zscore_28d'] = (
            (df.groupby('search_id')['vues_lr_log'].shift(28) - df['vues_ma_28d']) / 
            (df['vues_std_28d'] + 1e-6)
        )
        
        # Variabilité
        for window in [7, 28]:
            df[f'cv_{window}d'] = df[f'vues_std_{window}d'] / (df[f'vues_ma_{window}d'] + 1e-6)
        
        df['range_7d_norm'] = (df['vues_max_7d'] - df['vues_min_7d']) / (df['vues_ma_7d'] + 1e-6)
        df['range_28d_norm'] = (df['vues_max_28d'] - df['vues_min_28d']) / (df['vues_ma_28d'] + 1e-6)
        
        logger.info(f"   Features tendance créées")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features d'interaction"""
        logger.info("\n[6/6] Features interaction...")
        
        # Features agrégées par catégorie
        for feature in ['vues_lr_log', 'taux_conversion']:
            col_name = f'{feature}_ma_28d_par_cat1'
            df[col_name] = df.groupby('cat1')[feature].transform(
                lambda x: x.shift(28).rolling(window=28, min_periods=1).mean()
            )
        
        # Interactions catégorie × événement
        df['cat1_x_in_black_friday'] = (df['cat1'].astype(str) + '_BF_' + 
                                         df['in_black_friday'].astype(str)).astype('category')
        df['cat1_x_in_soldes_hiver'] = (df['cat1'].astype(str) + '_SH_' + 
                                         df['in_soldes_hiver'].astype(str)).astype('category')
        df['cat1_x_future_black_friday'] = (df['cat1'].astype(str) + '_FBF_' + 
                                             df['future_black_friday'].astype(str)).astype('category')
        df['cat1_x_future_soldes_hiver'] = (df['cat1'].astype(str) + '_FSH_' + 
                                             df['future_soldes_hiver'].astype(str)).astype('category')
        
        # Interaction catégorie × mois
        df['cat1_x_mois'] = (df['cat1'].astype(str) + '_M' + df['mois'].astype(str)).astype('category')
        
        # Interaction segment × événement
        df['segment_x_intensite_event'] = (df['segment_popularite'].astype(str) + '_I' + 
                                           df['intensite_solde_future'].astype(str)).astype('category')
        
        # Interactions numériques
        df['tail_x_intensite_future'] = df['tail_normalized'] * df['intensite_solde_future']
        df['tail_x_segment'] = df['tail_normalized'] * df['segment_popularite'].cat.codes
        
        logger.info(f"   Features interaction créées")
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage final des features"""
        logger.info("\nNettoyage des features...")
        
        # Identifier colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remplacer inf/-inf par 0
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # Remplir NaN dans les features lag/rolling
        lag_rolling_cols = [col for col in numeric_cols if any(x in col for x in 
            ['lag', 'ma_', 'std_', 'trend_', 'momentum_', 'ecart_', 'zscore_', 'cv_', 'range_'])]
        
        for col in lag_rolling_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df.groupby('search_id')[col].fillna(method='ffill')
                df[col] = df[col].fillna(0)
        
        logger.info(f"   Nettoyage terminé")
        
        return df