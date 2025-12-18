"""
Module de prétraitement des données
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe pour le prétraitement des données"""
    
    def __init__(self, reduction_rate: float = 0.2):
        """
        Args:
            reduction_rate: Taux d'échantillonnage pour lignes sans VA
        """
        self.reduction_rate = reduction_rate
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Charge et prépare les données brutes"""
        logger.info(f"Chargement du fichier {filepath}...")
        df = pd.read_csv(filepath)
        logger.info(f"Données chargées: {len(df):,} lignes, {len(df.columns)} colonnes")
        
        # Conversion des types
        df = self._convert_dtypes(df)
        
        return df
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertit les types de colonnes de manière optimale"""
        logger.info("Conversion des types de données...")
        
        # Date
        df['date'] = pd.to_datetime(df['date'])
        
        # Catégories
        categorical_cols = ['search_id', 'cat1', 'cat2', 'cat3']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Numériques en float32 pour économiser mémoire
        numeric_cols = df.select_dtypes(include=['float64']).columns
        df[numeric_cols] = df[numeric_cols].astype('float32')
        
        return df
    
    def reduce_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Réduit la mémoire en échantillonnant intelligemment"""
        logger.info("=" * 80)
        logger.info("RÉDUCTION MÉMOIRE - Échantillonnage intelligent")
        logger.info("=" * 80)
        
        initial_size = len(df)
        initial_memory = df.memory_usage(deep=True).sum() / 1e9
        
        logger.info(f"État initial:")
        logger.info(f"  - Lignes: {initial_size:,}")
        logger.info(f"  - Mémoire: {initial_memory:.2f} GB")
        logger.info(f"  - Lignes sans VA: {(df['va_lr'] == 0).sum():,} ({(df['va_lr'] == 0).mean()*100:.1f}%)")
        
        # Séparer données avec/sans VA
        df_with_va = df[df['va_lr'] > 0].copy()
        df_no_va = df[df['va_lr'] == 0].copy()
        
        logger.info(f"Lignes AVEC VA conservées: {len(df_with_va):,}")
        
        # Échantillonner intelligemment les lignes sans VA
        logger.info(f"Échantillonnage des lignes SANS VA (taux: {self.reduction_rate})...")
        df_no_va_sampled = self._smart_sample(df_no_va, self.reduction_rate)
        
        logger.info(f"Lignes SANS VA après échantillonnage: {len(df_no_va_sampled):,}")
        
        # Combiner
        df_reduced = pd.concat([df_with_va, df_no_va_sampled], ignore_index=True)
        df_reduced = df_reduced.sort_values(['search_id', 'date'])
        
        final_memory = df_reduced.memory_usage(deep=True).sum() / 1e9
        reduction_pct = (1 - len(df_reduced)/initial_size) * 100
        
        logger.info(f"Dataset final réduit:")
        logger.info(f"  - Lignes: {len(df_reduced):,} (réduction: {reduction_pct:.1f}%)")
        logger.info(f"  - Mémoire: {final_memory:.2f} GB")
        logger.info(f"  - Ratio avec VA: {(df_reduced['va_lr'] > 0).mean()*100:.1f}%")
        
        return df_reduced
    
    def _smart_sample(self, df: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
        """Échantillonne intelligemment par search_id"""
        
        def sample_group(group):
            # Garder toutes les lignes en événement si colonne existe
            if 'est_en_solde' in group.columns:
                in_event = group['est_en_solde'] == 1
                event_rows = group[in_event]
                normal_rows = group[~in_event]
            else:
                event_rows = pd.DataFrame()
                normal_rows = group
            
            # Échantillonner les lignes normales
            if len(normal_rows) > 0:
                n_sample = max(1, int(len(normal_rows) * sample_rate))
                sampled_normal = normal_rows.sample(n=n_sample, random_state=42)
            else:
                sampled_normal = pd.DataFrame()
            
            return pd.concat([event_rows, sampled_normal])
        
        return df.groupby('search_id', group_keys=False).apply(sample_group)
    
    def create_target(self, df: pd.DataFrame, horizon: int = 28) -> pd.DataFrame:
        """Crée la variable cible (vues futures)"""
        logger.info(f"Création de la cible (horizon: {horizon} jours)...")
        
        # Créer cible
        df['vues_future_28d'] = df.groupby('search_id')['vues_lr'].shift(-horizon)
        
        # Supprimer lignes sans cible
        initial_len = len(df)
        df = df[df['vues_future_28d'].notna()].copy()
        logger.info(f"Lignes supprimées (horizon): {initial_len - len(df):,}")
        
        # Transformation log
        df['vues_future_28d_log'] = np.log1p(df['vues_future_28d'])
        
        logger.info(f"Cible créée: vues_future_28d_log")
        logger.info(f"  - Min: {df['vues_future_28d_log'].min():.3f}")
        logger.info(f"  - Max: {df['vues_future_28d_log'].max():.3f}")
        logger.info(f"  - Moyenne: {df['vues_future_28d_log'].mean():.3f}")
        
        return df
    
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features de base (transformations log, ratios)"""
        logger.info("Création des features de base...")
        
        # Transformations log
        df['vues_lr_log'] = np.log1p(df['vues_lr'])
        df['clics_lr_log'] = np.log1p(df['clics_lr'])
        df['va_lr_log'] = np.log1p(df['va_lr'])
        df['commandes_lr_log'] = np.log1p(df['commandes_lr'])
        
        # Ratios métier
        df['taux_conversion'] = df['commandes_lr'] / df['clics_lr'].replace(0, 1)
        df['panier_moyen'] = df['va_lr'] / df['commandes_lr'].replace(0, 1)
        
        # Indicateurs binaires
        df['has_va'] = (df['va_lr'] > 0).astype(np.int8)
        df['has_commandes'] = (df['commandes_lr'] > 0).astype(np.int8)
        df['has_clics'] = (df['clics_lr'] > 0).astype(np.int8)
        df['has_vues'] = (df['vues_lr'] > 0).astype(np.int8)
        df['is_active_commerce'] = ((df['has_clics'] == 1) | (df['has_va'] == 1)).astype(np.int8)
        
        # Nettoyage
        df = df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Features de base créées")
        
        return df
    
    def create_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les segments de popularité"""
        logger.info("Création des segments de popularité...")
        
        # Vues totales par search_id
        vues_totales = df.groupby('search_id')['vues_lr'].sum()
        df['vues_totales_historique'] = df['search_id'].map(vues_totales)
        
        # Quantiles par date
        df['quantile_90'] = df.groupby('date')['vues_totales_historique'].transform(lambda x: x.quantile(0.90))
        df['quantile_50'] = df.groupby('date')['vues_totales_historique'].transform(lambda x: x.quantile(0.50))
        
        # Segmentation
        conditions = [
            df['vues_totales_historique'] >= df['quantile_90'],
            df['vues_totales_historique'] >= df['quantile_50']
        ]
        choices = ['HEAD', 'MID']
        df['segment_popularite'] = np.select(conditions, choices, default='TAIL')
        df['segment_popularite'] = df['segment_popularite'].astype('category')
        
        # Segment tail
        df['tail_segment'] = pd.cut(
            df['tail'], 
            bins=[0, 33, 45, 100], 
            labels=['haute_performance', 'moyenne_performance', 'faible_performance']
        )
        df['tail_normalized'] = df['tail'] / 50
        
        # Cleanup
        df = df.drop(['quantile_90', 'quantile_50'], axis=1, errors='ignore')
        
        logger.info(f"Segments créés:")
        logger.info(f"\n{df['segment_popularite'].value_counts()}")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Préparation finale avant feature engineering"""
        
        # Trier par date
        df = df.sort_values(by=['search_id', 'date'])
        
        # S'assurer que date est l'index
        if 'date' not in df.index.names:
            df = df.set_index('date').sort_index()
        
        return df