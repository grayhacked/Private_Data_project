"""
Module d'inférence pour prédictions
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional

from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model import ModelTrainer

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Classe pour faire des prédictions avec le modèle entraîné"""
    
    def __init__(self, model_path: str, artifacts_path: str, config: Dict):
        """
        Args:
            model_path: Chemin vers le modèle
            artifacts_path: Chemin vers les artifacts
            config: Configuration (calendar_events, etc.)
        """
        self.config = config
        
        # Charger modèle
        self.trainer = ModelTrainer(
            model_params=config['model']['params'],
            early_stopping_rounds=config['model']['early_stopping_rounds']
        )
        self.trainer.load(model_path, artifacts_path)
        
        # Initialiser preprocessor et feature engineer
        self.preprocessor = DataPreprocessor(
            reduction_rate=config['data'].get('reduction_rate', 0.2)
        )
        
        self.feature_engineer = FeatureEngineer(
            calendar_events=config['features']['calendar_events'],
            lag_windows=config['features']['lag_windows'],
            rolling_windows=config['features']['rolling_windows']
        )
        
        logger.info(" Prédicteur initialisé")
    
    def predict(self, df: pd.DataFrame, return_original_scale: bool = True) -> pd.DataFrame:
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            df: DataFrame avec les mêmes colonnes que les données d'entraînement
            return_original_scale: Si True, retransorme en échelle originale
            
        Returns:
            DataFrame avec prédictions
        """
        logger.info("=" * 80)
        logger.info("PRÉDICTION")
        logger.info("=" * 80)
        logger.info(f"Données entrée: {len(df):,} lignes")
        
        # Preprocessing
        df = self._preprocess_for_prediction(df)
        
        # Feature engineering
        df = self.feature_engineer.create_all_features(df)
        
        # Préparer features
        X = self._prepare_features(df)
        
        # Filtrer features sélectionnées
        X = X[self.trainer.selected_features]
        
        # Prédictions
        logger.info(f"Prédiction sur {len(X):,} observations...")
        predictions_log = self.trainer.model.predict(X)
        
        # Créer résultats
        results = pd.DataFrame({
            'date': df.index,
            'search_id': df['search_id'] if 'search_id' in df.columns else None,
            'vues_predites_log': predictions_log
        })
        
        # Retransformer si demandé
        if return_original_scale:
            results['vues_predites'] = np.expm1(predictions_log)
            logger.info(f" Prédictions en échelle originale")
        
        logger.info(f" {len(results):,} prédictions générées")
        
        return results
    
    def _preprocess_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing pour prédiction"""
        
        # Conversion types
        df = self.preprocessor._convert_dtypes(df)
        
        # Features de base
        df = self.preprocessor.create_base_features(df)
        
        # Segments
        df = self.preprocessor.create_segments(df)
        
        # Préparer pour feature engineering
        df = self.preprocessor.prepare_for_training(df)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features (encoding, normalisation)"""
        
        X = df.copy()
        
        # Target encoding
        for col, encoding_info in self.trainer.encoding_maps.items():
            original_col = col.replace('_encoded', '')
            if original_col in X.columns:
                encoding_map = encoding_info['encoding_map']
                global_mean = encoding_info['global_mean']
                X[col] = X[original_col].map(encoding_map).fillna(global_mean)
        
        # Normalisation
        if self.trainer.scaler is not None:
            cols_to_normalize = [col for col in X.columns 
                               if col in self.trainer.scaler.feature_names_in_]
            
            if len(cols_to_normalize) > 0:
                X[cols_to_normalize] = self.trainer.scaler.transform(
                    X[cols_to_normalize].fillna(0)
                )
        
        return X
    
    def predict_from_file(self, filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Prédit à partir d'un fichier CSV
        
        Args:
            filepath: Chemin vers fichier CSV
            output_path: Chemin pour sauvegarder prédictions (optionnel)
            
        Returns:
            DataFrame avec prédictions
        """
        logger.info(f"Chargement des données depuis {filepath}...")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")
        
        # Prédictions
        results = self.predict(df)
        
        # Sauvegarder si demandé
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f" Prédictions sauvegardées: {output_path}")
        
        return results
    
    def predict_top_k(self, df: pd.DataFrame, k: int = 30, 
                     date_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Prédit et retourne le top K recherches
        
        Args:
            df: DataFrame avec données
            k: Nombre de recherches à retourner
            date_filter: Date pour filtrer (optionnel, format 'YYYY-MM-DD')
            
        Returns:
            DataFrame avec top K recherches
        """
        # Prédictions
        results = self.predict(df)
        
        # Filtrer par date si demandé
        if date_filter:
            results = results[results['date'] == date_filter]
        
        # Agréger par search_id
        top_searches = results.groupby('search_id').agg({
            'vues_predites': 'sum',
            'vues_predites_log': 'mean'
        }).reset_index()
        
        # Trier et prendre top K
        top_searches = top_searches.sort_values('vues_predites', ascending=False).head(k)
        
        logger.info(f" Top {k} recherches identifiées")
        
        return top_searches