"""
Module de modélisation et entraînement
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from pathlib import Path

from .constants import (
    COLUMNS_TO_EXCLUDE, 
    NATIVE_CATEGORICAL_FEATURES,
    FEATURES_TO_ENCODE
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Classe pour l'entraînement du modèle"""
    
    def __init__(self, model_params: Dict, early_stopping_rounds: int = 50,
                 feature_selection_threshold: float = 0.98):
        """
        Args:
            model_params: Paramètres LightGBM
            early_stopping_rounds: Rounds pour early stopping
            feature_selection_threshold: Seuil pour sélection features
        """
        self.model_params = model_params
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_selection_threshold = feature_selection_threshold
        self.model = None
        self.scaler = None
        self.encoding_maps = {}
        self.selected_features = None
        self.categorical_features = None
        
    def prepare_data(self, df: pd.DataFrame, 
                    test_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                              pd.Series, pd.Series]:
        """Prépare les données pour l'entraînement"""
        logger.info("=" * 80)
        logger.info("PRÉPARATION DES DONNÉES")
        logger.info("=" * 80)
        
        # Définir X et Y
        Y = df['vues_future_28d_log'].copy()
        
        # Exclure colonnes
        cols_to_drop = [col for col in COLUMNS_TO_EXCLUDE if col in df.columns]
        X = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Nettoyer observations invalides
        mask_valide = ~Y.isnull()
        X = X[mask_valide]
        Y = Y[mask_valide]
        
        # Split temporel
        dates = X.index
        train_mask = dates < test_date
        test_mask = dates >= test_date
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = Y[train_mask]
        y_test = Y[test_mask]
        
        logger.info(f"Split effectué:")
        logger.info(f"  TRAIN: {len(X_train):,} obs ({dates[train_mask].min()} → {dates[train_mask].max()})")
        logger.info(f"  TEST:  {len(X_test):,} obs ({dates[test_mask].min()} → {dates[test_mask].max()})")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_target_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, smoothing: int = 10,
                              min_samples: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applique target encoding"""
        logger.info("\nTarget encoding...")
        
        # Identifier colonnes à encoder
        interaction_cols = [col for col in X_train.columns if '_x_' in col or 'interaction' in col]
        cols_to_encode = [col for col in FEATURES_TO_ENCODE if col in X_train.columns] + interaction_cols
        
        logger.info(f"  Colonnes à encoder: {len(cols_to_encode)}")
        
        encoded_cols = []
        for col in cols_to_encode:
            if col in X_train.columns:
                try:
                    X_train, X_test, encoded_name, encoding_map, global_mean = \
                        self._target_encode_column(X_train, X_test, y_train, col, smoothing, min_samples)
                    
                    self.encoding_maps[encoded_name] = {
                        'encoding_map': encoding_map,
                        'global_mean': global_mean
                    }
                    encoded_cols.append(encoded_name)
                except Exception as e:
                    logger.warning(f"  Erreur encoding {col}: {str(e)[:50]}")
        
        logger.info(f"   {len(encoded_cols)} colonnes encodées")
        
        return X_train, X_test
    
    def _target_encode_column(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, col: str, smoothing: int,
                              min_samples: int) -> Tuple:
        """Encode une colonne avec target encoding"""
        
        # Créer DataFrame temporaire
        temp_df = pd.DataFrame({'cat': X_train[col], 'target': y_train})
        
        # Calculer moyennes
        global_mean = y_train.mean()
        cat_stats = temp_df.groupby('cat')['target'].agg(['mean', 'count'])
        
        # Appliquer smoothing
        cat_stats['smoothed_mean'] = (
            (cat_stats['mean'] * cat_stats['count'] + global_mean * smoothing) /
            (cat_stats['count'] + smoothing)
        )
        
        cat_stats.loc[cat_stats['count'] < min_samples, 'smoothed_mean'] = global_mean
        
        encoding_map = cat_stats['smoothed_mean'].to_dict()
        
        # Appliquer
        encoded_name = f'{col}_encoded'
        X_train[encoded_name] = X_train[col].map(encoding_map).fillna(global_mean)
        X_test[encoded_name] = X_test[col].map(encoding_map).fillna(global_mean)
        
        return X_train, X_test, encoded_name, encoding_map, global_mean
    
    def apply_normalization(self, X_train: pd.DataFrame, 
                           X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applique normalisation avec RobustScaler"""
        logger.info("\nNormalisation...")
        
        # Identifier colonnes à normaliser
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        cols_to_normalize = []
        for col in numeric_cols:
            if any(exclude in col for exclude in [
                'is_', 'has_', 'in_', 'future_', '_encoded', 'intensite'
            ]):
                continue
            
            if X_train[col].nunique() > 10:
                cols_to_normalize.append(col)
        
        logger.info(f"  Colonnes à normaliser: {len(cols_to_normalize)}")
        
        if len(cols_to_normalize) > 0:
            self.scaler = RobustScaler()
            
            X_train_scaled = self.scaler.fit_transform(X_train[cols_to_normalize].fillna(0))
            X_test_scaled = self.scaler.transform(X_test[cols_to_normalize].fillna(0))
            
            X_train[cols_to_normalize] = X_train_scaled
            X_test[cols_to_normalize] = X_test_scaled
            
            logger.info(f"   Normalisation appliquée")
        
        return X_train, X_test
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                       validation_size: float = 0.15) -> List[str]:
        """Sélection des features importantes"""
        logger.info("\nSélection des features...")
        
        # Split interne
        split_idx = int(len(X_train) * (1 - validation_size))
        X_train_sel = X_train.iloc[:split_idx]
        X_val_sel = X_train.iloc[split_idx:]
        y_train_sel = y_train.iloc[:split_idx]
        y_val_sel = y_train.iloc[split_idx:]
        
        # Identifier catégories
        categorical_features = [col for col in NATIVE_CATEGORICAL_FEATURES 
                               if col in X_train.columns]
        
        # Modèle léger
        lgbm_selector = lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Entraînement
        lgbm_selector.fit(
            X_train_sel, y_train_sel,
            eval_set=[(X_val_sel, y_val_sel)],
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        # Importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': lgbm_selector.feature_importances_,
            'importance_pct': lgbm_selector.feature_importances_ / lgbm_selector.feature_importances_.sum() * 100
        }).sort_values('importance', ascending=False)
        
        # Sélectionner features
        cumsum_importance = feature_importance_df['importance_pct'].cumsum()
        n_features = (cumsum_importance <= self.feature_selection_threshold * 100).sum()
        n_features = max(80, min(n_features, 150))
        
        selected_features = feature_importance_df.head(n_features)['feature'].tolist()
        
        logger.info(f"   {len(selected_features)} features sélectionnées")
        logger.info(f"\n  Top 10 features:")
        for i, row in feature_importance_df.head(10).iterrows():
            logger.info(f"    {i+1}. {row['feature']}: {row['importance_pct']:.2f}%")
        
        self.selected_features = selected_features
        self.categorical_features = [f for f in categorical_features if f in selected_features]
        
        return selected_features
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             validation_size: float = 0.15) -> None:
        """Entraîne le modèle"""
        logger.info("\n" + "=" * 80)
        logger.info("ENTRAÎNEMENT DU MODÈLE")
        logger.info("=" * 80)
        
        # Filtrer features
        X_train = X_train[self.selected_features]
        
        # Split validation
        split_idx = int(len(X_train) * (1 - validation_size))
        X_train_core = X_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_train_core = y_train.iloc[:split_idx]
        y_val = y_train.iloc[split_idx:]
        
        logger.info(f"Données d'entraînement:")
        logger.info(f"  Train: {len(X_train_core):,}")
        logger.info(f"  Validation: {len(X_val):,}")
        logger.info(f"  Features: {len(self.selected_features)}")
        logger.info(f"  Catégorielles: {len(self.categorical_features)}")
        
        # Créer modèle
        self.model = lgb.LGBMRegressor(**self.model_params)
        
        # Entraînement
        logger.info("\nDémarrage entraînement...")
        self.model.fit(
            X_train_core, y_train_core,
            eval_set=[
                (X_train_core, y_train_core),
                (X_val, y_val)
            ],
            eval_names=['train', 'valid'],
            categorical_feature=self.categorical_features,
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=200)
            ]
        )
        
        logger.info(f"\n Entraînement terminé:")
        logger.info(f"  Best iteration: {self.model.best_iteration_}")
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Évalue le modèle"""
        logger.info("\n" + "=" * 80)
        logger.info("ÉVALUATION")
        logger.info("=" * 80)
        
        # Filtrer features
        X_test = X_test[self.selected_features]
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        
        # Métriques
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        wape = self._calculate_wape(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'wape': wape
        }
        
        logger.info(f"\nMÉTRIQUES TEST SET:")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  WAPE: {wape:.2f}%")
        
        # Interprétation
        if r2 > 0.80:
            logger.info(f"   EXCELLENT: R² > 0.80")
        elif r2 > 0.70:
            logger.info(f"   BON: R² > 0.70")
        elif r2 > 0.60:
            logger.info(f"   MOYEN: R² > 0.60")
        else:
            logger.info(f"   FAIBLE: R² < 0.60")
        
        return metrics
    
    def _calculate_wape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calcule le WAPE"""
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    def save(self, model_path: str, artifacts_path: str) -> None:
        """Sauvegarde le modèle et les artifacts"""
        logger.info("\nSauvegarde du modèle et artifacts...")
        
        # Créer dossiers
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(artifacts_path).mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder modèle
        joblib.dump(self.model, model_path)
        logger.info(f"   Modèle: {model_path}")
        
        # Sauvegarder artifacts
        artifacts = {
            'scaler': self.scaler,
            'encoding_maps': self.encoding_maps,
            'selected_features': self.selected_features,
            'categorical_features': self.categorical_features,
            'model_params': self.model_params
        }
        
        artifacts_file = Path(artifacts_path) / 'artifacts.pkl'
        joblib.dump(artifacts, artifacts_file)
        logger.info(f"   Artifacts: {artifacts_file}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_file = Path(artifacts_path) / 'feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False)
        logger.info(f"   Feature importance: {importance_file}")
    
    def load(self, model_path: str, artifacts_path: str) -> None:
        """Charge le modèle et les artifacts"""
        logger.info("Chargement du modèle et artifacts...")
        
        # Charger modèle
        self.model = joblib.load(model_path)
        logger.info(f"   Modèle chargé")
        
        # Charger artifacts
        artifacts_file = Path(artifacts_path) / 'artifacts.pkl'
        artifacts = joblib.load(artifacts_file)
        
        self.scaler = artifacts['scaler']
        self.encoding_maps = artifacts['encoding_maps']
        self.selected_features = artifacts['selected_features']
        self.categorical_features = artifacts['categorical_features']
        self.model_params = artifacts['model_params']
        
        logger.info(f"  Artifacts chargés")