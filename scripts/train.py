#!/usr/bin/env python
"""
Script d'entraînement du modèle LightGBM
"""
import sys
from pathlib import Path
import argparse
import logging

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, ensure_directories, print_banner
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer

logger = logging.getLogger(__name__)


def main(config_path: str = "config/config.yaml", 
         reduce_memory: bool = True,
         save_processed: bool = True):
    """
    Pipeline d'entraînement complète
    
    Args:
        config_path: Chemin vers fichier config
        reduce_memory: Si True, applique réduction mémoire
        save_processed: Si True, sauvegarde données preprocessées
    """
    
    # Charger config
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(
        level=config['logging']['level'],
        log_format=config['logging']['format']
    )
    
    # Créer répertoires
    ensure_directories(config)
    
    print_banner("🚀 ENTRAÎNEMENT MODÈLE LIGHTGBM", char="=")
    
    # ========================================================================
    # 1. PREPROCESSING
    # ========================================================================
    print_banner("1️⃣ PREPROCESSING", char="-")
    
    preprocessor = DataPreprocessor(
        reduction_rate=config['data']['reduction_rate']
    )
    
    # Charger données
    df = preprocessor.load_data(config['data']['raw_path'])
    
    # Réduction mémoire
    if reduce_memory:
        df = preprocessor.reduce_memory(df)
        
        if save_processed:
            processed_path = Path(config['data']['processed_path']) / 'df_reduced.csv'
            df.to_csv(processed_path, index=False)
            logger.info(f"✅ Données réduites sauvegardées: {processed_path}")
    
    # Créer cible
    df = preprocessor.create_target(df, horizon=28)
    
    # Features de base
    df = preprocessor.create_base_features(df)
    
    # Segments
    df = preprocessor.create_segments(df)
    
    # Préparer pour feature engineering
    df = preprocessor.prepare_for_training(df)
    
    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    print_banner("2️⃣ FEATURE ENGINEERING", char="-")
    
    feature_engineer = FeatureEngineer(
        calendar_events=config['features']['calendar_events'],
        lag_windows=config['features']['lag_windows'],
        rolling_windows=config['features']['rolling_windows']
    )
    
    df = feature_engineer.create_all_features(df)
    
    if save_processed:
        processed_path = Path(config['data']['processed_path']) / 'df_with_features.parquet'
        df.to_parquet(processed_path, index=True)
        logger.info(f"✅ Données avec features sauvegardées: {processed_path}")
    
    # ========================================================================
    # 3. PRÉPARATION DONNÉES
    # ========================================================================
    print_banner("3️⃣ PRÉPARATION TRAIN/TEST", char="-")
    
    trainer = ModelTrainer(
        model_params=config['model']['params'],
        early_stopping_rounds=config['model']['early_stopping_rounds'],
        feature_selection_threshold=config['model']['feature_selection_threshold']
    )
    
    # Split train/test
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, 
        test_date=config['split']['test_date']
    )
    
    # Target encoding
    X_train, X_test = trainer.apply_target_encoding(
        X_train, X_test, y_train,
        smoothing=config['encoding']['smoothing'],
        min_samples=config['encoding']['min_samples']
    )
    
    # Normalisation
    X_train, X_test = trainer.apply_normalization(X_train, X_test)
    
    # Sélection features
    selected_features = trainer.select_features(
        X_train, y_train,
        validation_size=config['split']['validation_size']
    )
    
    # ========================================================================
    # 4. ENTRAÎNEMENT
    # ========================================================================
    print_banner("4️⃣ ENTRAÎNEMENT", char="-")
    
    trainer.train(
        X_train, y_train,
        validation_size=config['split']['validation_size']
    )
    
    # ========================================================================
    # 5. ÉVALUATION
    # ========================================================================
    print_banner("5️⃣ ÉVALUATION", char="-")
    
    metrics = trainer.evaluate(X_test, y_test)
    
    # ========================================================================
    # 6. SAUVEGARDE
    # ========================================================================
    print_banner("6️⃣ SAUVEGARDE", char="-")
    
    trainer.save(
        model_path=config['output']['model_path'],
        artifacts_path=config['output']['artifacts_path']
    )
    
    # Sauvegarder métriques
    import json
    metrics_path = Path(config['output']['artifacts_path']) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Métriques sauvegardées: {metrics_path}")
    
    # ========================================================================
    # FIN
    # ========================================================================
    print_banner("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !", char="=")
    
    logger.info(f"\n📊 RÉSUMÉ:")
    logger.info(f"  • Modèle: {config['output']['model_path']}")
    logger.info(f"  • Artifacts: {config['output']['artifacts_path']}")
    logger.info(f"  • Features: {len(selected_features)}")
    logger.info(f"  • R²: {metrics['r2']:.4f}")
    logger.info(f"  • MAE: {metrics['mae']:.4f}")
    logger.info(f"  • WAPE: {metrics['wape']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement modèle LightGBM")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Chemin vers fichier config"
    )
    parser.add_argument(
        "--no-reduce-memory",
        action="store_true",
        help="Désactiver réduction mémoire"
    )
    parser.add_argument(
        "--no-save-processed",
        action="store_true",
        help="Ne pas sauvegarder données preprocessées"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        reduce_memory=not args.no_reduce_memory,
        save_processed=not args.no_save_processed
    )