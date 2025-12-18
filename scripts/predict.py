#!/usr/bin/env python
"""
Script de prédiction avec le modèle LightGBM
"""
import sys
from pathlib import Path
import argparse
import logging

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, print_banner
from src.inference import ModelPredictor

logger = logging.getLogger(__name__)


def main(input_file: str, 
         output_file: str = None,
         config_path: str = "config/config.yaml",
         top_k: int = None,
         date_filter: str = None):
    """
    Pipeline de prédiction
    
    Args:
        input_file: Chemin vers fichier CSV/parquet d'entrée
        output_file: Chemin vers fichier de sortie (optionnel)
        config_path: Chemin vers fichier config
        top_k: Si spécifié, retourne top K recherches
        date_filter: Date pour filtrer (format 'YYYY-MM-DD')
    """
    
    # Charger config
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(
        level=config['logging']['level'],
        log_format=config['logging']['format']
    )
    
    print_banner(" PRÉDICTION MODÈLE LIGHTGBM", char="=")
    
    # ========================================================================
    # 1. INITIALISATION PRÉDICTEUR
    # ========================================================================
    logger.info("Initialisation du prédicteur...")
    
    predictor = ModelPredictor(
        model_path=config['output']['model_path'],
        artifacts_path=config['output']['artifacts_path'],
        config=config
    )
    
    # ========================================================================
    # 2. PRÉDICTION
    # ========================================================================
    print_banner("PRÉDICTION EN COURS...", char="-")
    
    if top_k:
        # Prédire top K
        logger.info(f"Prédiction top {top_k} recherches...")
        
        import pandas as pd
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_parquet(input_file)
        
        results = predictor.predict_top_k(
            df, 
            k=top_k,
            date_filter=date_filter
        )
        
        logger.info(f"\n TOP {top_k} RECHERCHES:")
        logger.info(f"\n{results.to_string(index=False)}")
        
    else:
        # Prédiction standard
        results = predictor.predict_from_file(
            input_file,
            output_path=output_file
        )
        
        logger.info(f"\n APERÇU DES PRÉDICTIONS:")
        logger.info(f"\n{results.head(10).to_string(index=False)}")
    
    # ========================================================================
    # 3. SAUVEGARDE
    # ========================================================================
    if output_file:
        print_banner(" SAUVEGARDE", char="-")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_csv(output_file, index=False)
        logger.info(f" Prédictions sauvegardées: {output_file}")
        
        # Statistiques
        if 'vues_predites' in results.columns:
            logger.info(f"\n STATISTIQUES:")
            logger.info(f"  • Nombre prédictions: {len(results):,}")
            logger.info(f"  • Vues moyennes: {results['vues_predites'].mean():.1f}")
            logger.info(f"  • Vues médianes: {results['vues_predites'].median():.1f}")
            logger.info(f"  • Vues min: {results['vues_predites'].min():.1f}")
            logger.info(f"  • Vues max: {results['vues_predites'].max():.1f}")
    
    # ========================================================================
    # FIN
    # ========================================================================
    print_banner(" PRÉDICTION TERMINÉE AVEC SUCCÈS !", char="=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction avec modèle LightGBM")
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Chemin vers fichier CSV d'entrée"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Chemin vers fichier de sortie (optionnel)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Chemin vers fichier config"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=None,
        help="Retourner top K recherches"
    )
    
    parser.add_argument(
        "-d", "--date",
        type=str,
        default=None,
        help="Filtrer par date (format YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    # Générer nom de fichier de sortie par défaut si non spécifié
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_predictions.csv")
    
    main(
        input_file=args.input_file,
        output_file=args.output,
        config_path=args.config,
        top_k=args.top_k,
        date_filter=args.date
    )