"""
Utilitaires et fonctions helpers
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(level: str = "INFO", log_format: str = None) -> None:
    """Configure le logging"""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Charge la configuration depuis YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Sauvegarde la configuration"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_directories(config: Dict[str, Any]) -> None:
    """Crée les répertoires nécessaires"""
    directories = [
        Path(config['data']['processed_path']),
        Path(config['output']['model_path']).parent,
        Path(config['output']['artifacts_path']),
        Path(config['output']['predictions_path']),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def print_banner(text: str, char: str = "=", width: int = 80) -> None:
    """Affiche une bannière"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")