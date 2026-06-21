"""
Configuración centralizada para Machine Football Predictor
"""
import os
from pathlib import Path

class Config:
    """Clase de configuración base"""
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = DATA_DIR / 'models'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    
    # API Keys
    FOOTBALL_DATA_API_KEY = os.getenv('FOOTBALL_DATA_API_KEY', '')
    ESPN_API_KEY = os.getenv('ESPN_API_KEY', '')
    
    # Model settings
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'pipeline.log'
