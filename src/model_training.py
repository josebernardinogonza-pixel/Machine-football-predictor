"""
Script de entrenamiento de modelos para el pipeline ML Fútbol
"""
from src.data import FootballDataExtractor, FeatureEngineer
from src.models import ModelTrainer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Ejecuta el entrenamiento del modelo"""
    logger.info("Iniciando entrenamiento del modelo...")
    
    try:
        # Cargar datos
        logger.info("Cargando datos de entrenamiento...")
        extractor = FootballDataExtractor()
        df = extractor.extract_historical_data('esp', seasons=5)
        
        # Ingeniería de características
        logger.info("Ingeniería de características...")
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_features(df)
        
        # Entrenar modelos
        logger.info("Entrenando modelos...")
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df_features)
        
        # Entrenar múltiples modelos
        trained_models = trainer.train_models(X_train, y_train)
        logger.info(f"Modelos entrenados: {list(trained_models.keys())}")
        
        # Evaluar modelos
        logger.info("Evaluando modelos...")
        results = trainer.evaluate_models(X_test, y_test)
        
        for model_name, metrics in results.items():
            logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Guardar modelos
        logger.info("Guardando modelos...")
        trainer.save_models('data/models/')
        
        logger.info("Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()
