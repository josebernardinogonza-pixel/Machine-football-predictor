"""
Script de extracción de datos para el pipeline ML Fútbol
"""
from src.data import FootballDataExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Ejecuta la extracción de datos"""
    logger.info("Iniciando extracción de datos...")
    
    extractor = FootballDataExtractor()
    
    # Extraer datos de las principales ligas
    leagues = ['esp', 'eng', 'ita', 'fra', 'ger']
    
    for league in leagues:
        try:
            logger.info(f"Extrayendo datos para la liga: {league}")
            df = extractor.extract_historical_data(league, seasons=5)
            logger.info(f"Datos extraídos exitosamente para {league}: {len(df)} partidos")
        except Exception as e:
            logger.error(f"Error extrayendo datos para {league}: {e}")
    
    logger.info("Extracción de datos completada")

if __name__ == "__main__":
    main()
