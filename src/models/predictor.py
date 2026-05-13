"""
Production prediction service with confidence scoring
"""
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from src.data.features import FeatureEngineer

logger = logging.getLogger(__name__)

class MatchPredictor:
    """Professional prediction service with confidence intervals"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.model_path = model_path
        self.prediction_history = []
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str):
        """Load trained model"""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
    
    def predict_match(self, match_data: Dict) -> Dict:
        """Predict match outcome with confidence scoring"""
        # Prepare features
        df = pd.DataFrame([match_data])
        features = self.feature_engineer.create_features(df)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Calculate confidence
        confidence = max(probabilities)
        
        result_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
        
        prediction_result = {
            'prediction': result_map[prediction],
            'confidence': float(confidence),
            'probabilities': {
                'draw': float(probabilities[0]),
                'home_win': float(probabilities[1]) if len(probabilities) > 1 else 0,
                'away_win': float(probabilities[2]) if len(probabilities) > 2 else 0
            },
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_path
        }
        
        # Store prediction
        self.prediction_history.append(prediction_result)
        
        return prediction_result
    
    def predict_multiple(self, matches: List[Dict]) -> List[Dict]:
        """Predict multiple matches"""
        predictions = []
        for match in matches:
            prediction = self.predict_match(match)
            predictions.append(prediction)
        return predictions
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics"""
        if not self.prediction_history:
            return {'total_predictions': 0}
        
        predictions_df = pd.DataFrame(self.prediction_history)
        
        return {
            'total_predictions': len(self.pred
