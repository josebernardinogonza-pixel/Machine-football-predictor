"""
Ensemble learning: combina múltiples modelos para mayor precisión.
"""
import numpy as np
import joblib
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        self.models = []
        for path in model_paths:
            self.models.append(joblib.load(path))
        self.weights = weights if weights else [1/len(model_paths)] * len(model_paths)
        assert len(self.models) == len(self.weights), "Weights must match models"
    
    def predict_proba(self, X):
        # Weighted average of probabilities
        avg_proba = np.zeros((X.shape[0], 3))
        for model, w in zip(self.models, self.weights):
            avg_proba += w * model.predict_proba(X)
        return avg_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
