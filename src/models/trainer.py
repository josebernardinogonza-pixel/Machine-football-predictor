"""
Advanced model training with hyperparameter tuning and model selection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Professional model trainer with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'result') -> Tuple:
        """Prepare data for training"""
        X = df.drop(columns=[target_col, 'match_id', 'date', 'league', 
                            'home_team', 'away_team', 'home_score', 'away_score'], 
                    errors='ignore')
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train multiple models with hyperparameter tuning"""
        
        # Model configurations
        model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, use_label_encoder=False, 
                                          eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        trained_models = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}...")
            
            # Grid search
            grid_search = GridSearchCV(
                config['model'], 
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            trained_models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models"""
        results = {}
        
        for model_name, model_dict in self.models.items():
            model = model_dict['model']
            y_pred = model.predict(X_test)
            
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            results[model_name]['cv_mean'] = cv_scores.mean()
            results[model_name]['cv_std'] = cv_scores.std()
            
            logger.info(f"{model_name} - Accuracy: {results[model_name]['accuracy']:.4f}")
            logger.info(f"{model_name} - F1 Score: {results[model_name]['f1']:.4f}")
        
        self.results = results
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_model_name]['model']
        logger.info(f"Best model: {best_model_name}")
        
        return results
    
    def save_models(self, path: str = 'data/models/'):
        """Save trained models and results"""
        import json
        from pathlib import Path
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model
        if self.best_model:
            joblib.dump(self.best_model, f"{path}best_model_{timestamp}.pkl")
        
        # Save all models
        for model_name, model_dict in self.models.items():
            joblib.dump(model_dict['model'], f"{path}{model_name}_{timestamp}.pkl")
        
        # Save results
        with open(f"{path}results_{timestamp}.json", 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"Models saved to {path}")
