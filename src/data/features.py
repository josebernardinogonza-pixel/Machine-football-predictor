"""
Advanced feature engineering for football predictions
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Professional feature engineering with rolling statistics and form indicators"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df = df.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['league', 'date'])
        
        # Create team-level features
        df = self._create_team_features(df)
        
        # Create match-level features
        df = self._create_match_features(df)
        
        # Create rolling statistics
        df = self._create_rolling_statistics(df)
        
        # Create opponent strength features
        df = self._create_opponent_strength(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Drop non-feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['match_id', 'date', 'league', 'home_team', 'away_team', 
                        'home_score', 'away_score', 'status', 'result']]
        
        return df[feature_cols]
    
    def _create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team-specific features"""
        # Home advantage features
        df['is_home_favorite'] = self._calculate_favorite_status(df)
        
        # Team form (last 5 matches)
        for team_type in ['home', 'away']:
            df[f'{team_type}_form'] = self._calculate_form(df, team_type)
            df[f'{team_type}_goals_scored_avg_last5'] = self._rolling_avg(
                df, f'{team_type}_score', 5
            )
            df[f'{team_type}_goals_conceded_avg_last5'] = self._rolling_avg(
                df, f'{team_type}_score', 5, reversed=True
            )
            
        return df
    
    def _create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create match-specific features"""
        # Total goals
        df['total_goals'] = df['home_score'] + df['away_score']
        
        # Goal difference
        df['goal_difference'] = df['home_score'] - df['away_score']
        
        # Create result label
        df['result'] = np.where(df['home_score'] > df['away_score'], 1,
                               np.where(df['home_score'] < df['away_score'], 2, 0))
        
        # Shots efficiency
        if 'home_shots' in df.columns:
            df['home_shots_efficiency'] = df['home_score'] / (df['home_shots'] + 1)
            df['away_shots_efficiency'] = df['away_score'] / (df['away_shots'] + 1)
            
        return df
    
    def _create_rolling_statistics(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Create rolling statistics for teams"""
        for team_type in ['home', 'away']:
            # Points rolling average
            df[f'{team_type}_points_rolling_{window}'] = df.groupby(f'{team_type}_team')[
                f'{team_type}_score'
            ].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Goals variance
            df[f'{team_type}_goals_variance_{window}'] = df.groupby(f'{team_type}_team')[
                f'{team_type}_score'
            ].transform(
                lambda x: x.rolling(window, min_periods=1).var()
            )
            
        return df
    
    def _create_opponent_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent strength features"""
        # League position proxy (using rolling average)
        df['home_opponent_strength'] = df.groupby('home_team')['away_score'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['away_opponent_strength'] = df.groupby('away_team')['home_score'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # Team strength difference
        df['strength_difference'] = df['home_points_rolling_5'] - df['away_points_rolling_5']
        
        # Form difference
        df['form_difference'] = df['home_form'] - df['away_form']
        
        return df
    
    @staticmethod
    def _calculate_favorite_status(df: pd.DataFrame) -> int:
        """Calculate if home team is favorite based on league position"""
        # Placeholder - in production use actual league tables
        return 0
    
    @staticmethod
    def _calculate_form(df: pd.DataFrame, team_type: str) -> pd.Series:
        """Calculate team form based on last 5 matches"""
        # Points: win=3, draw=1, loss=0
        def get_points(x):
            if x > 0:
                return 3
            elif x < 0:
                return 0
            return 1
        
        if team_type == 'home':
            diff = df['goal_difference']
        else:
            diff = -df['goal_difference']
        
        return diff.rolling(5, min_periods=1).apply(
            lambda x: sum(get_points(d) for d in x)
        )
    
    @staticmethod
    def _rolling_avg(df: pd.Series, window: int, reversed: bool = False) -> pd.Series:
        """Calculate rolling average"""
        if reversed:
            # For conceded goals
            return df.groupby(level=0).transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
        return df.rolling(window, min_periods=1).mean()
