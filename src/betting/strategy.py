# Crear nuevo archivo: src/betting/strategy.py

class BettingStrategy:
    """Sistema de gestión de apuestas Kelly Criterion"""
    
    def __init__(self, initial_bankroll: float = 1000):
        self.bankroll = initial_bankroll
        self.bet_history = []
        self.min_confidence = 0.60  # Solo apostar con >60% confianza
        
    def calculate_kelly_stake(self, probability: float, odds: float) -> float:
        """Kelly Criterion: f = (bp - q) / b"""
        q = 1 - probability
        b = odds - 1
        kelly = (b * probability - q) / b
        
        # Usar Kelly fraccionario (25%) para reducir riesgo
        return max(0, kelly * 0.25)
    
    def should_bet(self, prediction: Dict, odds: Dict) -> Dict:
        """Decidir si apostar basado en value betting"""
        confidence = prediction['confidence']
        predicted_outcome = prediction['prediction']
        
        # Solo apostar si confianza > umbral
        if confidence < self.min_confidence:
            return {'bet': False, 'reason': 'Low confidence'}
        
        # Calcular valor esperado
        market_odds = odds.get(predicted_outcome, 0)
        implied_prob = 1 / market_odds if market_odds > 0 else 0
        
        # Value betting: solo apostar si hay valor
        if confidence > implied_prob * 1.05:  # 5% margen
            stake_fraction = self.calculate_kelly_stake(confidence, market_odds)
            stake_amount = self.bankroll * stake_fraction
            
            return {
                'bet': True,
                'outcome': predicted_outcome,
                'stake': stake_amount,
                'odds': market_odds,
                'expected_value': (confidence * market_odds) - 1
            }
        
        return {'bet': False, 'reason': 'No value'}
