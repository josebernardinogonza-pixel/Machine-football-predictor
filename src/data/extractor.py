"""
Professional data extractor from multiple football data sources
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
from config.settings import Config

logger = logging.getLogger(__name__)

class FootballDataExtractor:
    """Advanced data extractor with multiple sources and caching"""
    
    SOURCES = {
        'espn': 'https://site.api.espn.com/apis/site/v2/sports/soccer/',
        'football_data': 'https://api.football-data.org/v4/',
        'openligadb': 'https://api.openligadb.de/v1/'
    }
    
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.session.headers.update({
            'X-Auth-Token': self.config.FOOTBALL_DATA_API_KEY,
            'User-Agent': 'MachineFootballPredictor/1.0'
        })
        
    def extract_historical_data(self, league: str, seasons: int = 5) -> pd.DataFrame:
        """Extract historical data for multiple seasons"""
        all_matches = []
        
        for season in range(seasons):
            year = datetime.now().year - season
            matches = self._get_season_matches(league, year)
            all_matches.extend(matches)
            logger.info(f"Extracted {len(matches)} matches for {league} season {year}")
            time.sleep(1.5)  # Rate limiting
            
        df = pd.DataFrame(all_matches)
        self._save_raw_data(df, league)
        return df
    
    def _get_season_matches(self, league: str, year: int) -> List[Dict]:
        """Get all matches for a specific season"""
        # ESPN API implementation
        url = f"{self.SOURCES['espn']}{league}/scoreboard?dates={year}"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        
        matches = []
        for event in data.get('events', []):
            match = self._parse_match(event, league)
            if match:
                matches.append(match)
        return matches
    
    def _parse_match(self, event: Dict, league: str) -> Optional[Dict]:
        """Parse match data with advanced features"""
        try:
            comp = event.get('competitions', [])[0]
            teams = comp.get('competitors', [])
            home = next(t for t in teams if t['homeAway'] == 'home')
            away = next(t for t in teams if t['homeAway'] == 'away')
            
            return {
                'match_id': event.get('id'),
                'league': league,
                'season': event.get('season', {}).get('year'),
                'date': event.get('date'),
                'home_team': home['team']['name'],
                'away_team': away['team']['name'],
                'home_score': int(home.get('score', 0)) if home.get('score') else 0,
                'away_score': int(away.get('score', 0)) if away.get('score') else 0,
                'home_shots': int(home.get('statistics', {}).get('shots', 0)),
                'away_shots': int(away.get('statistics', {}).get('shots', 0)),
                'home_shots_on_target': int(home.get('statistics', {}).get('shotsOnGoal', 0)),
                'away_shots_on_target': int(away.get('statistics', {}).get('shotsOnGoal', 0)),
                'possession_home': float(home.get('statistics', {}).get('possession', 0)),
                'possession_away': float(away.get('statistics', {}).get('possession', 0)),
                'status': event['status']['type']['name']
            }
        except Exception as e:
            logger.error(f"Error parsing match: {e}")
            return None
    
    def _save_raw_data(self, df: pd.DataFrame, league: str):
        """Save raw data with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/raw/{league}_{timestamp}.parquet"
        df.to_parquet(filename)
        logger.info(f"Saved raw data to {filename}")
