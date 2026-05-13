import requests
import pandas as pd
import time

LEAGUES = {
    'Premier League': 'epl',
    'La Liga': 'esp.1',
    'Bundesliga': 'ger.1',
    'Serie A': 'ita.1',
    'Ligue 1': 'fra.1'
}

def get_fixtures(league):
    url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get('events', [])

def parse_match(event):
    comp = event.get('competitions', [])[0]
    teams = comp.get('competitors', [])
    home = next(t for t in teams if t['homeAway'] == 'home')
    away = next(t for t in teams if t['homeAway'] == 'away')
    
    return {
        'league': event['shortName'],
        'date': event['date'],
        'home_team': home['team']['name'],
        'away_team': away['team']['name'],
        'home_score': int(home.get('score', 0)) if home.get('score') else 0,
        'away_score': int(away.get('score', 0)) if away.get('score') else 0,
        'status': event['status']['type']['name']
    }

def extract_all_leagues():
    all_matches = []
    for name, league_code in LEAGUES.items():
        print(f"Extrayendo datos de {name}")
        matches = get_fixtures(league_code)
        for event in matches:
            match = parse_match(event)
            if match['status'] == 'STATUS_FINAL':
                all_matches.append(match)
        time.sleep(1)  # Para evitar saturar la API
    df = pd.DataFrame(all_matches)
    df.to_csv('futbol_espn.csv', index=False)
    print('Datos guardados en futbol_espn.csv')

if __name__ == "__main__":
    extract_all_leagues()
