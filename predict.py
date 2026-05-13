import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('modelo_espn_rf.pkl')

# Ejemplo nuevo partido a predecir
# Rellena con datos reales o estimados (home_score y away_score aquí se usan solo como ejemplo)
nuevo_partido = {'home_score': 1, 'away_score': 2}

df_nuevo = pd.DataFrame([nuevo_partido])

prediccion = model.predict(df_nuevo)[0]

resultados_map = {0: 'Empate', 1: 'Gana local', 2: 'Gana visitante'}
print(f'Predicción para el nuevo partido: {resultados_map[prediccion]}')
