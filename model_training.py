import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Cargar datos
df = pd.read_csv('futbol_espn.csv')

# Crear columna resultado
def resultado(row):
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] < row['away_score']:
        return 2
    else:
        return 0

df['resultado'] = df.apply(resultado, axis=1)

# Aquí solo usaremos goles como características de ejemplo
X = df[['home_score', 'away_score']]
y = df['resultado']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Guardar modelo entrenado
joblib.dump(model, 'modelo_espn_rf.pkl')
