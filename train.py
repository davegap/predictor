import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def entrenar():
    # Leer el dataset
    dataset = pd.read_csv('dataset.csv')

    # Entradas (X) y salida (y)
    X = dataset.drop('ventas', axis=1).to_numpy()
    y = dataset['ventas'].to_numpy()

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear carpeta si no existe
    carpeta_modelos = 'modelos'
    os.makedirs(carpeta_modelos, exist_ok=True)

    # Diccionario de modelos
    modelos = {
        'linear_regression': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(random_state=42),
        'knn': KNeighborsRegressor(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Lista para almacenar resultados
    resultados = []

    # Entrenamiento de cada modelo
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        resultados.append({
            'modelo': nombre,
            'mse': round(mse, 2),
            'r2_score': round(r2, 4)
        })

        # Guardar modelo entrenado
        joblib.dump(modelo, os.path.join(carpeta_modelos, f'{nombre}.pkl'))

    # Guardar resultados
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv(os.path.join(carpeta_modelos, 'resultados_modelos.csv'), index=False)
    print("Todos los modelos y métricas han sido guardados en la carpeta 'modelos/'.")


if __name__ == "__main__":
    entrenar()
