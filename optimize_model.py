import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import numpy as np

# Cargar datos
df = pd.read_csv("dataset.csv")

# Separar variables predictoras y target
X = df.drop("ventas", axis=1)
y = df["ventas"]

# ---------- Modelo 1: Gradient Boosting ----------
print("🔧 Optimización - Gradient Boosting")

gb_params = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4]
}

gb_model = GradientBoostingRegressor(random_state=42)
grid_gb = GridSearchCV(gb_model, gb_params, cv=5, scoring='r2')
grid_gb.fit(X, y)

best_gb = grid_gb.best_estimator_
gb_cv_score = cross_val_score(best_gb, X, y, cv=5, scoring="r2")
print("Gradient Boosting - Mejores hiperparámetros:", grid_gb.best_params_)
print("Gradient Boosting - R2 promedio (CV):", round(np.mean(gb_cv_score), 4))

# Guardar modelo optimizado
joblib.dump(best_gb, "modelos/gradient_boosting_optimized.pkl")


# ---------- Modelo 2: Random Forest ----------
print("\n🔧 Optimización - Random Forest")

rf_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_model = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2')
grid_rf.fit(X, y)

best_rf = grid_rf.best_estimator_
rf_cv_score = cross_val_score(best_rf, X, y, cv=5, scoring="r2")
print("Random Forest - Mejores hiperparámetros:", grid_rf.best_params_)
print("Random Forest - R2 promedio (CV):", round(np.mean(rf_cv_score), 4))

# Guardar modelo optimizado
joblib.dump(best_rf, "modelos/random_forest_optimized.pkl")
