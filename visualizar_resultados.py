import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def visualizar():
    carpeta_modelos = "modelos"
    carpeta_graficos = "graficos"
    archivo_resultados = "resultados_modelos.csv"
    ruta_resultados = os.path.join(carpeta_modelos, archivo_resultados)

    # Crear carpeta de gráficos si no existe
    os.makedirs(carpeta_graficos, exist_ok=True)

    # Leer resultados
    resultados = pd.read_csv(ruta_resultados)
    print("Resultados de evaluación de modelos:")
    print(resultados)

    # Crear gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(resultados['modelo'], resultados['r2_score'], color='salmon')
    plt.title('Comparación de modelos según R² Score')
    plt.xlabel('Modelo')
    plt.ylabel('R² Score')

    min_r2 = resultados['r2_score'].min()
    plt.ylim(min_r2 - 0.1, 0.1)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Construir nombre dinámico con timestamp y base del dataset
    base_nombre = os.path.splitext("dataset.csv")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"grafico_{base_nombre}_{timestamp}.png"
    ruta_grafico = os.path.join(carpeta_graficos, nombre_archivo)

    # Guardar gráfico
    plt.savefig(ruta_grafico)
    plt.show()
    print(f"📊 Gráfico guardado como: {ruta_grafico}")

if __name__ == "__main__":
    visualizar()
