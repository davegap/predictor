# 🧥 Proyecto_Chaquetas - Predicción de Ventas con Machine Learning

Este repositorio contiene un proyecto de machine learning orientado a predecir la cantidad de ventas de chaquetas a partir de características como temporada, precio, promoción, fecha de venta, entre otras.

## 📁 Estructura del proyecto

```
Proyecto_Chaquetas/
├── graficos/                   # Imágenes generadas por visualización
├── modelos/                    # Modelos entrenados (.pkl)
├── dataset.csv                 # Datos históricos de ventas
├── train.py                    # Entrenamiento de modelos
├── visualizar_resultados.py    # Comparación visual de modelos
├── api.py                      # Predicción manual desde consola
├── requirements.txt            # Librerías necesarias
├── Dockerfile                  # Configuración de contenedor Docker
└── README.md                   # Descripción del proyecto
```

## ⚙️ Requisitos

- Python 3.10 o superior
- Se recomienda uso de entorno virtual (`venv` o `conda`)

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## 🚀 Instrucciones de uso

### 1. Entrenar los modelos – `train.py`

```bash
python train.py
```

Este script se encarga de:

- Cargar el archivo `dataset.csv` con los datos históricos.
- Entrenar 5 modelos de machine learning:
  - `LinearRegression`
  - `DecisionTreeRegressor`
  - `KNeighborsRegressor`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
- Evaluar cada modelo utilizando métricas como **R² Score** y **MSE**.
- Guardar los modelos entrenados en la carpeta `modelos/` como archivos `.pkl`.
- Generar un archivo `resultados_modelos.csv` con las métricas de evaluación de cada modelo.

📁 **Resultado esperado**:
- Carpeta `modelos/` con 5 archivos `.pkl`
- Archivo `resultados_modelos.csv` con la tabla comparativa de desempeño de los modelos.

---

### 2. Visualizar resultados – `visualizar_resultados.py`

```bash
python visualizar_resultados.py
```

Este script:

- Lee el archivo `resultados_modelos.csv`.
- Genera un gráfico de barras comparando el **R² Score** de los 5 modelos.
- Guarda automáticamente la imagen del gráfico en la carpeta `graficos/` con fecha y hora en el nombre.

📁 **Resultado esperado**:
- Carpeta `graficos/` con una imagen `.png` como `grafico_dataset_YYYYMMDD_HHMMSS.png`.

---

### 3. Realizar predicciones manuales – `api.py`

```bash
python api.py
```

Este script interactivo:

- Solicita al usuario los 11 datos requeridos para hacer una predicción (temporada, color, talla, precio, etc.).
- Carga el modelo `gradient_boosting.pkl`.
- Calcula la predicción de unidades vendidas.
- Muestra el resultado estimado en pantalla.

🧠 Este paso simula cómo se utilizaría el modelo ya entrenado en un entorno de consulta real.

📈 **Resultado esperado**:
- Un mensaje en consola como: `📈 Ventas estimadas: 16.2 unidades`

---

## 🐳 Uso con Docker (opcional)

### Construir imagen

```bash
docker build -t predictor_usc .
```

### Ejecutar contenedor

```bash
docker run -it predictor_usc
```

---

## 👨‍💻 Autores

- **David Alfonso Vega Palacio**  
- **Manuel Alejandro Mena López**  
- **Joan Andrés Buriticá Salazar**  
**Universidad Santiago de Cali**  
*Maestría en Informática*