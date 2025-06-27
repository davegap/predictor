# Imagen base oficial de Python
FROM python:3.13.5

# Establecer directorio de trabajo
WORKDIR /predictor

# Copiar archivos del proyecto al contenedor
COPY modelos/gradient_boosting.pkl modelos/gradient_boosting.pkl
COPY api.py requirements.txt ./

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar tu script principal
CMD ["python", "api.py"]
