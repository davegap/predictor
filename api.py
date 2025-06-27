import joblib
import numpy as np

def predecir(precio, promocion, temporada, mes, dia):
    # Cargar modelo entrenado
    modelo = joblib.load("modelos/gradient_boosting.pkl")

    # Valores fijos para los campos no solicitados
    color = 1
    talla = 2
    material = 1
    genero = 0
    ubicacion = 0
    anio = 2023

    # Vector de entrada en orden
    entrada = np.array([temporada, color, talla, material, genero,
                        precio, ubicacion, promocion, anio, mes, dia]).reshape(1, -1)

    # Realizar predicción
    prediccion = modelo.predict(entrada)
    return prediccion[0]

if __name__ == "__main__":
    print("Predicción de Ventas de Chaquetas\n")

    # Capturar entradas desde teclado
    precio = float(input("Precio de la chaqueta: "))
    promocion = int(input("¿Tiene promoción? (0=No, 1=Sí): "))
    temporada = int(input("Temporada (0=Verano, 1=Otoño, 2=Invierno, 3=Primavera): "))
    mes = int(input("Mes de venta (1-12): "))
    dia = int(input("Día de venta (1-31): "))

    resultado = predecir(precio, promocion, temporada, mes, dia)
    print(f"\n📈 Ventas estimadas: {round(resultado, 2)} unidades")
