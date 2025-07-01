import joblib
import pandas as pd
import os

def listar_modelos():
    modelos = []
    for archivo in os.listdir("modelos"):
        if archivo.endswith(".pkl"):
            modelos.append(archivo.replace(".pkl", ""))
    return modelos

def mostrar_menu(modelos):
    print("📦 Modelos disponibles:")
    for i, modelo in enumerate(modelos, start=1):
        print(f"  {i}. {modelo}")
    while True:
        try:
            opcion = int(input("\nSelecciona el número del modelo a utilizar: "))
            if 1 <= opcion <= len(modelos):
                return modelos[opcion - 1]
            else:
                print("⚠️ Opción inválida. Intenta nuevamente.")
        except ValueError:
            print("⚠️ Entrada no válida. Debes ingresar un número.")

def predecir(modelo_nombre, precio, promocion, temporada, mes, dia):
    ruta = f"modelos/{modelo_nombre}.pkl"
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"❌ El modelo '{modelo_nombre}' no existe.")

    modelo = joblib.load(ruta)

    entrada = pd.DataFrame([{
        "temporada": temporada,
        "color": 1,
        "talla": 2,
        "material": 1,
        "genero": 0,
        "precio": precio,
        "ubicacion": 0,
        "promocion": promocion,
        "anio": 2023,
        "mes": mes,
        "dia": dia
    }])

    return modelo.predict(entrada)[0]

if __name__ == "__main__":
    print("🧥 Predicción de Ventas de Chaquetas\n")

    modelos_disponibles = listar_modelos()
    modelo = mostrar_menu(modelos_disponibles)

    precio = float(input("\nPrecio de la chaqueta: "))
    promocion = int(input("¿Tiene promoción? (0=No, 1=Sí): "))
    temporada = int(input("Temporada (0=Verano, 1=Otoño, 2=Invierno, 3=Primavera): "))
    mes = int(input("Mes de venta (1-12): "))
    dia = int(input("Día de venta (1-31): "))

    try:
        resultado = predecir(modelo, precio, promocion, temporada, mes, dia)
        print(f"\n📈 Ventas estimadas con '{modelo}': {round(resultado, 2)} unidades")
    except Exception as e:
        print(f"\n❌ Error al predecir: {e}")
