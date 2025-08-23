from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
import random

# configura rutas de templates/static según tu estructura
app = Flask(__name__,template_folder='../templates',static_folder='../static')

# carga tu modelo y lista de columnas (se asume ../modelo/modelo.pkl y columnas.pkl)
modelo = pickle.load(open(os.path.join(os.path.dirname(__file__), '../modelo/modelo.pkl'), 'rb'))
columnas = pickle.load(open(os.path.join(os.path.dirname(__file__), '../modelo/columnas.pkl'), 'rb'))

# Nombres solicitados
nombres = [
    "radio_promedio",
    "perímetro_promedio",
    "área_promedio",
    "concavidad_promedio",
    "puntos_concavos_promedio",
    "radio_peor",
    "perímetro_peor",
    "área_peor",
    "concavidad_peor",
    "puntos_concavos_peor"
]

# RANGOS INTERNOS: modifica aquí los rangos (min, max) para cada variable según desees.
# Los valores se generan con uniform(min,max) y se redondean a 4 decimales.
rangos = {
    "radio_promedio": (5.0, 30.0),
    "perímetro_promedio": (40.0, 200.0),
    "área_promedio": (130.0, 2500.0),
    "concavidad_promedio": (0.0, 0.5),
    "puntos_concavos_promedio": (0.0, 0.20),
    "radio_peor": (5.0, 40.0),
    "perímetro_peor": (50.0, 300.0),
    "área_peor": (180.0, 5000.0),
    "concavidad_peor": (0.0, 1.5),
    "puntos_concavos_peor": (0.0, 35.0)
}

@app.route('/', methods=['GET'])
def formulario():
    # entrega la lista de nombres para que el template construya las casillas
    return render_template('formulario.html', nombres=nombres)

@app.route('/generar', methods=['POST'])
def generar():
    # generar valores aleatorios según rangos internos
    valores = {}
    for nombre in nombres:
        low, high = rangos.get(nombre, (0.0, 1.0))
        val = round(random.uniform(low, high), 4)
        valores[nombre] = val

    # crear dataframe (una sola fila) para la predicción
    df = pd.DataFrame([valores])
    df_encoded = pd.get_dummies(df)

    # asegurar que estén todas las columnas esperadas por el modelo
    for col in columnas:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # reindexar en el orden de 'columnas' (rellena con 0 si falta)
    X = df_encoded.reindex(columns=columnas, fill_value=0)

    # intentar predecir con el modelo
    try:
        pred = modelo.predict(X)[0]
        # si quieres redondear:
        try:
            pred_display = float(pred)
            pred_display = round(pred_display, 4)
        except:
            pred_display = pred
        error = None
    except Exception as e:
        pred_display = None
        error = str(e)

    # renderizar resultado con la tabla de valores y la predicción
    return render_template('resultado.html', valores=valores, prediccion=pred_display, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
