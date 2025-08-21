
# Triki (Tic-Tac-Toe) con Q-Learning

Aplicación educativa en **Python + Streamlit** para enseñar **Aprendizaje por Refuerzo tabular** (Q-Learning) entrenando un agente que juega **Triki**.

## 🧰 Estructura
- `tictactoe_rl.py`: Entorno y agente Q-Learning **documentados en español**.
- `streamlit_app.py`: Interfaz gráfica para **entrenar** y **jugar** contra el agente.
- `requirements.txt`: Dependencias mínimas.

## ▶️ Cómo ejecutar
1. **(Opcional)** Crea y activa un entorno virtual.
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la app:
   ```bash
   streamlit run streamlit_app.py
   ```

> **Requisitos**: Python 3.10+ (probado con 3.12).

## 🎮 Cómo se usa
- En la barra lateral:
  - Ajusta hiperparámetros `α`, `γ`, `ε`.
  - Entrena el agente por *N* episodios (contra un oponente aleatorio).
  - Activa "Aprender durante la partida" para que el modelo **siga aprendiendo** cuando juegue contigo.
  - Guarda / carga el modelo (Q-table) a/desde un archivo JSON.
- En la vista principal:
  - Reinicia la partida o elige quién empieza (humano/agente).
  - Haz clic en una celda para jugar.

## 🧠 Qué aprende el agente
El agente maximiza recompensa: **+1 ganar, -1 perder, 0 empatar**.
Usa **ε-greedy** para explorar y la regla de actualización de **Q-Learning**:

```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

## 💾 Persistencia del modelo
- Guarda el modelo con **"Guardar modelo"** (genera `q_table.json`).
- Cárgalo con **"Cargar modelo"** para continuar más tarde.

## 📜 Licencia
MIT — úsalo libremente en clases, demos y talleres.
