# -*- coding: utf-8 -*-
"""
streamlit_app.py
----------------------------------
Interfaz gráfica con Streamlit para entrenar y jugar 'Triki' (Tic-Tac-Toe)
usando un agente Q-Learning que puede seguir aprendiendo durante las partidas.

Cómo ejecutar localmente (con Python 3.10+ recomendado):
    1) Instala dependencias:
        pip install -r requirements.txt
    2) Ejecuta la app:
        streamlit run streamlit_app.py

La interfaz te permite:
    - Entrenar al agente contra un oponente aleatorio.
    - Jugar contra el agente con X u O.
    - (Opcional) Que el agente siga aprendiendo mientras juega contra ti.
    - Guardar y cargar el modelo (Q-table) a/desde JSON.
"""

import json
import time
from pathlib import Path

import streamlit as st

from tictactoe_rl import (
    TicTacToeEnv,
    RLAgent,
    X, O, EMPTY,
    state_to_key,
    key_to_state,
    evaluate_winner,
    reward_from_perspective,
    agent_move,
)

# --------------------------------------------------------------------------------------
# Configuración básica de la página
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Triki con Q-Learning", page_icon="🎮", layout="centered")

# --------------------------------------------------------------------------------------
# Utilidades UI
# --------------------------------------------------------------------------------------

def render_board(state, disabled=False):
    """Dibuja el tablero 3x3 en la interfaz como una cuadrícula de botones."""
    symbols = {X: "X", O: "O", EMPTY: " "}
    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            idx = row * 3 + col
            label = symbols[state[idx]]
            with cols[col]:
                st.button(
                    label if label.strip() else "·",
                    key=f"cell_{idx}_{time.time()}",
                    disabled=disabled or state[idx] != EMPTY or st.session_state.game_over,
                    on_click=lambda i=idx: human_click_cell(i),
                    use_container_width=True,
                )


def human_click_cell(i: int):
    """Callback cuando el humano hace clic en una celda."""
    if st.session_state.game_over:
        return

    env = st.session_state.env
    agent = st.session_state.agent

    # Juega humano si la celda está libre y es su turno
    if env.state[i] == EMPTY and env.current_player == st.session_state.human_player:
        next_state, winner, done = env.step(i)
        st.session_state.last_agent_s_key = st.session_state.last_agent_s_key  # sin cambios aquí

        if done:
            end_game_update(winner, last_update_from_opponent=True)
            return

        # Turno del agente (automatizado)
        if env.current_player == st.session_state.agent_player:
            next_state, winner, done, s_key, action = agent_move(
                env, agent, st.session_state.agent_player, learn=st.session_state.learn_during_play
            )
            # Guardamos la última jugada del agente por si el humano cierra el juego después
            st.session_state.last_agent_s_key = s_key
            st.session_state.last_agent_action = action

            if done:
                end_game_update(winner, last_update_from_opponent=False)


def end_game_update(winner, last_update_from_opponent: bool):
    """Gestiona el cierre de la partida, contadores y actualización final del agente si aplica."""
    st.session_state.game_over = True
    if winner == X:
        msg = "¡Ganó X!"
    elif winner == O:
        msg = "¡Ganó O!"
    else:
        msg = "¡Empate!"

    # Estadísticas
    if winner == st.session_state.agent_player:
        st.session_state.stats["Agente"] += 1
    elif winner == st.session_state.human_player:
        st.session_state.stats["Humano"] += 1
    else:
        st.session_state.stats["Empates"] += 1

    st.session_state.status_msg = msg

    # Si el humano cerró el juego y queremos aprendizaje, actualizamos la última jugada del agente
    if last_update_from_opponent and st.session_state.learn_during_play:
        if st.session_state.last_agent_s_key is not None and st.session_state.last_agent_action is not None:
            agent = st.session_state.agent
            env = st.session_state.env
            final_r = reward_from_perspective(winner, st.session_state.agent_player)
            agent.update(
                st.session_state.last_agent_s_key,
                st.session_state.last_agent_action,
                final_r,
                state_to_key(env.state),
                [],
                True,
            )


def reset_game(new_start: str = "auto"):
    """Reinicia el juego con tablero limpio y, opcionalmente, decide quién empieza."""
    if new_start == "Humano":
        st.session_state.env.reset(st.session_state.human_player)
    elif new_start == "Agente":
        st.session_state.env.reset(st.session_state.agent_player)
    else:
        # Automático: aleatorio entre X y O
        who = st.session_state.agent_player if st.session_state.start_random else st.session_state.human_player
        st.session_state.env.reset(who if st.session_state.start_random else st.session_state.human_player)

    st.session_state.game_over = False
    st.session_state.status_msg = "Tu turno" if st.session_state.env.current_player == st.session_state.human_player else "Turno del agente"
    st.session_state.last_agent_s_key = None
    st.session_state.last_agent_action = None

    # Si comienza el agente, que haga su primer movimiento
    if st.session_state.env.current_player == st.session_state.agent_player and not st.session_state.game_over:
        next_state, winner, done, s_key, action = agent_move(
            st.session_state.env,
            st.session_state.agent,
            st.session_state.agent_player,
            learn=st.session_state.learn_during_play,
        )
        st.session_state.last_agent_s_key = s_key
        st.session_state.last_agent_action = action
        if done:
            end_game_update(winner, last_update_from_opponent=False)


# --------------------------------------------------------------------------------------
# Estado de la sesión (persistente mientras viva la app)
# --------------------------------------------------------------------------------------
if "env" not in st.session_state:
    st.session_state.env = TicTacToeEnv(start_player=X)

if "agent" not in st.session_state:
    st.session_state.agent = RLAgent(alpha=0.4, gamma=0.99, epsilon=0.15)

if "human_player" not in st.session_state:
    st.session_state.human_player = O  # por defecto humano = O
    st.session_state.agent_player = X

if "game_over" not in st.session_state:
    st.session_state.game_over = False
    st.session_state.status_msg = "¡Bienvenido! Elige quién juega con X/O y entrena al agente."
    st.session_state.last_agent_s_key = None
    st.session_state.last_agent_action = None
    st.session_state.stats = {"Agente": 0, "Humano": 0, "Empates": 0}

if "learn_during_play" not in st.session_state:
    st.session_state.learn_during_play = True

if "start_random" not in st.session_state:
    st.session_state.start_random = True

# --------------------------------------------------------------------------------------
# Sidebar: Configuración, entrenamiento, guardar/cargar
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración & Entrenamiento")

# Config jugador
side_cols = st.sidebar.columns(2)
with side_cols[0]:
    human_symbol = st.selectbox("Ficha del humano", options=["X", "O"], index=1)
with side_cols[1]:
    start_mode = st.selectbox("Quién inicia", options=["Auto", "Humano", "Agente"], index=0)

# Actualiza símbolos
if human_symbol == "X" and st.session_state.human_player != X:
    st.session_state.human_player = X
    st.session_state.agent_player = O
elif human_symbol == "O" and st.session_state.human_player != O:
    st.session_state.human_player = O
    st.session_state.agent_player = X

st.session_state.start_random = (start_mode == "Auto")

st.session_state.learn_during_play = st.sidebar.toggle("Aprender durante la partida", value=True,
                                                      help="Si está activo, el agente actualiza su Q-table mientras juega contigo.")

# Hiperparámetros
st.sidebar.subheader("Hiperparámetros del agente")
alpha = st.sidebar.slider("α (tasa de aprendizaje)", 0.01, 1.0, st.session_state.agent.alpha, 0.01)
gamma = st.sidebar.slider("γ (descuento futuro)", 0.50, 0.999, st.session_state.agent.gamma, 0.001)
epsilon = st.sidebar.slider("ε (exploración)", 0.0, 1.0, st.session_state.agent.epsilon, 0.01)

# Actualiza en el agente
st.session_state.agent.alpha = alpha
st.session_state.agent.gamma = gamma
st.session_state.agent.epsilon = epsilon

# Entrenamiento
st.sidebar.subheader("Entrenamiento")
episodes = st.sidebar.number_input("Episodios de entrenamiento", min_value=1, max_value=100_000, value=5_000, step=100)
train_btn = st.sidebar.button("🏋️ Entrenar ahora")

if train_btn:
    from tictactoe_rl import train_episode  # import local para evitar recargas innecesarias

    prog = st.sidebar.progress(0, text="Entrenando...")
    wins = {X: 0, O: 0, 0: 0}
    for i in range(episodes):
        winner = train_episode(st.session_state.env, st.session_state.agent, agent_player=st.session_state.agent_player)
        wins[winner] += 1
        if (i + 1) % max(episodes // 100, 1) == 0:
            prog.progress((i + 1) / episodes, text=f"Episodio {i+1}/{episodes}")
    prog.empty()
    st.sidebar.success(f"✅ Entrenamiento completado. Agente ganó: {wins[st.session_state.agent_player]}, " 
                       f"Humano (oponente aleatorio): {wins[st.session_state.human_player]}, Empates: {wins[0]}")

# Guardar / Cargar modelo
st.sidebar.subheader("Modelo (Q-table)")
save_name = st.sidebar.text_input("Nombre de archivo para guardar", "q_table.json")
if st.sidebar.button("💾 Guardar modelo"):
    payload = st.session_state.agent.to_json()
    Path(save_name).write_text(payload, encoding="utf-8")
    st.sidebar.success(f"Modelo guardado en {save_name}")

file_up = st.sidebar.file_uploader("Cargar modelo (.json)", type=["json"])
if file_up is not None:
    try:
        data = file_up.read().decode("utf-8")
        st.session_state.agent = RLAgent.from_json(data)
        st.sidebar.success("Modelo cargado correctamente.")
    except Exception as e:
        st.sidebar.error(f"Error al cargar: {e}")

# --------------------------------------------------------------------------------------
# Contenido principal
# --------------------------------------------------------------------------------------
st.title("🎮 Triki (Tic-Tac-Toe) con Aprendizaje por Refuerzo")
st.caption("Agente Q-Learning tabular: entrena, juega y sigue aprendiendo.")

# Controles de partida
ccols = st.columns(3)
with ccols[0]:
    if st.button("🔄 Nuevo juego"):
        reset_game(new_start="auto" if st.session_state.start_random else "Humano")
with ccols[1]:
    if st.button("👤 Empieza el humano"):
        reset_game(new_start="Humano")
with ccols[2]:
    if st.button("🤖 Empieza el agente"):
        reset_game(new_start="Agente")

st.markdown("---")

# Tablero
render_board(st.session_state.env.state)

# Estado / mensajes
st.info(st.session_state.status_msg)

# Estadísticas de la sesión
st.subheader("Estadísticas de esta sesión")
st.write(f"**Agente:** {st.session_state.stats['Agente']}  |  **Humano:** {st.session_state.stats['Humano']}  |  **Empates:** {st.session_state.stats['Empates']}")

# Instrucciones didácticas
with st.expander("📘 ¿Cómo funciona este agente Q-Learning?"):
    st.markdown(
        """
**Idea básica (Q-Learning):** El agente aprende una tabla Q(s, a) que estima la calidad de
hacer la acción *a* en el estado *s*. En cada jugada actualiza la estimación con la regla:

\\[ Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s', a') - Q(s,a)] \\]

- **α (alpha)**: cuánto corrige cuando recibe nueva información (tasa de aprendizaje).
- **γ (gamma)**: cuánto valora el futuro (factor de descuento).
- **ε (epsilon)**: con qué probabilidad explora (elige movimientos aleatorios).

En el entrenamiento, el agente juega contra un oponente aleatorio. Durante la partida
contra un humano, si activas *Aprender durante la partida*, seguirá actualizando su Q-table
para adaptarse a tu estilo de juego.
"""
    )
