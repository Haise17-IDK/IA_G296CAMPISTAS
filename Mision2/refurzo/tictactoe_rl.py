# -*- coding: utf-8 -*-
"""
tictactoe_rl.py
----------------------------------
Entorno de 'Triki' (Tic-Tac-Toe) + Agente Q-Learning en Python.

✔ Pensado para enseñar los conceptos de aprendizaje por refuerzo tabular.
✔ Documentado en español, con funciones claras y comentarios didácticos.
✔ El agente puede:
    - Entrenarse contra un oponente aleatorio.
    - Seguir aprendiendo cuando juega contra un usuario humano.

Autor: Luis Fernando Gallego H
Licencia: MIT
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple, Dict
import json

# --------------------------------------------------------------------------------------
# Utilidades y Constantes
# --------------------------------------------------------------------------------------

# Representaremos el tablero como una lista de 9 enteros:
# 1  -> 'X'
# -1 -> 'O'
# 0  -> celda vacía
#
# El jugador 'X' empieza por convención, pero lo hacemos configurable.
X = 1
O = -1
EMPTY = 0

WIN_COMBOS = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def state_to_key(state: List[int]) -> str:
    """Convierte un estado (lista de 9 celdas) a una clave de texto.
    Esto hace que el diccionario Q sea serializable y legible.
    """
    m = {X: "X", O: "O", EMPTY: "-"}
    return "".join(m[v] for v in state)


def key_to_state(key: str) -> List[int]:
    """Inversa de state_to_key (por si la necesitas)."""
    m = {"X": X, "O": O, "-": EMPTY}
    return [m[ch] for ch in key]


def switch_player(player: int) -> int:
    """Cambia de jugador: X -> O y O -> X."""
    return -player


def evaluate_winner(state: List[int]) -> Optional[int]:
    """Determina si hay un ganador en el estado dado.
    Devuelve:
        X  (1) si gana X,
        O (-1) si gana O,
        0 si hay empate (tablero lleno sin ganador),
        None si el juego continúa.
    """
    for a, b, c in WIN_COMBOS:
        s = state[a] + state[b] + state[c]
        if s == 3:   # X X X
            return X
        if s == -3:  # O O O
            return O

    if EMPTY not in state:
        return 0  # Empate

    return None  # Sigue en juego


# --------------------------------------------------------------------------------------
# Entorno de Triki
# --------------------------------------------------------------------------------------

class TicTacToeEnv:
    """Entorno simple de Triki.

    Métodos clave:
        - reset(start_player=X): reinicia el tablero.
        - step(action): aplica una acción (0..8).
        - valid_actions(state=None): lista de acciones válidas (celdas vacías).
    """

    def __init__(self, start_player: int = X):
        self.state: List[int] = [EMPTY] * 9
        self.current_player: int = start_player

    def reset(self, start_player: Optional[int] = None) -> List[int]:
        """Resetea el tablero y decide quién empieza."""
        self.state = [EMPTY] * 9
        if start_player is None:
            start_player = X
        self.current_player = start_player
        return self.state.copy()

    def valid_actions(self, state: Optional[List[int]] = None) -> List[int]:
        """Devuelve las posiciones (índices 0..8) disponibles para jugar."""
        s = self.state if state is None else state
        return [i for i, v in enumerate(s) if v == EMPTY]

    def step(self, action: int) -> Tuple[List[int], Optional[int], bool]:
        """Aplica la acción del jugador actual.

        Args:
            action: índice de 0 a 8 donde colocar la ficha.

        Returns:
            next_state: nuevo estado del tablero.
            winner: X, O, 0 (empate) o None si no hay final aún.
            done: bool que indica si el juego terminó.
        """
        if action not in self.valid_actions():
            raise ValueError(f"Acción inválida {action}. Celdas libres: {self.valid_actions()}")

        self.state[action] = self.current_player
        winner = evaluate_winner(self.state)
        done = winner is not None

        # Cambiamos de jugador sólo si el juego no terminó
        if not done:
            self.current_player = switch_player(self.current_player)

        return self.state.copy(), winner, done


# --------------------------------------------------------------------------------------
# Agente Q-Learning
# --------------------------------------------------------------------------------------

class RLAgent:
    """Agente tabular con Q-Learning para jugar Triki.

    Q[(state_key, action)] = valor esperado de tomar 'action' en 'state_key'.

    Parámetros:
        alpha (tasa de aprendizaje): cuánto corriges el valor en cada actualización.
        gamma (descuento): importancia del futuro.
        epsilon (exploración): probabilidad de tomar una acción aleatoria.
    """

    def __init__(self, alpha: float = 0.4, gamma: float = 0.99, epsilon: float = 0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[Tuple[str, int], float] = {}

    # -------------------------
    # Política (elección acción)
    # -------------------------
    def choose_action(self, state: List[int], valid_actions: List[int]) -> int:
        """Política ε-greedy: con prob. ε elige al azar, si no elige la mejor acción conocida."""
        if not valid_actions:
            raise ValueError("No hay acciones válidas disponibles.")

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Elegir acción con mayor Q; desempatar al azar
        state_key = state_to_key(state)
        q_vals = [(a, self.Q.get((state_key, a), 0.0)) for a in valid_actions]
        max_q = max(q_vals, key=lambda t: t[1])[1]
        best_actions = [a for a, q in q_vals if q == max_q]
        return random.choice(best_actions)

    # -------------------------
    # Actualización Q-Learning
    # -------------------------
    def update(self,
               s_key: str,
               action: int,
               reward: float,
               next_s_key: Optional[str],
               next_valid_actions: List[int],
               done: bool):
        """Regla de actualización de Q-Learning.

        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

        Si 'done' es True, entonces max_a' Q(s',a') = 0 por convención.
        """
        sa = (s_key, action)
        q_sa = self.Q.get(sa, 0.0)

        if done or next_s_key is None or not next_valid_actions:
            td_target = reward
        else:
            max_next = max(self.Q.get((next_s_key, a2), 0.0) for a2 in next_valid_actions)
            td_target = reward + self.gamma * max_next

        self.Q[sa] = q_sa + self.alpha * (td_target - q_sa)

    # -------------------------
    # Persistencia del modelo
    # -------------------------
    def to_json(self) -> str:
        """Serializa la Q-table y los hiperparámetros a JSON."""
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "Q": {f"{k[0]}|{k[1]}": v for k, v in self.Q.items()},
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(s: str) -> "RLAgent":
        """Construye un agente desde un JSON generado por to_json()."""
        data = json.loads(s)
        agent = RLAgent(alpha=data["alpha"], gamma=data["gamma"], epsilon=data["epsilon"])
        agent.Q = {}
        for k_str, v in data["Q"].items():
            skey, a_str = k_str.split("|")
            agent.Q[(skey, int(a_str))] = float(v)
        return agent


# --------------------------------------------------------------------------------------
# Entrenamiento y juego
# --------------------------------------------------------------------------------------

def reward_from_perspective(winner: Optional[int], agent_player: int) -> float:
    """Calcula la recompensa final desde la perspectiva del agente.
    +1 si gana el agente, -1 si pierde, 0 si empata o no ha terminado.
    """
    if winner is None:
        return 0.0
    if winner == agent_player:
        return 1.0
    if winner == 0:
        return 0.0
    return -1.0


def train_episode(env: TicTacToeEnv, agent: RLAgent, agent_player: int = X) -> int:
    """Ejecuta UN episodio de entrenamiento contra un oponente aleatorio.

    - Sólo actualizamos Q cuando el que juega es el agente.
    - Si la jugada del oponente termina el juego, realizamos una última
      actualización con la recompensa final para la ÚLTIMA jugada del agente.

    Devuelve:
        winner: X, O o 0 (empate)
    """
    state = env.reset(start_player=random.choice([X, O]))  # aleatorizamos quién empieza
    last_agent_s_key = None
    last_agent_action = None

    while True:
        valid = env.valid_actions(state)
        if env.current_player == agent_player:
            # Turno del agente
            action = agent.choose_action(state, valid)
            s_key = state_to_key(state)
            next_state, winner, done = env.step(action)

            # Guardar última jugada del agente (por si el oponente cierra el juego)
            last_agent_s_key = s_key
            last_agent_action = action

            # Recompensa intermedia (0 si no termina)
            r = reward_from_perspective(winner, agent_player)
            agent.update(
                s_key,
                action,
                r if done else 0.0,
                state_to_key(next_state),
                env.valid_actions(next_state),
                done,
            )

            state = next_state
            if done:
                return winner
        else:
            # Turno del oponente: juega aleatorio
            action = random.choice(valid)
            next_state, winner, done = env.step(action)
            state = next_state

            if done:
                # Si el oponente cerró el juego, actualizamos la última acción del agente
                if last_agent_s_key is not None and last_agent_action is not None:
                    final_r = reward_from_perspective(winner, agent_player)
                    agent.update(
                        last_agent_s_key,
                        last_agent_action,
                        final_r,
                        state_to_key(state),
                        [],
                        True,
                    )
                return winner


def agent_move(env: TicTacToeEnv, agent: RLAgent, agent_player: int, learn: bool = True):
    """Hace una jugada del agente dentro de una partida (por ejemplo, contra humano).

    Si 'learn' es True, actualiza la Q del agente en línea, incluyendo la corrección
    final si la partida termina con la jugada del oponente después.
    """
    state = env.state.copy()
    valid = env.valid_actions(state)
    action = agent.choose_action(state, valid)
    s_key = state_to_key(state)
    next_state, winner, done = env.step(action)

    # Actualización intermedia
    if learn:
        r = reward_from_perspective(winner, agent_player)
        agent.update(
            s_key,
            action,
            r if done else 0.0,
            state_to_key(next_state),
            env.valid_actions(next_state),
            done,
        )

    return next_state, winner, done, s_key, action


def opponent_move_random(env: TicTacToeEnv):
    """Una jugada aleatoria (para oponente básico)."""
    action = random.choice(env.valid_actions())
    next_state, winner, done = env.step(action)
    return next_state, winner, done
