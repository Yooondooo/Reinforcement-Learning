import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.model = self._build_optimized_model()
        self.target_model = self._build_optimized_model()
        self.update_target_model()
        self.batch_size = 128
        self.train_interval = 4
        self.train_counter = 0
        self.episode_count = 0

    def _build_optimized_model(self):
        """Оптимальная архитектура для лабиринта 16×16"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,),
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        return model

    def act(self, state, agent_pos=None, maze=None):
        """
        УЛУЧШЕННЫЙ выбор действия для больших лабиринтов
        """
        state = self._normalize_state(state)
        if np.random.random() <= self.epsilon:
            action = self._smart_exploration(agent_pos, maze)
        else:
            act_values = self.model(state, training=False)
            action = np.argmax(act_values[0])

        return action

    def _smart_exploration(self, agent_pos, maze):
        """
        Умное исследование, учитывающее структуру лабиринта
        """
        if agent_pos is None or maze is None:
            return random.randrange(self.action_size)
        possible_actions = []
        x, y = agent_pos
        directions = [
            (0, -1, 0),
            (1, 0, 1),
            (0, 1, 2),
            (-1, 0, 3)
        ]

        for dx, dy, action in directions:
            new_x, new_y = x + dx, y + dy
            if (len(maze[0]) > new_x >= 0 == maze[new_y][new_x] and
                    0 <= new_y < len(maze)):
                possible_actions.append(action)
        if possible_actions:
            return random.choice(possible_actions)
        else:
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """Сохраняем опыт с нормализованным форматом"""
        state = self._normalize_state(state)
        next_state = self._normalize_state(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def _normalize_state(self, state):
        """Приводим state к стандартному формату"""
        if isinstance(state, (list, np.ndarray)):
            state = np.array(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        elif state.ndim == 3:
            state = state.reshape(state.shape[0], -1)
        return state

    def replay(self):
        """Метод обучения"""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.vstack([transition[0] for transition in minibatch])
        next_states = np.vstack([transition[3] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        target_q = current_q.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        target_q[batch_index, actions] = rewards + self.gamma * np.max(next_q, axis=1) * ~dones
        self.model.fit(states, target_q, epochs=1, verbose=0, batch_size=self.batch_size)

    def end_episode(self):
        """Обновление в конце эпизода"""
        self.episode_count += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if hasattr(self, 'last_actions'):
            self.last_actions.clear()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name if name.endswith('.weights.h5') else name + '.weights.h5')

    def save(self, name):
        self.model.save_weights(name if name.endswith('.weights.h5') else name + '.weights.h5')
