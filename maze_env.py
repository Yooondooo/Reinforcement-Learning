import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time
from collections import deque


class MazeEnv(gym.Env):
    def __init__(self, maze_size=15, render_mode=None, max_steps=150, difficulty=1):
        super(MazeEnv, self).__init__()

        self.maze_size = maze_size
        self.cell_size = 30
        self.window_size = maze_size * self.cell_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.difficulty = difficulty

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(21,), dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.font = None
        if render_mode == "human":
            self._init_render()

    def _init_render(self):
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(f"RL Maze - {self.maze_size}x{self.maze_size} (Уровень {self.difficulty})")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        except Exception as e:
            print(f"Ошибка инициализации Pygame: {e}")
            self.render_mode = None

    def _generate_maze(self):
        # Генерируем лабиринт
        self.maze = np.ones((self.maze_size, self.maze_size))

        stack = []
        visited = set()

        start_x, start_y = 1, 1
        self.maze[start_y, start_x] = 0
        stack.append((start_x, start_y))
        visited.add((start_x, start_y))

        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]

        while stack:
            current_x, current_y = stack[-1]

            possible_dirs = []
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if (0 < nx < self.maze_size - 1 and
                        0 < ny < self.maze_size - 1 and
                        (nx, ny) not in visited):
                    possible_dirs.append((dx, dy, nx, ny))

            if possible_dirs:
                dx, dy, next_x, next_y = random.choice(possible_dirs)
                self.maze[current_y + dy // 2, current_x + dx // 2] = 0
                self.maze[next_y, next_x] = 0
                stack.append((next_x, next_y))
                visited.add((next_x, next_y))
            else:
                stack.pop()

        self._add_more_obstacles()

        # Сначала устанавливаем позиции агента и цели
        if random.random() < 0.5:  # 50% вероятность для каждого случая
            # Случай 1: Агент слева сверху, финиш справа снизу
            self.agent_pos = [1, 1]
            self.target_pos = [self.maze_size - 2, self.maze_size - 2]
            self.scenario = "start_to_end"
        else:
            # Случай 2: Агент справа снизу, финиш слева сверху
            self.agent_pos = [self.maze_size - 2, self.maze_size - 2]
            self.target_pos = [1, 1]
            self.scenario = "end_to_start"

        print(f"🎯 Сценарий: {self.scenario}")
        print(f"   Агент: {self.agent_pos}, Цель: {self.target_pos}")

        # Теперь гарантируем путь между установленными позициями
        self._ensure_path_to_corner()

    def _add_more_obstacles(self):
        """Добавляет дополнительные препятствия"""
        extra_walls = self.difficulty * (self.maze_size // 3)

        for _ in range(extra_walls):
            for attempt in range(10):
                x = random.randint(1, self.maze_size - 2)
                y = random.randint(1, self.maze_size - 2)
                if self.maze[y, x] == 0 and self._is_safe_wall(x, y):
                    self.maze[y, x] = 1
                    break

    def _is_safe_wall(self, x, y):
        """Проверяет, не изолирует ли стена важные области"""
        temp_maze = self.maze.copy()
        temp_maze[y, x] = 0

        visited = set()
        stack = [(1, 1)]

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.maze_size and 0 <= ny < self.maze_size and
                        temp_maze[ny, nx] == 0 and (nx, ny) not in visited):
                    stack.append((nx, ny))

        return len(visited) > (self.maze_size * self.maze_size) * 0.7

    def _ensure_path_to_corner(self):
        """Гарантирует путь между агентом и целью"""
        start = tuple(self.agent_pos)
        target = tuple(self.target_pos)

        queue = [start]
        visited = set([start])

        while queue:
            x, y = queue.pop(0)
            if (x, y) == target:
                return

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.maze_size and 0 <= ny < self.maze_size and
                        self.maze[ny, nx] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # Если путь не найден, создаем его
        self._create_path_between_points()

    def _create_path_between_points(self):
        """Создает путь между агентом и целью"""
        start_x, start_y = self.agent_pos
        target_x, target_y = self.target_pos

        x, y = start_x, start_y

        # Двигаемся сначала по X, потом по Y
        while x != target_x or y != target_y:
            if x < target_x and self.maze[y, x + 1] == 1:
                self.maze[y, x + 1] = 0
                x += 1
            elif x > target_x and self.maze[y, x - 1] == 1:
                self.maze[y, x - 1] = 0
                x -= 1
            elif y < target_y and self.maze[y + 1, x] == 1:
                self.maze[y + 1, x] = 0
                y += 1
            elif y > target_y and self.maze[y - 1, x] == 1:
                self.maze[y - 1, x] = 0
                y -= 1
            else:
                # Если не можем двигаться напрямую, ищем обходной путь
                if x < target_x:
                    x += 1
                elif x > target_x:
                    x -= 1
                elif y < target_y:
                    y += 1
                elif y > target_y:
                    y -= 1

    def _place_agent(self):
        """Размещает агента в соответствии с выбранным сценарием"""
        # Позиции уже установлены в _generate_maze
        self.current_step = 0
        self.time_remaining = self.max_steps

        # Инициализируем историю посещений
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        self.cell_visit_count = {}
        self.cell_visit_count[tuple(self.agent_pos)] = 1

    def _get_sensor_readings(self):
        """Расширенные сенсоры для больших лабиринтов"""
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),  # Ближние
            (0, 1), (-1, 1), (-1, 0), (-1, -1),  # Ближние
            (0, -2), (2, 0), (0, 2), (-2, 0),  # Дальние
            (0, -3), (3, 0), (0, 3), (-3, 0)  # Очень дальние
        ]

        sensor_readings = []
        max_distance = 12  # Увеличили дальность для 16×16

        for dx, dy in directions:
            distance = 0
            for i in range(1, max_distance + 1):
                check_x = self.agent_pos[0] + dx * i
                check_y = self.agent_pos[1] + dy * i

                if (0 <= check_x < self.maze_size and
                        0 <= check_y < self.maze_size and
                        self.maze[check_y, check_x] == 0):
                    distance = i / max_distance
                else:
                    break
            sensor_readings.append(distance)

        return sensor_readings

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_maze()  # Теперь сценарий выбирается здесь
        self._place_agent()
        observation = self._get_observation()
        info = {
            "difficulty": self.difficulty,
            "target_position": self.target_pos,
            "scenario": getattr(self, "scenario", "start_to_end")
        }
        return observation, info

    def _get_observation(self):
        agent_x = self.agent_pos[0] / (self.maze_size - 1)
        agent_y = self.agent_pos[1] / (self.maze_size - 1)
        target_x = self.target_pos[0] / (self.maze_size - 1)
        target_y = self.target_pos[1] / (self.maze_size - 1)
        rel_x = target_x - agent_x
        rel_y = target_y - agent_y
        time_remaining = self.time_remaining / self.max_steps

        sensor_readings = self._get_sensor_readings()
        observation = [agent_x, agent_y, rel_x, rel_y, time_remaining] + sensor_readings
        return np.array(observation, dtype=np.float32)

    # def step(self, action):
    #     self.current_step += 1
    #     self.time_remaining = self.max_steps - self.current_step
    #
    #     old_pos = self.agent_pos.copy()
    #     old_pos_tuple = tuple(old_pos)
    #
    #     # Движение агента
    #     if action == 0:  # вверх
    #         self.agent_pos[1] = max(1, self.agent_pos[1] - 1)
    #     elif action == 1:  # вправо
    #         self.agent_pos[0] = min(self.maze_size - 2, self.agent_pos[0] + 1)
    #     elif action == 2:  # вниз
    #         self.agent_pos[1] = min(self.maze_size - 2, self.agent_pos[1] + 1)
    #     elif action == 3:  # влево
    #         self.agent_pos[0] = max(1, self.agent_pos[0] - 1)
    #
    #     current_pos_tuple = tuple(self.agent_pos)
    #
    #     # СМЯГЧЕННАЯ СИСТЕМА НАГРАД
    #     reward = 0
    #     terminated = False
    #     truncated = False
    #
    #     # 1. Если агент пытается попасть в стену - умеренный штраф
    #     if self.maze[self.agent_pos[1], self.agent_pos[0]] == 1:
    #         self.agent_pos = old_pos  # Возвращаем на старую позицию
    #         reward -= 1.0  # Умеренный штраф за стену
    #         current_pos_tuple = tuple(self.agent_pos)
    #         # НЕ сбрасываем прогресс!
    #
    #     else:
    #         # 2. Основная награда: исследование новых клеток
    #         if current_pos_tuple not in self.visited_cells:
    #             reward += 2.0  # Хорошая награда за новую клетку
    #             self.visited_cells.add(current_pos_tuple)
    #             self.cell_visit_count[current_pos_tuple] = 1
    #
    #     # 3. УМЕРЕННЫЙ штраф за возврат на предыдущую клетку
    #     if current_pos_tuple == old_pos_tuple and not self.maze[self.agent_pos[1], self.agent_pos[0]] == 1:
    #         reward -= 2.0  # Умеренный штраф за возврат
    #
    #     # 4. ОГРОМНАЯ награда за достижение цели
    #     if self.agent_pos == self.target_pos:
    #         base_reward = 50.0
    #         exploration_bonus = len(self.visited_cells) * 1.0  # Бонус за исследование
    #         reward = base_reward + exploration_bonus
    #         terminated = True
    #         print(f"🎉 УСПЕХ! Награда: {reward:.1f}")
    #         print(f"   Исследовано клеток: {len(self.visited_cells)}")
    #
    #     # 5. Штраф за превышение времени
    #     if self.current_step >= self.max_steps:
    #         reward -= 5.0  # Умеренный штраф за время
    #         truncated = True
    #
    #     # 6. Очень маленький штраф за каждый шаг
    #     reward -= 0.01
    #
    #     info = {
    #         "steps": self.current_step,
    #         "time_remaining": self.time_remaining,
    #         "visited_cells": len(self.visited_cells)
    #     }
    #
    #     observation = self._get_observation()
    #     return observation, reward, terminated, truncated, info
    def step(self, action):
        self.current_step += 1
        self.time_remaining = self.max_steps - self.current_step

        old_pos = self.agent_pos.copy()
        old_pos_tuple = tuple(old_pos)

        # Движение агента
        if action == 0:  # вверх
            self.agent_pos[1] = max(1, self.agent_pos[1] - 1)
        elif action == 1:  # вправо
            self.agent_pos[0] = min(self.maze_size - 2, self.agent_pos[0] + 1)
        elif action == 2:  # вниз
            self.agent_pos[1] = min(self.maze_size - 2, self.agent_pos[1] + 1)
        elif action == 3:  # влево
            self.agent_pos[0] = max(1, self.agent_pos[0] - 1)

        current_pos_tuple = tuple(self.agent_pos)

        # CURIOUSITY-DRIVEN НАГРАДЫ
        reward = 0
        terminated = False
        truncated = False

        # 1. Штраф за стену
        if self.maze[self.agent_pos[1], self.agent_pos[0]] == 1:
            self.agent_pos = old_pos
            reward -= 0.5  # Маленький штраф
            current_pos_tuple = tuple(self.agent_pos)
        else:
            # 2. Награда за "редкие" состояния (посещенные мало раз)
            visit_count = self.cell_visit_count.get(current_pos_tuple, 0)
            if visit_count == 0:
                reward += 5.0  # Большая награда за первую посещение
            elif visit_count < 3:
                reward += 1.0  # Награда за редкие посещения
            else:
                reward -= 0.2  # Маленький штраф за частые посещения

            self.cell_visit_count[current_pos_tuple] = visit_count + 1
            self.visited_cells.add(current_pos_tuple)

        # 3. Бонус за разнообразие маршрутов (избегаем циклов)
        if hasattr(self, 'last_positions'):
            if current_pos_tuple in self.last_positions:
                reward -= 0.3  # Штраф за циклы
        self.last_positions = getattr(self, 'last_positions', deque(maxlen=8))
        self.last_positions.append(current_pos_tuple)

        # 4. ОГРОМНАЯ награда за достижение цели
        if self.agent_pos == self.target_pos:
            reward = 300.0  # Очень большая фиксированная награда
            terminated = True

        # 5. Мягкий штраф за время
        if self.current_step >= self.max_steps:
            reward -= 1.0
            truncated = True

        info = {
            "steps": self.current_step,
            "time_remaining": self.time_remaining,
            "visited_cells": len(self.visited_cells)
        }

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render_with_info(self, episode=None, step=None, reward=None, epsilon=None, level=None):
        if self.render_mode != "human" or self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))

        # Отрисовка лабиринта
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
                    pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)

        # Агент (синий)
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.cell_size + 2,
            self.agent_pos[1] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

        # Цель (зеленый)
        target_rect = pygame.Rect(
            self.target_pos[0] * self.cell_size + 2,
            self.target_pos[1] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, (0, 255, 0), target_rect)

        # Полупрозрачный фон для текста
        text_bg = pygame.Surface((250, 140), pygame.SRCALPHA)
        text_bg.fill((0, 0, 0, 180))
        self.screen.blit(text_bg, (5, 5))

        # Текст белым цветом
        text_color = (255, 255, 255)
        y_offset = 10

        if episode is not None:
            episode_text = self.font.render(f'Эпизод: {episode}', True, text_color)
            self.screen.blit(episode_text, (10, y_offset))
            y_offset += 20

        if level is not None:
            level_text = self.font.render(f'Уровень: {level}', True, text_color)
            self.screen.blit(level_text, (10, y_offset))
            y_offset += 20

        if step is not None:
            step_text = self.font.render(f'Шаги: {step}/{self.max_steps}', True, text_color)
            self.screen.blit(step_text, (10, y_offset))
            y_offset += 20

        if reward is not None:
            if reward >= 0:
                reward_color = (0, 255, 0)
            else:
                reward_color = (255, 100, 100)

            reward_text = self.font.render(f'Награда: {reward:.1f}', True, reward_color)
            self.screen.blit(reward_text, (10, y_offset))
            y_offset += 20

        if epsilon is not None:
            epsilon_text = self.font.render(f'Исследование: {epsilon:.3f}', True, text_color)
            self.screen.blit(epsilon_text, (10, y_offset))

        pygame.display.flip()
        self.clock.tick(30)

    def test_step(self, action):
        """
        Версия step для тестирования - с интеллектуальным избеганием стен
        """
        self.current_step += 1
        self.time_remaining = self.max_steps - self.current_step

        old_pos = self.agent_pos.copy()

        # Пробуем выполнить выбранное действие
        new_pos = self.agent_pos.copy()

        if action == 0:  # вверх
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # вправо
            new_pos[0] = min(self.maze_size - 1, new_pos[0] + 1)
        elif action == 2:  # вниз
            new_pos[1] = min(self.maze_size - 1, new_pos[1] + 1)
        elif action == 3:  # влево
            new_pos[0] = max(0, new_pos[0] - 1)

        # Проверяем, можно ли переместиться в новую позицию
        if (0 <= new_pos[0] < self.maze_size and
                0 <= new_pos[1] < self.maze_size and
                self.maze[new_pos[1], new_pos[0]] == 0):
            # Если нет стены - перемещаемся
            self.agent_pos = new_pos
            moved = True
            hit_wall = False
        else:
            # Если стена - остаемся на месте, но ищем альтернативное действие
            moved = False
            hit_wall = True

            # Пробуем найти другое возможное действие
            possible_actions = []
            for alt_action in range(4):
                alt_pos = old_pos.copy()

                if alt_action == 0:  # вверх
                    alt_pos[1] = max(0, alt_pos[1] - 1)
                elif alt_action == 1:  # вправо
                    alt_pos[0] = min(self.maze_size - 1, alt_pos[0] + 1)
                elif alt_action == 2:  # вниз
                    alt_pos[1] = min(self.maze_size - 1, alt_pos[1] + 1)
                elif alt_action == 3:  # влево
                    alt_pos[0] = max(0, alt_pos[0] - 1)

                if (0 <= alt_pos[0] < self.maze_size and
                        0 <= alt_pos[1] < self.maze_size and
                        self.maze[alt_pos[1], alt_pos[0]] == 0 and
                        alt_pos != old_pos):  # Не оставаться на месте
                    possible_actions.append(alt_action)

            # Если есть возможные действия, выбираем случайное
            if possible_actions:
                alt_action = random.choice(possible_actions)
                if alt_action == 0:  # вверх
                    self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
                elif alt_action == 1:  # вправо
                    self.agent_pos[0] = min(self.maze_size - 1, self.agent_pos[0] + 1)
                elif alt_action == 2:  # вниз
                    self.agent_pos[1] = min(self.maze_size - 1, self.agent_pos[1] + 1)
                elif alt_action == 3:  # влево
                    self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
                moved = True
                print(f"🔄 Автоматическое избегание стены: действие {action} -> {alt_action}")

        current_pos_tuple = tuple(self.agent_pos)

        # Отслеживаем посещенные клетки для статистики
        if current_pos_tuple not in self.visited_cells:
            self.visited_cells.add(current_pos_tuple)

        terminated = False
        truncated = False

        # Проверка достижения цели
        if self.agent_pos == self.target_pos:
            terminated = True
            print(f"🎉 ЦЕЛЬ ДОСТИГНУТА! Шагов: {self.current_step}")

        # Проверка превышения лимита шагов
        if self.current_step >= self.max_steps:
            truncated = True
            print(f"⏰ ЛИМИТ ШАГОВ ИСЧЕРПАН: {self.current_step}")

        info = {
            "steps": self.current_step,
            "time_remaining": self.time_remaining,
            "visited_cells": len(self.visited_cells),
            "moved": moved,
            "hit_wall": hit_wall,
            "position": self.agent_pos.copy()
        }

        observation = self._get_observation()

        # Возвращаем нулевую награду и observation
        return observation, 0, terminated, truncated, info

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None