import numpy as np
from maze_env import MazeEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import time


def train_with_levels():
    # Настройки обучения
    maze_size = 16
    max_steps = 300
    successes_required = 1
    max_attempts_per_maze = 150  # Максимальное количество попыток на один лабиринт

    current_level = 1
    max_level = 15

    env = MazeEnv(maze_size=maze_size, render_mode="human", max_steps=max_steps, difficulty=current_level)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    # Увеличиваем exploration
    agent.epsilon = 1.0
    agent.epsilon_min = 0.8
    agent.epsilon_decay = 0.98

    print(f"🚀 Обучение с системой уровней")
    print(f"📐 Лабиринт: {maze_size}x{maze_size}")
    print(f"🎯 Начальный уровень: {current_level}")
    print(f"🏆 Максимальный уровень: {max_level}")
    print(f"✅ Требуется успешных прохождений подряд: {successes_required}")
    print(f"🔄 Максимум попыток на лабиринт: {max_attempts_per_maze}")
    print(f"🔁 Один и тот же лабиринт до успешного прохождения!")
    print(f"🎮 Epsilon: начинается с 1.0, уменьшается после каждого эпизода")

    # Статистика
    all_scores = []
    all_steps = []
    level_successes = {level: 0 for level in range(1, max_level + 1)}
    level_attempts = {level: 0 for level in range(1, max_level + 1)}
    maze_generations = {level: 0 for level in range(1, max_level + 1)}

    start_time = time.time()
    total_episodes = 0

    print(f"\n=== НАЧАЛО ОБУЧЕНИЯ ===")
    print(f"🎮 Уровень {current_level}")

    try:
        while current_level <= max_level:
            # Генерируем ОДИН лабиринт для текущего уровня
            maze_generations[current_level] += 1
            print(f"\n🔄 Генерация лабиринта #{maze_generations[current_level]} для уровня {current_level}...")
            env.difficulty = current_level
            initial_observation, _ = env.reset()

            # Сохраняем начальное состояние лабиринта
            maze_info = {
                'observation': initial_observation.copy(),
                'agent_pos': env.agent_pos.copy(),
                'target_pos': env.target_pos.copy(),
                'maze': env.maze.copy(),
                'scenario': getattr(env, 'scenario', 'start_to_end')
            }

            successes_this_maze = 0
            attempts_this_maze = 0
            consecutive_successes = 0

            # Обучаем на ОДНОМ И ТОМ ЖЕ лабиринте
            while consecutive_successes < successes_required and attempts_this_maze < max_attempts_per_maze:
                total_episodes += 1
                attempts_this_maze += 1
                level_attempts[current_level] += 1

                # ВОССТАНАВЛИВАЕМ начальное состояние лабиринта
                env.agent_pos = maze_info['agent_pos'].copy()
                env.target_pos = maze_info['target_pos'].copy()
                env.maze = maze_info['maze'].copy()
                env.current_step = 0
                env.time_remaining = env.max_steps

                state = maze_info['observation'].copy()
                state = np.reshape(state, [1, state_size])
                total_reward = 0
                step_count = 0
                terminated = False
                truncated = False

                # Запускаем эпизод
                for time_step in range(env.max_steps):
                    # Визуализация с информацией об уровне
                    env.render_with_info(
                        episode=total_episodes,
                        step=step_count + 1,
                        reward=total_reward,
                        epsilon=agent.epsilon,
                        level=current_level
                    )

                    # Выбор действия
                    action = agent.act(state, env.agent_pos)

                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])

                    agent.remember(state, action, reward, next_state, terminated or truncated)
                    state = next_state

                    total_reward += reward
                    step_count += 1

                    # Обучение с интервалом
                    agent.train_counter += 1
                    if len(agent.memory) > agent.batch_size and agent.train_counter % agent.train_interval == 0:
                        agent.replay()

                    if terminated or truncated:
                        break

                all_scores.append(total_reward)
                all_steps.append(step_count)

                if terminated:
                    successes_this_maze += 1
                    level_successes[current_level] += 1
                    consecutive_successes += 1

                    print(f"✅ Уровень {current_level}: Успех #{consecutive_successes}/{successes_required} "
                          f"(Попытка {attempts_this_maze}/{max_attempts_per_maze}, "
                          f"Шаги: {step_count}, Награда: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f})")

                    if consecutive_successes >= successes_required:
                        break

                else:
                    consecutive_successes = 0
                    if attempts_this_maze % 5 == 0 or attempts_this_maze == max_attempts_per_maze:
                        print(
                            f"❌ Уровень {current_level}: Неудача (Попытка {attempts_this_maze}/{max_attempts_per_maze}, "
                            f"Шаги: {step_count}, Награда: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f})")

                # Обновление в конце эпизода
                agent.end_episode()

                # Обновление целевой сети
                if total_episodes % 20 == 0:
                    agent.update_target_model()

            # Анализ результатов для этого лабиринта
            if consecutive_successes >= successes_required:
                # УСПЕХ: переходим на следующий уровень
                if current_level < max_level:
                    current_level += 1
                    success_rate = (level_successes[current_level - 1] / level_attempts[current_level - 1]) * 100
                    print(f"\n🎉 ПЕРЕХОД НА УРОВЕНЬ {current_level}!")
                    print(f"📊 Статистика уровня {current_level - 1}:")
                    print(f"   Успешных прохождений: {level_successes[current_level - 1]}")
                    print(f"   Всего попыток: {level_attempts[current_level - 1]}")
                    print(f"   Сгенерировано лабиринтов: {maze_generations[current_level - 1]}")
                    print(f"   Процент успеха: {success_rate:.1f}%")
                    print(f"   Текущий epsilon: {agent.epsilon:.3f}")
                    print(f"🎮 Начинаем уровень {current_level}")
                else:
                    print(f"\n🏆 ВЫ ПРОШЛИ ВСЕ УРОВНИ!")
                    break
            else:
                # НЕУДАЧА: слишком много попыток на этом лабиринте
                print(f"\n🔄 Лабиринт уровня {current_level} слишком сложен после {attempts_this_maze} попыток")
                print(f"   Успешных прохождений на этом лабиринте: {successes_this_maze}")
                print(
                    f"   Лучший результат: {max(all_scores[-attempts_this_maze:]) if attempts_this_maze > 0 else 0:.1f}")
                print(f"   Генерируем новый лабиринт...")

                # Не переходим на следующий уровень, просто генерируем новый лабиринт
                if current_level >= max_level:
                    print(f"\n🏁 Достигнут максимальный уровень {max_level}")
                    break

        # Финальная статистика
        total_time = time.time() - start_time

        print(f"\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
        print(f"🏁 Достигнутый уровень: {current_level}")
        print(f"📊 Всего эпизодов: {total_episodes}")
        print(f"⏱️ Общее время: {total_time:.1f} сек")
        print(f"⚡ Средняя скорость: {total_episodes / total_time:.1f} эпизодов/сек")
        print(f"🎯 Финальный epsilon: {agent.epsilon:.3f}")

        # Статистика по уровням
        print(f"\n📈 СТАТИСТИКА ПО УРОВНЯМ:")
        for level in range(1, min(current_level + 1, max_level + 1)):
            if level_attempts[level] > 0:
                success_rate = (level_successes[level] / level_attempts[level]) * 100
                print(f"   Уровень {level}:")
                print(f"     Успешных прохождений: {level_successes[level]}")
                print(f"     Всего попыток: {level_attempts[level]}")
                print(f"     Сгенерировано лабиринтов: {maze_generations[level]}")
                print(f"     Процент успеха: {success_rate:.1f}%")
                if maze_generations[level] > 0:
                    attempts_per_maze = level_attempts[level] / maze_generations[level]
                    print(f"     Среднее попыток на лабиринт: {attempts_per_maze:.1f}")

    except KeyboardInterrupt:
        print(f"\nОбучение прервано на уровне {current_level}")
        print(f"Текущий epsilon: {agent.epsilon:.3f}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent.save(f"maze_levels_model")
        print(f"Модель сохранена: maze_levels_model")

        if all_scores:  # Проверяем что есть данные для графиков
            plot_training_results(all_scores, all_steps, total_episodes, current_level)
        env.close()


def plot_training_results(scores, steps, episodes, max_level):
    plt.figure(figsize=(15, 5))

    # График 1: Награды по эпизодам
    plt.subplot(1, 3, 1)
    if scores:  # Проверяем что массив не пустой
        plt.plot(scores, alpha=0.7, linewidth=1)
        plt.title(f'Награды по эпизодам (Уровень {max_level})')
        plt.xlabel('Эпизод')
        plt.ylabel('Награда')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет данных',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('Награды по эпизодам')

    # График 2: Шаги по эпизодам
    plt.subplot(1, 3, 2)
    if steps:
        plt.plot(steps, alpha=0.7, linewidth=1, color='green')
        plt.axhline(y=100, color='r', linestyle='--', label='Лимит шагов')
        plt.title(f'Шаги по эпизодам (Уровень {max_level})')
        plt.xlabel('Эпизод')
        plt.ylabel('Шаги')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет данных',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('Шаги по эпизодам')

    # График 3: Скользящее среднее наград
    plt.subplot(1, 3, 3)
    if scores:
        window = min(20, len(scores) // 4)
        if len(scores) >= window:
            scores_smooth = np.convolve(scores, np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(scores)), scores_smooth, 'r-', linewidth=2, label=f'Среднее ({window} эп.)')
        plt.plot(scores, alpha=0.3, color='blue', label='Сырые данные')
        plt.title('Прогресс обучения')
        plt.xlabel('Эпизод')
        plt.ylabel('Награда')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет данных',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('Прогресс обучения')

    plt.tight_layout()
    if scores:
        plt.savefig(f'training_results_levels_{max_level}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train_with_levels()