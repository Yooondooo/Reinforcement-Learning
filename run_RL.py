import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(__file__))

from maze_env import MazeEnv
from dqn_agent import DQNAgent


def test_trained_agent(model_path="maze_levels_model", maze_size=12, difficulty=5, num_tests=10):
    """
    Тестирование обученного агента на новых лабиринтах
    """
    # Инициализация окружения
    env = MazeEnv(maze_size=maze_size, render_mode="human", max_steps=200, difficulty=difficulty)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Инициализация агента
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0  # Отключаем исследование при тестировании
    agent.epsilon_min = 0.0

    # Загрузка обученных весов
    try:
        agent.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {model_path}")
        print(f"   Ошибка: {e}")
        env.close()
        return

    # Статистика
    results = {
        'success_rate': 0,
        'avg_steps': 0,
        'min_steps': float('inf'),
        'max_steps': 0,
        'successful_runs': []
    }

    print(f"\n🧪 НАЧАЛО ТЕСТИРОВАНИЯ")
    print(f"📐 Размер лабиринта: {maze_size}x{maze_size}")
    print(f"🎯 Сложность: {difficulty}")
    print(f"🔢 Количество тестов: {num_tests}")
    print(f"🎮 Режим: только эксплуатация (epsilon=0.0)")

    for test_num in range(1, num_tests + 1):
        print(f"\n--- Тест {test_num}/{num_tests} ---")

        # Создаем новый лабиринт
        observation, _ = env.reset()
        state = np.reshape(observation, [1, state_size])

        step_count = 0
        terminated = False
        truncated = False

        # Запускаем эпизод без обучения
        for step in range(env.max_steps):
            # Визуализация
            env.render_with_info(
                episode=test_num,
                step=step_count + 1,
                reward=0,  # Награды не отображаем
                epsilon=0.0,
                level=difficulty
            )

            # Выбор действия (только эксплуатация)
            action = agent.act(state)

            # Выполняем шаг в окружении (используем тестовый метод)
            next_state, _, terminated, truncated, info = env.test_step(action)
            next_state = np.reshape(next_state, [1, state_size])

            state = next_state
            step_count += 1

            if terminated or truncated:
                break

        # Анализ результатов теста
        if terminated:
            results['success_rate'] += 1
            results['successful_runs'].append(step_count)
            results['avg_steps'] += step_count
            results['min_steps'] = min(results['min_steps'], step_count)
            results['max_steps'] = max(results['max_steps'], step_count)

            print(f"✅ УСПЕХ: пройден за {step_count} шагов")
        else:
            print(f"❌ НЕУДАЧА: не пройден за {step_count} шагов")
            print(f"   Позиция агента: {env.agent_pos}, Цель: {env.target_pos}")

        # Небольшая пауза между тестами для лучшей визуализации
        time.sleep(0.5)

    # Вычисляем средние значения
    if results['success_rate'] > 0:
        results['avg_steps'] /= results['success_rate']

    results['success_rate'] = (results['success_rate'] / num_tests) * 100

    # Вывод итоговой статистики
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"✅ Процент успеха: {results['success_rate']:.1f}%")

    if results['successful_runs']:
        print(f"📈 Статистика по успешным прохождениям:")
        print(f"   Среднее количество шагов: {results['avg_steps']:.1f}")
        print(f"   Минимальное количество шагов: {results['min_steps']}")
        print(f"   Максимальное количество шагов: {results['max_steps']}")
        print(f"   Все успешные прохождения: {results['successful_runs']}")
    else:
        print(f"😞 Не было успешных прохождений")

    # Визуализация результатов
    plot_test_results(results, num_tests, difficulty)

    env.close()
    return results


def plot_test_results(results, num_tests, difficulty):
    """
    Визуализация результатов тестирования
    """
    plt.figure(figsize=(12, 4))

    # График 1: Процент успеха
    plt.subplot(1, 3, 1)
    labels = ['Успех', 'Неудача']
    sizes = [results['success_rate'], 100 - results['success_rate']]
    colors = ['#4CAF50', '#F44336']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'Процент успеха\n(Сложность: {difficulty})')

    # График 2: Распределение шагов в успешных прохождениях
    plt.subplot(1, 3, 2)
    if results['successful_runs']:
        plt.hist(results['successful_runs'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(results['avg_steps'], color='red', linestyle='--', label=f'Среднее: {results["avg_steps"]:.1f}')
        plt.xlabel('Количество шагов')
        plt.ylabel('Частота')
        plt.title('Распределение шагов\nв успешных прохождениях')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет успешных\nпрохождений',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('Распределение шагов')

    # График 3: Сводная статистика
    plt.subplot(1, 3, 3)
    if results['successful_runs']:
        stats_data = [results['min_steps'], results['avg_steps'], results['max_steps']]
        stats_labels = ['Мин.', 'Сред.', 'Макс.']

        bars = plt.bar(stats_labels, stats_data, color=['#FF9800', '#2196F3', '#E91E63'])
        plt.ylabel('Количество шагов')
        plt.title('Сводная статистика шагов')

        # Добавляем значения на столбцы
        for bar, value in zip(bars, stats_data):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{value:.1f}', ha='center', va='bottom')

        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет данных\nдля статистики',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('Сводная статистика')

    plt.tight_layout()
    plt.savefig(f'test_results_difficulty_{difficulty}.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_multiple_difficulties(model_path="maze_levels_model", maze_size=12, tests_per_difficulty=5):
    """
    Тестирование на разных уровнях сложности
    """
    difficulties = [1, 2, 3, 4, 5]
    all_results = {}

    print(f"🎯 ТЕСТИРОВАНИЕ НА РАЗНЫХ УРОВНЯХ СЛОЖНОСТИ")
    print(f"📐 Размер лабиринта: {maze_size}x{maze_size}")
    print(f"🔢 Тестов на уровень: {tests_per_difficulty}")

    for difficulty in difficulties:
        print(f"\n{'=' * 50}")
        print(f"🧪 Тестирование на сложности {difficulty}")
        print(f"{'=' * 50}")

        results = test_trained_agent(
            model_path=model_path,
            maze_size=maze_size,
            difficulty=difficulty,
            num_tests=tests_per_difficulty
        )

        all_results[difficulty] = results

        # Пауза между уровнями сложности
        time.sleep(2)

    # Сводный отчет по всем уровням сложности
    print(f"\n{'=' * 60}")
    print(f"📊 СВОДНЫЙ ОТЧЕТ ПО ВСЕМ УРОВНЯМ СЛОЖНОСТИ")
    print(f"{'=' * 60}")

    plt.figure(figsize=(10, 6))

    success_rates = []
    avg_steps = []

    for difficulty in difficulties:
        results = all_results[difficulty]
        success_rates.append(results['success_rate'])

        if results['successful_runs']:
            avg_steps.append(results['avg_steps'])
        else:
            avg_steps.append(0)

        print(f"Уровень {difficulty}: {results['success_rate']:.1f}% успеха, "
              f"средние шаги: {avg_steps[-1]:.1f}")

    # График успеваемости по уровням сложности
    plt.subplot(1, 2, 1)
    plt.plot(difficulties, success_rates, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Уровень сложности')
    plt.ylabel('Процент успеха (%)')
    plt.title('Успеваемость по уровням сложности')
    plt.grid(True, alpha=0.3)
    plt.xticks(difficulties)

    # График средних шагов по уровням сложности
    plt.subplot(1, 2, 2)
    plt.plot(difficulties, avg_steps, 'o-', linewidth=2, markersize=8, color='red')
    plt.xlabel('Уровень сложности')
    plt.ylabel('Среднее количество шагов')
    plt.title('Сложность прохождения по уровням')
    plt.grid(True, alpha=0.3)
    plt.xticks(difficulties)

    plt.tight_layout()
    plt.savefig('comprehensive_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_results


if __name__ == "__main__":
    # Тестирование одной модели
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ОБУЧЕННОЙ МОДЕЛИ")

    # Укажите путь к вашей обученной модели
    MODEL_PATH = "maze_levels_model"  # или "maze_levels_model.weights.h5"

    # Вариант 1: Тестирование на одном уровне сложности
    test_trained_agent(
        model_path=MODEL_PATH,
        maze_size=12,
        difficulty=5,
        num_tests=10
    )

    # Вариант 2: Тестирование на всех уровнях сложности (раскомментируйте если нужно)
    test_multiple_difficulties(
        model_path=MODEL_PATH,
        maze_size=12,
        tests_per_difficulty=5
    )