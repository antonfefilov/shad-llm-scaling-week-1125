import sys
import numpy as np

def solve():
    # Чтение входных данных
    data = sys.stdin.read()
    lines = data.strip().split('\n')
    n, m = map(int, lines[0].split())
    sx, sy = map(int, lines[1].split())

    # Собираем позиции каждого типа адреса
    type_positions = {}
    for i in range(n):
        row = lines[2 + i]
        for j, char in enumerate(row):
            if char not in type_positions:
                type_positions[char] = []
            type_positions[char].append((i, j))

    # Преобразуем в numpy массивы
    for key in type_positions:
        type_positions[key] = np.array(type_positions[key], dtype=np.int16)

    s = lines[2 + n]

    # Текущие позиции и времена
    current_pos = np.array([[sx - 1, sy - 1]], dtype=np.int16)
    current_times = np.array([0], dtype=np.int32)

    MAX_POS = 210  # Увеличиваем для корректности

    # Обрабатываем каждую доставку
    for char in s:
        if char not in type_positions:
            continue

        targets = type_positions[char]
        n_targets = len(targets)

        # Фильтрация: берем только лучшие позиции
        if len(current_times) > MAX_POS:
            idx = np.argpartition(current_times, MAX_POS)[:MAX_POS]
            current_pos = current_pos[idx]
            current_times = current_times[idx]

        n_current = len(current_times)

        # Оптимизация: выбираем метод в зависимости от размера
        if n_current * n_targets < 40000:  # Уменьшаем порог для ускорения
            # Матричный метод для малых размеров
            diff = np.abs(current_pos[:, np.newaxis, :] - targets[np.newaxis, :, :])
            distances = diff[:, :, 0] + diff[:, :, 1]
            times = current_times[:, np.newaxis] + distances
            min_times = times.min(axis=0).astype(np.int32)
        else:
            # Поэлементный метод для больших размеров (меньше памяти)
            min_times = np.empty(n_targets, dtype=np.int32)
            cx = current_pos[:, 0]
            cy = current_pos[:, 1]
            for j in range(n_targets):
                tx, ty = targets[j]
                dists = np.abs(cx - tx) + np.abs(cy - ty)
                min_times[j] = (current_times + dists).min()

        # Обновляем состояние
        current_pos = targets
        current_times = min_times

    print(int(current_times.min()))

if __name__ == "__main__":
    solve()