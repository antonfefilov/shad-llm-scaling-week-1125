#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    int sx, sy;
    cin >> sx >> sy;
    sx--; sy--;  // 0-индексация

    cin.ignore();

    // Читаем сетку
    vector<string> grid(n);
    for (int i = 0; i < n; i++) {
        getline(cin, grid[i]);
    }

    // Читаем последовательность доставок
    string deliveries;
    getline(cin, deliveries);

    // Собираем позиции для каждого типа адреса
    unordered_map<char, vector<pair<int, int>>> positions;
    positions.reserve(26);  // Оптимизация для латинских букв

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            positions[grid[i][j]].push_back({i, j});
        }
    }

    // DP: текущие позиции и минимальные времена
    vector<pair<int, int>> curr_pos;
    vector<int> curr_time;
    curr_pos.reserve(n * m);
    curr_time.reserve(n * m);

    curr_pos.push_back({sx, sy});
    curr_time.push_back(0);

    for (char delivery_type : deliveries) {
        auto it = positions.find(delivery_type);
        if (it == positions.end()) {
            continue;
        }

        const auto& targets = it->second;
        int n_targets = targets.size();
        int n_current = curr_pos.size();

        // Вычисляем минимальные времена для каждой целевой позиции
        vector<int> next_time(n_targets, numeric_limits<int>::max());

        // ОПТИМИЗАЦИЯ: меняем порядок циклов для лучшей локальности кэша
        // Внешний цикл - по текущим позициям (обычно их меньше)
        for (int i = 0; i < n_current; i++) {
            int cx = curr_pos[i].first;
            int cy = curr_pos[i].second;
            int ct = curr_time[i];

            // Внутренний цикл - по целевым позициям
            for (int j = 0; j < n_targets; j++) {
                int tx = targets[j].first;
                int ty = targets[j].second;

                // Manhattan distance - inline вычисление
                int dist = abs(tx - cx) + abs(ty - cy);
                int new_time = ct + dist;

                // ОПТИМИЗАЦИЯ: прямое сравнение без min()
                if (new_time < next_time[j]) {
                    next_time[j] = new_time;
                }
            }
        }

        // Обновляем состояние
        curr_pos = targets;
        curr_time = std::move(next_time);
    }

    // Ответ - минимальное время
    cout << *min_element(curr_time.begin(), curr_time.end()) << '\n';

    return 0;
}
