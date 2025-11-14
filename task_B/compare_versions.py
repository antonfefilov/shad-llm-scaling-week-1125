#!/usr/bin/env python3
"""
Compare overfitting across different versions
"""

# Результаты из логов
versions = {
    'v3 (Weak Reg)': {
        'cv_auc': 0.8677,
        'train_auc': 1.0000,
        'test_score': 84  # баллы
    },
    'v4 (Feature Selection)': {
        'cv_auc': 0.8692,
        'train_auc': 0.9668,
        'test_score': 74  # хуже - удалили важные признаки
    },
    'v5 (Strong Reg)': {
        'cv_auc': 0.8809,
        'train_auc': 0.9866,
        'test_score': None  # еще не проверили
    }
}

print("=" * 70)
print("OVERFITTING ANALYSIS")
print("=" * 70)

for version, stats in versions.items():
    cv = stats['cv_auc']
    train = stats['train_auc']
    gap = train - cv
    test = stats['test_score']

    print(f"\n{version}")
    print(f"  Cross-Val AUC:   {cv:.4f}")
    print(f"  Train AUC:       {train:.4f}")
    print(f"  Overfit Gap:     {gap:.4f}  {'⚠️ HIGH' if gap > 0.12 else '✅ OK'}")

    if test:
        expected_from_cv = 100 * max(min((cv - 0.8) / 0.08, 1), 0)
        print(f"  Expected (CV):   {expected_from_cv:.1f} points")
        print(f"  Actual (Test):   {test} points")
        print(f"  Generalization:  {test - expected_from_cv:+.1f} points {'😞' if test < expected_from_cv else '😊'}")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. v3: Train=1.000, Gap=0.132 → ПЕРЕОБУЧЕНИЕ
   - Модель выучила тренировочные данные наизусть
   - CV говорит 84 балла, получили 84 (хорошо совпало)

2. v4: Train=0.967, Gap=0.098 → меньше overfitting, НО
   - Feature selection удалил важные признаки
   - CV говорит 86 баллов, получили 74 (плохо!)

3. v5: Train=0.987, Gap=0.106 → баланс
   - Контролируемое обучение
   - CV говорит 100 баллов → ожидаем ~90-95 на test
   - Все признаки сохранены
""")

print("\n" + "=" * 70)
print("РЕГУЛЯРИЗАЦИЯ В v5")
print("=" * 70)
print("""
Параметр              v3 (слабая)    v5 (сильная)    Эффект
─────────────────────────────────────────────────────────────
max_depth             6              4               ↓ Менее сложные деревья
learning_rate         0.03           0.03            ✓ Медленное обучение
min_child_weight      3              10              ↑ Больше данных для решения
min_child_samples     20             30              ↑ Надежнее правила
reg_alpha (L1)        0.5            2.0             ↑ Убирает шумные признаки
reg_lambda (L2)       2.0            5.0             ↑ Уменьшает веса
subsample             0.8            0.7             ↓ Меньше данных на итерацию
colsample_bytree      0.8            0.7             ↓ Меньше признаков на дерево

Результат: Train↓ (0.99 вместо 1.00), CV↑ (0.88 вместо 0.87)
""")
