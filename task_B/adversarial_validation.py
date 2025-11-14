#!/usr/bin/env python3
"""
Adversarial Validation: check if train and test distributions are similar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ADVERSARIAL VALIDATION")
print("=" * 70)
print("\nЦель: Проверить, насколько train похож на test")
print("Если AUC ~0.5 → похожи (хорошо)")
print("Если AUC >0.7 → сильно отличаются (проблема)")

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"\nTrain: {len(train_df)} samples")
print(f"Test:  {len(test_df)} samples")

# Preprocessing
def preprocess(df):
    df = df.copy()
    # Encode categorical
    df['C'] = (df['C'] == '+').astype(int)
    # Fill missing
    df['E'] = df['E'].fillna(df['E'].median())
    return df

X_train = preprocess(train_df.drop('target', axis=1))
X_test = preprocess(test_df)

# Create adversarial dataset
X_train['is_test'] = 0
X_test['is_test'] = 1

X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_combined = X_combined['is_test']
X_combined = X_combined.drop('is_test', axis=1)

print(f"\nCombined dataset: {len(X_combined)} samples")
print(f"  Train (is_test=0): {(y_combined == 0).sum()}")
print(f"  Test  (is_test=1): {(y_combined == 1).sum()}")

# Train adversarial model
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ ADVERSARIAL CLASSIFIER")
print("=" * 70)

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_combined, y_combined)):
    X_tr = X_combined.iloc[train_idx]
    y_tr = y_combined.iloc[train_idx]
    X_val = X_combined.iloc[val_idx]
    y_val = y_combined.iloc[val_idx]

    model.fit(X_tr, y_tr)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    cv_scores.append(auc)
    print(f"Fold {fold+1}: AUC = {auc:.4f}")

mean_auc = np.mean(cv_scores)
std_auc = np.std(cv_scores)

print(f"\nMean CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

print("\n" + "=" * 70)
print("ИНТЕРПРЕТАЦИЯ")
print("=" * 70)

if mean_auc < 0.55:
    print("✅ ОТЛИЧНО: Train и Test очень похожи")
    print("   Распределения практически идентичны")
    print("   CV должен точно предсказать Test")
elif mean_auc < 0.65:
    print("⚠️  ХОРОШО: Есть небольшие различия")
    print("   Распределения немного отличаются")
    print("   CV в целом надежен, но могут быть расхождения")
elif mean_auc < 0.75:
    print("❌ ПРОБЛЕМА: Заметные различия")
    print("   Train и Test из разных распределений")
    print("   CV может плохо предсказать Test")
else:
    print("❌❌ СЕРЬЕЗНАЯ ПРОБЛЕМА: Сильное различие")
    print("   Train и Test очень разные")
    print("   Нужны специальные техники (domain adaptation)")

# Feature importance
print("\n" + "=" * 70)
print("КАКИЕ ПРИЗНАКИ РАЗЛИЧАЮТСЯ БОЛЬШЕ ВСЕГО?")
print("=" * 70)

model.fit(X_combined, y_combined)
importance = pd.DataFrame({
    'feature': X_combined.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop признаки, по которым train отличается от test:")
for idx, row in importance.head(10).iterrows():
    print(f"  {row['feature']:15s}: {row['importance']:.4f}")

# Statistical comparison
print("\n" + "=" * 70)
print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ ПРИЗНАКОВ")
print("=" * 70)

X_train_clean = X_train.drop('is_test', axis=1)
X_test_clean = X_test.drop('is_test', axis=1)

print(f"\n{'Feature':<15} {'Train Mean':<12} {'Test Mean':<12} {'Diff %':<10}")
print("-" * 55)

for col in X_train_clean.columns:
    train_mean = X_train_clean[col].mean()
    test_mean = X_test_clean[col].mean()
    diff_pct = 100 * (test_mean - train_mean) / (abs(train_mean) + 1e-10)

    marker = ""
    if abs(diff_pct) > 20:
        marker = " ⚠️"
    elif abs(diff_pct) > 50:
        marker = " ❌"

    print(f"{col:<15} {train_mean:<12.4f} {test_mean:<12.4f} {diff_pct:<+10.1f}{marker}")

print("\n" + "=" * 70)
print("РЕКОМЕНДАЦИИ")
print("=" * 70)

if mean_auc < 0.55:
    print("""
✅ Ваши данные в хорошем состоянии!
   - Train и Test похожи
   - Можно сфокусироваться на улучшении модели
   - CV надежно предсказывает Test

Следующие шаги:
1. Оптимизация весов ансамбля
2. Продвинутая feature engineering
3. Стекинг с мета-моделью
""")
elif mean_auc < 0.70:
    print("""
⚠️  Есть небольшие различия в распределениях

Рекомендуемые действия:
1. Использовать более сильную регуляризацию
2. Фокусироваться на признаках с наименьшей важностью в adversarial модели
3. Рассмотреть sample weighting по adversarial score
""")
else:
    print("""
❌ Существенное различие между train и test

Необходимые действия:
1. Sample re-weighting: давать больший вес train примерам, похожим на test
2. Удалить "outlier" train примеры
3. Domain adaptation техники
4. Использовать adversarial score как признак
""")

print("\n" + "=" * 70)
