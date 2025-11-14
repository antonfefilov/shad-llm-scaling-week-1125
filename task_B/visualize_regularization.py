#!/usr/bin/env python3
"""
Visualize the effect of regularization on actual data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('train.csv')
X = train_df.drop('target', axis=1).copy()
y = train_df['target'].copy()

# Simple preprocessing
X['C'] = (X['C'] == '+').astype(int)
X['E'] = X['E'].fillna(X['E'].median())

n_pos = y.sum()
n_neg = len(y) - n_pos
scale = n_neg / n_pos

print("=" * 70)
print("REGULARIZATION IMPACT ON YOUR DATA")
print("=" * 70)
print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Class balance: {n_neg} negative, {n_pos} positive (ratio: {scale:.2f})")

# Test different regularization levels
configs = [
    {
        'name': 'NO Regularization (overfits)',
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'min_child_samples': 5,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }
    },
    {
        'name': 'WEAK Regularization',
        'params': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    },
    {
        'name': 'STRONG Regularization (v5)',
        'params': {
            'n_estimators': 400,
            'max_depth': 4,
            'learning_rate': 0.03,
            'min_child_samples': 30,
            'reg_alpha': 2.0,
            'reg_lambda': 5.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7
        }
    }
]

print("\n" + "=" * 70)
print("TESTING DIFFERENT REGULARIZATION LEVELS")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for config in configs:
    print(f"\n{config['name']}")
    print("-" * 70)

    params = config['params'].copy()
    params['scale_pos_weight'] = scale
    params['random_state'] = 42
    params['verbose'] = -1
    params['metric'] = 'auc'

    model = lgb.LGBMClassifier(**params)

    # Cross-validation
    cv_scores = []
    train_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)

        train_pred = model.predict_proba(X_train_fold)[:, 1]
        val_pred = model.predict_proba(X_val_fold)[:, 1]

        train_auc = roc_auc_score(y_train_fold, train_pred)
        val_auc = roc_auc_score(y_val_fold, val_pred)

        train_scores.append(train_auc)
        cv_scores.append(val_auc)

    train_mean = np.mean(train_scores)
    cv_mean = np.mean(cv_scores)
    gap = train_mean - cv_mean

    print(f"  Train AUC:     {train_mean:.4f}")
    print(f"  CV AUC:        {cv_mean:.4f}")
    print(f"  Overfit Gap:   {gap:.4f} {'⚠️  HIGH!' if gap > 0.12 else '✅ Good'}")

    results.append({
        'name': config['name'],
        'train': train_mean,
        'cv': cv_mean,
        'gap': gap
    })

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for r in results:
    expected_test = 100 * max(min((r['cv'] - 0.8) / 0.08, 1), 0)
    print(f"\n{r['name']}")
    print(f"  Train AUC:       {r['train']:.4f}")
    print(f"  CV AUC:          {r['cv']:.4f}")
    print(f"  Gap:             {r['gap']:.4f}")
    print(f"  Expected Score:  ~{expected_test:.1f} points")
    print(f"  Status:          {'✅ Good generalization' if r['gap'] < 0.12 else '⚠️  Overfitting'}")

print("\n" + "=" * 70)
print("WHAT DOES IT MEAN?")
print("=" * 70)
print("""
Train AUC = 1.000 (идеально):
  ❌ Модель ЗАПОМНИЛА тренировочные данные
  ❌ Она выучила шум и случайные паттерны
  ❌ На новых данных работает хуже

Train AUC = 0.987 (хорошо, не идеально):
  ✅ Модель ПОНЯЛА общие закономерности
  ✅ Она не запоминает шум
  ✅ На новых данных работает лучше

Overfit Gap (Train - CV):
  < 0.10: Отлично! Модель хорошо обобщает
  0.10-0.12: Хорошо, приемлемый уровень
  > 0.12: Проблема, модель переобучена

Регуляризация СПЕЦИАЛЬНО ухудшает Train AUC,
чтобы улучшить обобщающую способность!
""")
