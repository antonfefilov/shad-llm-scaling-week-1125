#!/usr/bin/env python3
"""
Classifier v7: Stacking Meta-Learner
Uses out-of-fold predictions as features for a meta-model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CLASSIFIER V7: STACKING META-LEARNER")
print("=" * 70)
print("\nИдея: Использовать OOF предсказания как признаки для мета-модели")

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X_full = train_df.drop('target', axis=1).copy()
y_full = train_df['target'].copy()
X_test_full = test_df.copy()

def create_conservative_features(df, is_train=True):
    df = df.copy()
    df['E_missing'] = df['E'].isnull().astype(int)
    df['E'] = df['E'].fillna(df['E'].median() if is_train else 2.0)
    df['C'] = (df['C'] == '+').astype(int)

    df['A_E'] = df['A'] * df['E']
    df['A_G'] = df['A'] * df['G']
    df['A_H'] = df['A'] * df['H']
    df['G_H'] = df['G'] * df['H']

    df['A_squared'] = df['A'] ** 2
    df['G_squared'] = df['G'] ** 2
    df['H_squared'] = df['H'] ** 2

    df['log_A'] = np.log1p(df['A'])
    df['log_G'] = np.log1p(df['G'] + 10)

    df['A_D_ratio'] = df['A'] / (df['D'] + 0.001)
    df['G_I_ratio'] = df['G'] / (np.abs(df['I']) + 1)

    df['sum_GHI'] = df['G'] + df['H'] + df['I']

    return df

X_train = create_conservative_features(X_full, is_train=True)
X_test = create_conservative_features(X_test_full, is_train=False)

n_pos = y_full.sum()
n_neg = len(y_full) - n_pos
scale_pos_weight = n_neg / n_pos

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")
print(f"Features: {X_train.shape[1]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LEVEL 1: Base models
print("\n" + "=" * 70)
print("LEVEL 1: TRAINING BASE MODELS")
print("=" * 70)

oof_predictions = np.zeros((len(X_train), 5))  # 5 models
test_predictions = np.zeros((len(X_test), 5))

model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest', 'GradientBoosting']

# Model 1: LightGBM
print("\n[1/5] LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    min_child_samples=30, reg_alpha=2.0, reg_lambda=5.0,
    subsample=0.7, colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
)

test_preds_lgb = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    lgb_model.fit(X_tr, y_tr)
    oof_predictions[val_idx, 0] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds_lgb.append(lgb_model.predict_proba(X_test)[:, 1])

test_predictions[:, 0] = np.mean(test_preds_lgb, axis=0)
lgb_oof_auc = roc_auc_score(y_full, oof_predictions[:, 0])
print(f"  OOF AUC: {lgb_oof_auc:.4f}")

# Model 2: XGBoost
print("[2/5] XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    gamma=0.5, reg_alpha=2.0, reg_lambda=5.0,
    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='auc'
)

test_preds_xgb = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    xgb_model.fit(X_tr, y_tr)
    oof_predictions[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds_xgb.append(xgb_model.predict_proba(X_test)[:, 1])

test_predictions[:, 1] = np.mean(test_preds_xgb, axis=0)
xgb_oof_auc = roc_auc_score(y_full, oof_predictions[:, 1])
print(f"  OOF AUC: {xgb_oof_auc:.4f}")

# Model 3: CatBoost
print("[3/5] CatBoost...")
cb_model = cb.CatBoostClassifier(
    iterations=400, depth=4, learning_rate=0.03,
    l2_leaf_reg=5.0, random_strength=2.0, bagging_temperature=0.7,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=0
)

test_preds_cb = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    cb_model.fit(X_tr, y_tr)
    oof_predictions[val_idx, 2] = cb_model.predict_proba(X_val)[:, 1]
    test_preds_cb.append(cb_model.predict_proba(X_test)[:, 1])

test_predictions[:, 2] = np.mean(test_preds_cb, axis=0)
cb_oof_auc = roc_auc_score(y_full, oof_predictions[:, 2])
print(f"  OOF AUC: {cb_oof_auc:.4f}")

# Model 4: RandomForest
print("[4/5] RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_split=30,
    min_samples_leaf=15, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)

test_preds_rf = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    rf_model.fit(X_tr, y_tr)
    oof_predictions[val_idx, 3] = rf_model.predict_proba(X_val)[:, 1]
    test_preds_rf.append(rf_model.predict_proba(X_test)[:, 1])

test_predictions[:, 3] = np.mean(test_preds_rf, axis=0)
rf_oof_auc = roc_auc_score(y_full, oof_predictions[:, 3])
print(f"  OOF AUC: {rf_oof_auc:.4f}")

# Model 5: GradientBoosting
print("[5/5] GradientBoosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.03,
    subsample=0.7, min_samples_split=30, min_samples_leaf=15,
    max_features='sqrt', random_state=42
)

test_preds_gb = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    gb_model.fit(X_tr, y_tr)
    oof_predictions[val_idx, 4] = gb_model.predict_proba(X_val)[:, 1]
    test_preds_gb.append(gb_model.predict_proba(X_test)[:, 1])

test_predictions[:, 4] = np.mean(test_preds_gb, axis=0)
gb_oof_auc = roc_auc_score(y_full, oof_predictions[:, 4])
print(f"  OOF AUC: {gb_oof_auc:.4f}")

# LEVEL 2: Meta-model
print("\n" + "=" * 70)
print("LEVEL 2: TRAINING META-MODEL")
print("=" * 70)

# Try different meta-models
meta_models = {
    'LogisticRegression': LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=1000, random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
    )
}

meta_results = {}

for meta_name, meta_model in meta_models.items():
    print(f"\n{meta_name}:")

    meta_oof = np.zeros(len(X_train))
    meta_test = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
        X_meta_tr = oof_predictions[train_idx]
        y_meta_tr = y_full.iloc[train_idx]
        X_meta_val = oof_predictions[val_idx]

        meta_model.fit(X_meta_tr, y_meta_tr)
        meta_oof[val_idx] = meta_model.predict_proba(X_meta_val)[:, 1]

        # Predict on test using Level 1 predictions
        meta_test.append(meta_model.predict_proba(test_predictions)[:, 1])

    meta_auc = roc_auc_score(y_full, meta_oof)
    meta_test_pred = np.mean(meta_test, axis=0)

    meta_results[meta_name] = {
        'oof_auc': meta_auc,
        'test_pred': meta_test_pred
    }

    expected_score = 100 * max(min((meta_auc - 0.8) / 0.08, 1), 0)
    print(f"  OOF AUC:       {meta_auc:.4f}")
    print(f"  Expected:      {expected_score:.1f} points")

# Baseline: simple average
baseline_oof = oof_predictions.mean(axis=1)
baseline_auc = roc_auc_score(y_full, baseline_oof)
baseline_test = test_predictions.mean(axis=1)

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"\nBaseline (Simple Average):")
print(f"  OOF AUC: {baseline_auc:.4f}")

best_meta_name = max(meta_results.keys(), key=lambda k: meta_results[k]['oof_auc'])
best_meta = meta_results[best_meta_name]

print(f"\nBest Meta-Model ({best_meta_name}):")
print(f"  OOF AUC: {best_meta['oof_auc']:.4f}")
print(f"  Improvement: {best_meta['oof_auc'] - baseline_auc:+.4f} AUC")

improvement_points = (best_meta['oof_auc'] - baseline_auc) * (100 / 0.08)
print(f"  Expected gain: {improvement_points:+.1f} points")

# Save predictions
print("\n" + "=" * 70)
print("SAVING PREDICTIONS")
print("=" * 70)

# Best meta-model
output_df = pd.DataFrame({'target': best_meta['test_pred']})
output_df.to_csv('answers_v7_stacking.csv', index=False)
print(f"✅ Best meta-model ({best_meta_name}): answers_v7_stacking.csv")

# Save all meta-models for comparison
for meta_name, meta_data in meta_results.items():
    filename = f'answers_v7_stacking_{meta_name.lower().replace(" ", "_")}.csv'
    output_df = pd.DataFrame({'target': meta_data['test_pred']})
    output_df.to_csv(filename, index=False)
    print(f"✅ {meta_name}: {filename}")

# Baseline for reference
output_df_baseline = pd.DataFrame({'target': baseline_test})
output_df_baseline.to_csv('answers_v7_baseline.csv', index=False)
print(f"✅ Baseline (simple avg): answers_v7_baseline.csv")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Level 1 (Base models):
  LightGBM:         {lgb_oof_auc:.4f}
  XGBoost:          {xgb_oof_auc:.4f}
  CatBoost:         {cb_oof_auc:.4f}
  RandomForest:     {rf_oof_auc:.4f}
  GradientBoosting: {gb_oof_auc:.4f}

Level 2 (Meta-models):
  Simple Average:   {baseline_auc:.4f}
  {best_meta_name}: {best_meta['oof_auc']:.4f} ⭐ BEST

Stacking использует OOF предсказания как признаки для мета-модели,
что позволяет лучше комбинировать предсказания базовых моделей.
""")

print("\n" + "=" * 70)
print("РЕКОМЕНДАЦИИ")
print("=" * 70)

print("""
1. Сначала протестируйте v5_optimized (Optuna веса)
2. Если v5_optimized < 100 баллов, протестируйте v7_stacking
3. v7_stacking может дать +1-3 балла благодаря мета-обучению
""")

print("=" * 70)
