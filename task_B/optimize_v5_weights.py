#!/usr/bin/env python3
"""
Optimize ensemble weights for v5 models using Optuna
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OPTIMIZE V5 ENSEMBLE WEIGHTS")
print("=" * 70)

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

# Collect OOF predictions
print("\n" + "=" * 70)
print("COLLECTING OUT-OF-FOLD PREDICTIONS")
print("=" * 70)

oof_preds = {
    'lgb': np.zeros(len(X_train)),
    'xgb': np.zeros(len(X_train)),
    'cb': np.zeros(len(X_train)),
    'rf': np.zeros(len(X_train)),
    'gb': np.zeros(len(X_train))
}

test_preds = {
    'lgb': [],
    'xgb': [],
    'cb': [],
    'rf': [],
    'gb': []
}

# LightGBM
print("\n[1/5] Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    min_child_samples=30, reg_alpha=2.0, reg_lambda=5.0,
    subsample=0.7, colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    lgb_model.fit(X_tr, y_tr)
    oof_preds['lgb'][val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds['lgb'].append(lgb_model.predict_proba(X_test)[:, 1])

lgb_oof_auc = roc_auc_score(y_full, oof_preds['lgb'])
print(f"  OOF AUC: {lgb_oof_auc:.4f}")

# XGBoost
print("[2/5] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    gamma=0.5, reg_alpha=2.0, reg_lambda=5.0,
    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='auc'
)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    xgb_model.fit(X_tr, y_tr)
    oof_preds['xgb'][val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds['xgb'].append(xgb_model.predict_proba(X_test)[:, 1])

xgb_oof_auc = roc_auc_score(y_full, oof_preds['xgb'])
print(f"  OOF AUC: {xgb_oof_auc:.4f}")

# CatBoost
print("[3/5] Training CatBoost...")
cb_model = cb.CatBoostClassifier(
    iterations=400, depth=4, learning_rate=0.03,
    l2_leaf_reg=5.0, random_strength=2.0, bagging_temperature=0.7,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=0
)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    cb_model.fit(X_tr, y_tr)
    oof_preds['cb'][val_idx] = cb_model.predict_proba(X_val)[:, 1]
    test_preds['cb'].append(cb_model.predict_proba(X_test)[:, 1])

cb_oof_auc = roc_auc_score(y_full, oof_preds['cb'])
print(f"  OOF AUC: {cb_oof_auc:.4f}")

# RandomForest
print("[4/5] Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_split=30,
    min_samples_leaf=15, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    rf_model.fit(X_tr, y_tr)
    oof_preds['rf'][val_idx] = rf_model.predict_proba(X_val)[:, 1]
    test_preds['rf'].append(rf_model.predict_proba(X_test)[:, 1])

rf_oof_auc = roc_auc_score(y_full, oof_preds['rf'])
print(f"  OOF AUC: {rf_oof_auc:.4f}")

# GradientBoosting
print("[5/5] Training GradientBoosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.03,
    subsample=0.7, min_samples_split=30, min_samples_leaf=15,
    max_features='sqrt', random_state=42
)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr, y_tr = X_train.iloc[train_idx], y_full.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_full.iloc[val_idx]

    gb_model.fit(X_tr, y_tr)
    oof_preds['gb'][val_idx] = gb_model.predict_proba(X_val)[:, 1]
    test_preds['gb'].append(gb_model.predict_proba(X_test)[:, 1])

gb_oof_auc = roc_auc_score(y_full, oof_preds['gb'])
print(f"  OOF AUC: {gb_oof_auc:.4f}")

# Average test predictions across folds
for key in test_preds:
    test_preds[key] = np.mean(test_preds[key], axis=0)

# Simple weighted average (baseline)
print("\n" + "=" * 70)
print("BASELINE: CV-WEIGHTED ENSEMBLE")
print("=" * 70)

total_cv = lgb_oof_auc + xgb_oof_auc + cb_oof_auc + rf_oof_auc + gb_oof_auc
w_lgb = lgb_oof_auc / total_cv
w_xgb = xgb_oof_auc / total_cv
w_cb = cb_oof_auc / total_cv
w_rf = rf_oof_auc / total_cv
w_gb = gb_oof_auc / total_cv

baseline_oof = (w_lgb * oof_preds['lgb'] +
                w_xgb * oof_preds['xgb'] +
                w_cb * oof_preds['cb'] +
                w_rf * oof_preds['rf'] +
                w_gb * oof_preds['gb'])

baseline_auc = roc_auc_score(y_full, baseline_oof)

print(f"Weights: LGB={w_lgb:.3f}, XGB={w_xgb:.3f}, CB={w_cb:.3f}, RF={w_rf:.3f}, GB={w_gb:.3f}")
print(f"OOF AUC: {baseline_auc:.4f}")

# Optimize weights with Optuna
print("\n" + "=" * 70)
print("OPTUNA: OPTIMIZING ENSEMBLE WEIGHTS")
print("=" * 70)

def objective(trial):
    # Suggest weights (they will be normalized later)
    w1 = trial.suggest_float('w_lgb', 0.0, 1.0)
    w2 = trial.suggest_float('w_xgb', 0.0, 1.0)
    w3 = trial.suggest_float('w_cb', 0.0, 1.0)
    w4 = trial.suggest_float('w_rf', 0.0, 1.0)
    w5 = trial.suggest_float('w_gb', 0.0, 1.0)

    # Normalize
    total = w1 + w2 + w3 + w4 + w5
    if total == 0:
        return 0.0

    w1, w2, w3, w4, w5 = w1/total, w2/total, w3/total, w4/total, w5/total

    # Ensemble OOF predictions
    ensemble_oof = (w1 * oof_preds['lgb'] +
                    w2 * oof_preds['xgb'] +
                    w3 * oof_preds['cb'] +
                    w4 * oof_preds['rf'] +
                    w5 * oof_preds['gb'])

    auc = roc_auc_score(y_full, ensemble_oof)
    return auc

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200, show_progress_bar=True)

best_params = study.best_params
total = sum(best_params.values())
opt_w_lgb = best_params['w_lgb'] / total
opt_w_xgb = best_params['w_xgb'] / total
opt_w_cb = best_params['w_cb'] / total
opt_w_rf = best_params['w_rf'] / total
opt_w_gb = best_params['w_gb'] / total

optimized_oof = (opt_w_lgb * oof_preds['lgb'] +
                 opt_w_xgb * oof_preds['xgb'] +
                 opt_w_cb * oof_preds['cb'] +
                 opt_w_rf * oof_preds['rf'] +
                 opt_w_gb * oof_preds['gb'])

optimized_auc = roc_auc_score(y_full, optimized_oof)

print(f"\nOptimal weights:")
print(f"  LightGBM:        {opt_w_lgb:.3f}")
print(f"  XGBoost:         {opt_w_xgb:.3f}")
print(f"  CatBoost:        {opt_w_cb:.3f}")
print(f"  RandomForest:    {opt_w_rf:.3f}")
print(f"  GradientBoosting:{opt_w_gb:.3f}")
print(f"\nOOF AUC: {optimized_auc:.4f}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

improvement = optimized_auc - baseline_auc
improvement_points = improvement * (100 / 0.08)

print(f"Baseline (CV-weighted):  {baseline_auc:.4f}")
print(f"Optimized (Optuna):      {optimized_auc:.4f}")
print(f"Improvement:             {improvement:+.4f} AUC ({improvement_points:+.1f} points)")

# Create optimized test predictions
optimized_test = (opt_w_lgb * test_preds['lgb'] +
                  opt_w_xgb * test_preds['xgb'] +
                  opt_w_cb * test_preds['cb'] +
                  opt_w_rf * test_preds['rf'] +
                  opt_w_gb * test_preds['gb'])

# Save predictions
output_df = pd.DataFrame({'target': optimized_test})
output_df.to_csv('answers_v5_optimized.csv', index=False)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)
print(f"✅ Optimized ensemble: answers_v5_optimized.csv")

expected_score = 100 * max(min((optimized_auc - 0.8) / 0.08, 1), 0)
print(f"\nExpected score: {expected_score:.1f} points")
print(f"v5 original:    94 points")
print(f"Expected gain:  {expected_score - 94:+.1f} points")

print("\n" + "=" * 70)
