#!/usr/bin/env python3
"""
Classifier v6: Domain Adaptation for distribution shift
Based on adversarial validation findings (AUC=0.9034)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CLASSIFIER V6: DOMAIN ADAPTATION")
print("=" * 70)
print("\nПроблема: Adversarial AUC = 0.9034 (train ≠ test)")
print("Решение: Sample re-weighting + domain features")

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X_full = train_df.drop('target', axis=1).copy()
y_full = train_df['target'].copy()
X_test_full = test_df.copy()

print(f"\nTrain: {len(X_full)} samples")
print(f"Test:  {len(X_test_full)} samples")

def create_domain_adaptive_features(X_train, X_test, fit_scaler=True):
    """
    Create features with domain adaptation
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Encode categorical
    X_train['C'] = (X_train['C'] == '+').astype(int)
    X_test['C'] = (X_test['C'] == '+').astype(int)

    # Handle missing values
    E_median = X_train['E'].median()
    X_train['E_missing'] = X_train['E'].isnull().astype(int)
    X_test['E_missing'] = X_test['E'].isnull().astype(int)
    X_train['E'] = X_train['E'].fillna(E_median)
    X_test['E'] = X_test['E'].fillna(E_median)

    # Robust scaling for features with distribution shift
    # H, I, A are the most different
    if fit_scaler:
        scaler = RobustScaler()
        shift_features = ['H', 'I', 'A', 'C']
        X_train[shift_features] = scaler.fit_transform(X_train[shift_features])
        X_test[shift_features] = scaler.transform(X_test[shift_features])

    # Key interactions
    X_train['A_E'] = X_train['A'] * X_train['E']
    X_test['A_E'] = X_test['A'] * X_test['E']

    X_train['A_G'] = X_train['A'] * X_train['G']
    X_test['A_G'] = X_test['A'] * X_test['G']

    X_train['A_H'] = X_train['A'] * X_train['H']
    X_test['A_H'] = X_test['A'] * X_test['H']

    X_train['G_H'] = X_train['G'] * X_train['H']
    X_test['G_H'] = X_test['G'] * X_test['H']

    # Squared terms
    X_train['A_squared'] = X_train['A'] ** 2
    X_test['A_squared'] = X_test['A'] ** 2

    X_train['G_squared'] = X_train['G'] ** 2
    X_test['G_squared'] = X_test['G'] ** 2

    X_train['H_squared'] = X_train['H'] ** 2
    X_test['H_squared'] = X_test['H'] ** 2

    # Log transforms
    X_train['log_A'] = np.log1p(np.abs(X_train['A']))
    X_test['log_A'] = np.log1p(np.abs(X_test['A']))

    X_train['log_G'] = np.log1p(X_train['G'] + 10)
    X_test['log_G'] = np.log1p(X_test['G'] + 10)

    # Ratios
    X_train['A_D_ratio'] = X_train['A'] / (np.abs(X_train['D']) + 0.001)
    X_test['A_D_ratio'] = X_test['A'] / (np.abs(X_test['D']) + 0.001)

    X_train['G_I_ratio'] = X_train['G'] / (np.abs(X_train['I']) + 1)
    X_test['G_I_ratio'] = X_test['G'] / (np.abs(X_test['I']) + 1)

    # Aggregations
    X_train['sum_GHI'] = X_train['G'] + X_train['H'] + X_train['I']
    X_test['sum_GHI'] = X_test['G'] + X_test['H'] + X_test['I']

    return X_train, X_test

# Compute adversarial weights
print("\n" + "=" * 70)
print("STEP 1: COMPUTING ADVERSARIAL WEIGHTS")
print("=" * 70)

X_train_adv = X_full.copy()
X_test_adv = X_test_full.copy()

# Simple preprocessing for adversarial model
X_train_adv['C'] = (X_train_adv['C'] == '+').astype(int)
X_test_adv['C'] = (X_test_adv['C'] == '+').astype(int)
X_train_adv['E'] = X_train_adv['E'].fillna(X_train_adv['E'].median())
X_test_adv['E'] = X_test_adv['E'].fillna(X_test_adv['E'].median())

# Create adversarial dataset
X_train_adv['is_test'] = 0
X_test_adv['is_test'] = 1
X_combined = pd.concat([X_train_adv, X_test_adv], axis=0, ignore_index=True)
y_combined = X_combined['is_test']
X_combined = X_combined.drop('is_test', axis=1)

# Train adversarial model
adv_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
adv_model.fit(X_combined, y_combined)

# Get probabilities that train samples look like test
train_test_probs = adv_model.predict_proba(X_train_adv.drop('is_test', axis=1))[:, 1]

print(f"Adversarial probabilities (train looks like test):")
print(f"  Min:    {train_test_probs.min():.4f}")
print(f"  Mean:   {train_test_probs.mean():.4f}")
print(f"  Median: {np.median(train_test_probs):.4f}")
print(f"  Max:    {train_test_probs.max():.4f}")

# Sample weights: higher weight for train samples that look like test
sample_weights = train_test_probs ** 2  # Square to amplify effect
sample_weights = sample_weights / sample_weights.mean()  # Normalize

print(f"\nSample weights:")
print(f"  Min:    {sample_weights.min():.4f}")
print(f"  Mean:   {sample_weights.mean():.4f}")
print(f"  Max:    {sample_weights.max():.4f}")

# Feature engineering
X_train, X_test = create_domain_adaptive_features(X_full, X_test_full)

print(f"\nFeatures: {X_train.shape[1]}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# Class imbalance
n_pos = y_full.sum()
n_neg = len(y_full) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"\nClass balance: {n_neg} negative, {n_pos} positive")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

print("\n" + "=" * 70)
print("STEP 2: TRAINING WITH SAMPLE WEIGHTS")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: LightGBM with sample weights
print("\n[1/5] LightGBM + Sample Weights")
print("-" * 70)

lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    min_child_samples=30,
    reg_alpha=2.0,
    reg_lambda=5.0,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

lgb_cv_scores = []
lgb_train_scores = []
lgb_test_preds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    # Use sample weights
    weights_tr = sample_weights[train_idx]

    lgb_model.fit(X_tr, y_tr, sample_weight=weights_tr)

    train_pred = lgb_model.predict_proba(X_tr)[:, 1]
    val_pred = lgb_model.predict_proba(X_val)[:, 1]
    test_pred = lgb_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_tr, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    lgb_train_scores.append(train_auc)
    lgb_cv_scores.append(val_auc)
    lgb_test_preds.append(test_pred)

    print(f"  Fold {fold+1}: Train={train_auc:.4f}, Val={val_auc:.4f}")

lgb_train_mean = np.mean(lgb_train_scores)
lgb_cv_mean = np.mean(lgb_cv_scores)
lgb_gap = lgb_train_mean - lgb_cv_mean

print(f"  Train AUC:   {lgb_train_mean:.4f}")
print(f"  CV AUC:      {lgb_cv_mean:.4f}")
print(f"  Overfit Gap: {lgb_gap:.4f} {'✅' if lgb_gap < 0.12 else '⚠️'}")

lgb_test_pred_final = np.mean(lgb_test_preds, axis=0)

# Model 2: XGBoost with sample weights
print("\n[2/5] XGBoost + Sample Weights")
print("-" * 70)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    gamma=0.5,
    reg_alpha=2.0,
    reg_lambda=5.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

xgb_cv_scores = []
xgb_train_scores = []
xgb_test_preds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    weights_tr = sample_weights[train_idx]

    xgb_model.fit(X_tr, y_tr, sample_weight=weights_tr)

    train_pred = xgb_model.predict_proba(X_tr)[:, 1]
    val_pred = xgb_model.predict_proba(X_val)[:, 1]
    test_pred = xgb_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_tr, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    xgb_train_scores.append(train_auc)
    xgb_cv_scores.append(val_auc)
    xgb_test_preds.append(test_pred)

    print(f"  Fold {fold+1}: Train={train_auc:.4f}, Val={val_auc:.4f}")

xgb_train_mean = np.mean(xgb_train_scores)
xgb_cv_mean = np.mean(xgb_cv_scores)
xgb_gap = xgb_train_mean - xgb_cv_mean

print(f"  Train AUC:   {xgb_train_mean:.4f}")
print(f"  CV AUC:      {xgb_cv_mean:.4f}")
print(f"  Overfit Gap: {xgb_gap:.4f} {'✅' if xgb_gap < 0.12 else '⚠️'}")

xgb_test_pred_final = np.mean(xgb_test_preds, axis=0)

# Model 3: CatBoost with sample weights
print("\n[3/5] CatBoost + Sample Weights")
print("-" * 70)

cb_model = cb.CatBoostClassifier(
    iterations=400,
    depth=4,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    random_strength=2.0,
    bagging_temperature=0.7,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=0
)

cb_cv_scores = []
cb_train_scores = []
cb_test_preds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    weights_tr = sample_weights[train_idx]

    cb_model.fit(X_tr, y_tr, sample_weight=weights_tr)

    train_pred = cb_model.predict_proba(X_tr)[:, 1]
    val_pred = cb_model.predict_proba(X_val)[:, 1]
    test_pred = cb_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_tr, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    cb_train_scores.append(train_auc)
    cb_cv_scores.append(val_auc)
    cb_test_preds.append(test_pred)

    print(f"  Fold {fold+1}: Train={train_auc:.4f}, Val={val_auc:.4f}")

cb_train_mean = np.mean(cb_train_scores)
cb_cv_mean = np.mean(cb_cv_scores)
cb_gap = cb_train_mean - cb_cv_mean

print(f"  Train AUC:   {cb_train_mean:.4f}")
print(f"  CV AUC:      {cb_cv_mean:.4f}")
print(f"  Overfit Gap: {cb_gap:.4f} {'✅' if cb_gap < 0.12 else '⚠️'}")

cb_test_pred_final = np.mean(cb_test_preds, axis=0)

# Model 4: RandomForest with sample weights
print("\n[4/5] RandomForest + Sample Weights")
print("-" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_cv_scores = []
rf_train_scores = []
rf_test_preds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    weights_tr = sample_weights[train_idx]

    rf_model.fit(X_tr, y_tr, sample_weight=weights_tr)

    train_pred = rf_model.predict_proba(X_tr)[:, 1]
    val_pred = rf_model.predict_proba(X_val)[:, 1]
    test_pred = rf_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_tr, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    rf_train_scores.append(train_auc)
    rf_cv_scores.append(val_auc)
    rf_test_preds.append(test_pred)

    print(f"  Fold {fold+1}: Train={train_auc:.4f}, Val={val_auc:.4f}")

rf_train_mean = np.mean(rf_train_scores)
rf_cv_mean = np.mean(rf_cv_scores)
rf_gap = rf_train_mean - rf_cv_mean

print(f"  Train AUC:   {rf_train_mean:.4f}")
print(f"  CV AUC:      {rf_cv_mean:.4f}")
print(f"  Overfit Gap: {rf_gap:.4f} {'✅' if rf_gap < 0.12 else '⚠️'}")

rf_test_pred_final = np.mean(rf_test_preds, axis=0)

# Model 5: GradientBoosting with sample weights
print("\n[5/5] GradientBoosting + Sample Weights")
print("-" * 70)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features='sqrt',
    random_state=42
)

gb_cv_scores = []
gb_train_scores = []
gb_test_preds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_full)):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    weights_tr = sample_weights[train_idx]

    gb_model.fit(X_tr, y_tr, sample_weight=weights_tr)

    train_pred = gb_model.predict_proba(X_tr)[:, 1]
    val_pred = gb_model.predict_proba(X_val)[:, 1]
    test_pred = gb_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_tr, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    gb_train_scores.append(train_auc)
    gb_cv_scores.append(val_auc)
    gb_test_preds.append(test_pred)

    print(f"  Fold {fold+1}: Train={train_auc:.4f}, Val={val_auc:.4f}")

gb_train_mean = np.mean(gb_train_scores)
gb_cv_mean = np.mean(gb_cv_scores)
gb_gap = gb_train_mean - gb_cv_mean

print(f"  Train AUC:   {gb_train_mean:.4f}")
print(f"  CV AUC:      {gb_cv_mean:.4f}")
print(f"  Overfit Gap: {gb_gap:.4f} {'✅' if gb_gap < 0.12 else '⚠️'}")

gb_test_pred_final = np.mean(gb_test_preds, axis=0)

# Summary
print("\n" + "=" * 70)
print("SUMMARY - DOMAIN ADAPTED MODELS")
print("=" * 70)

results = [
    ("LightGBM", lgb_cv_mean, lgb_train_mean, lgb_gap),
    ("XGBoost", xgb_cv_mean, xgb_train_mean, xgb_gap),
    ("CatBoost", cb_cv_mean, cb_train_mean, cb_gap),
    ("RandomForest", rf_cv_mean, rf_train_mean, rf_gap),
    ("GradientBoosting", gb_cv_mean, gb_train_mean, gb_gap),
]

results.sort(key=lambda x: x[1], reverse=True)

for name, cv_auc, train_auc, gap in results:
    expected_score = 100 * max(min((cv_auc - 0.8) / 0.08, 1), 0)
    print(f"\n{name}:")
    print(f"  CV AUC:      {cv_auc:.4f} (Expected: {expected_score:.1f} points)")
    print(f"  Train AUC:   {train_auc:.4f}")
    print(f"  Overfit Gap: {gap:.4f} {'✅' if gap < 0.12 else '⚠️'}")

# Ensemble predictions
print("\n" + "=" * 70)
print("ENSEMBLE")
print("=" * 70)

# Weighted ensemble (based on CV scores)
total_cv = lgb_cv_mean + xgb_cv_mean + cb_cv_mean + rf_cv_mean + gb_cv_mean
w_lgb = lgb_cv_mean / total_cv
w_xgb = xgb_cv_mean / total_cv
w_cb = cb_cv_mean / total_cv
w_rf = rf_cv_mean / total_cv
w_gb = gb_cv_mean / total_cv

ensemble_pred = (w_lgb * lgb_test_pred_final +
                 w_xgb * xgb_test_pred_final +
                 w_cb * cb_test_pred_final +
                 w_rf * rf_test_pred_final +
                 w_gb * gb_test_pred_final)

print(f"Weights: LightGBM={w_lgb:.3f}, XGBoost={w_xgb:.3f}, CatBoost={w_cb:.3f}, RF={w_rf:.3f}, GB={w_gb:.3f}")

# Simple average
avg_pred = (lgb_test_pred_final + xgb_test_pred_final + cb_test_pred_final +
            rf_test_pred_final + gb_test_pred_final) / 5

# Save predictions
best_model_name = results[0][0]
if best_model_name == "LightGBM":
    best_pred = lgb_test_pred_final
elif best_model_name == "XGBoost":
    best_pred = xgb_test_pred_final
elif best_model_name == "CatBoost":
    best_pred = cb_test_pred_final
elif best_model_name == "RandomForest":
    best_pred = rf_test_pred_final
else:
    best_pred = gb_test_pred_final

# Save
output_df = pd.DataFrame({'target': best_pred})
output_df.to_csv('answers_v6.csv', index=False)
print(f"\n✅ Best model ({best_model_name}) predictions saved to: answers_v6.csv")

output_df_ensemble = pd.DataFrame({'target': ensemble_pred})
output_df_ensemble.to_csv('answers_v6_ensemble.csv', index=False)
print(f"✅ Weighted ensemble predictions saved to: answers_v6_ensemble.csv")

output_df_avg = pd.DataFrame({'target': avg_pred})
output_df_avg.to_csv('answers_v6_avg.csv', index=False)
print(f"✅ Simple average predictions saved to: answers_v6_avg.csv")

print("\n" + "=" * 70)
print("COMPARISON: v5 vs v6")
print("=" * 70)
print("""
v5 (без domain adaptation):
  CV AUC:  0.8809 → Test: 94 балла
  Gap:     0.1057
  Problem: Train и Test из разных распределений (Adv AUC=0.90)

v6 (с domain adaptation):
  Sample re-weighting: Train примеры, похожие на Test, получают больший вес
  Robust scaling: Нормализация H, I, A (признаки с наибольшим shift)
  Expected: Лучше обобщение на Test благодаря domain adaptation
""")

print("\n" + "=" * 70)
print("РЕКОМЕНДАЦИИ")
print("=" * 70)
print("""
Протестируйте все три файла:
1. answers_v6.csv          - лучшая одиночная модель
2. answers_v6_ensemble.csv - взвешенный ансамбль (рекомендуется)
3. answers_v6_avg.csv      - простое среднее

Ожидаем улучшение над v5 (94 балла) благодаря domain adaptation.
""")

print("=" * 70)
