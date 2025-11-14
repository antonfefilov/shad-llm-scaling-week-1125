#!/usr/bin/env python3
"""
Advanced Binary Classification Pipeline for Task B - Version 4
Optimized to reduce overfitting and improve test ROC-AUC
Target: ROC-AUC >= 0.88 (100 points)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import optuna
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False


def load_data():
    """Load training and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Training: {train_df.shape}, Test: {test_df.shape}")
    print(f"Target distribution: {dict(train_df['target'].value_counts(normalize=True).round(3))}")

    return train_df, test_df


def create_advanced_features(df, is_train=True):
    """Create advanced feature engineering with focus on generalization"""
    df = df.copy()

    # 1. Missing value indicator
    df['E_missing'] = df['E'].isnull().astype(int)

    # Fill missing values BEFORE creating features
    median_E = df['E'].median() if is_train else 2.0
    df['E'] = df['E'].fillna(median_E)

    # 2. Two-way interactions (most important)
    df['A_E'] = df['A'] * df['E']
    df['A_G'] = df['A'] * df['G']
    df['A_H'] = df['A'] * df['H']
    df['E_G'] = df['E'] * df['G']
    df['E_H'] = df['E'] * df['H']
    df['G_H'] = df['G'] * df['H']

    # 3. Polynomial features (squared only - simpler)
    df['A_squared'] = df['A'] ** 2
    df['G_squared'] = df['G'] ** 2
    df['H_squared'] = df['H'] ** 2

    # 4. Logarithmic transforms (handle negatives)
    df['log_A'] = np.log1p(df['A'])
    df['log_G'] = np.log1p(df['G'] + 10)  # Shift to make positive

    # 5. Ratios (most informative)
    df['A_D_ratio'] = df['A'] / (df['D'] + 0.001)
    df['G_I_ratio'] = df['G'] / (np.abs(df['I']) + 1)

    # 6. Aggregations
    df['sum_GHI'] = df['G'] + df['H'] + df['I']
    df['mean_AGH'] = (df['A'] + df['G'] + df['H']) / 3

    # 7. Binning (discretization can help with generalization)
    df['A_bin'] = pd.cut(df['A'], bins=5, labels=False, duplicates='drop')
    df['G_bin'] = pd.cut(df['G'], bins=5, labels=False, duplicates='drop')

    return df


def preprocess_data(train_df, test_df):
    """Preprocess features with advanced engineering"""
    print("\nPreprocessing with advanced features...")

    # Separate target
    X_train = train_df.drop('target', axis=1).copy()
    y_train = train_df['target'].copy()
    X_test = test_df.copy()

    # Encode categorical
    X_train['C'] = (X_train['C'] == '+').astype(int)
    X_test['C'] = (X_test['C'] == '+').astype(int)

    # Create features
    X_train = create_advanced_features(X_train, is_train=True)
    X_test = create_advanced_features(X_test, is_train=False)

    # Fill any remaining missing values
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    print(f"Features after engineering: {X_train.shape[1]}")

    return X_train, y_train, X_test


def apply_smote(X_train, y_train):
    """Apply SMOTE for balanced data"""
    if not HAS_SMOTE:
        return X_train, y_train

    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original: {len(y_train)}, After SMOTE: {len(y_resampled)}")
    print(f"New distribution: {dict(pd.Series(y_resampled).value_counts(normalize=True).round(3))}")

    return X_resampled, y_resampled


def optimize_xgboost(X_train, y_train, n_trials=30):
    """Optimize XGBoost hyperparameters with Optuna"""
    if not HAS_OPTUNA or not HAS_XGB:
        # Return default model
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=n_neg/n_pos,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )

    print(f"\nOptimizing XGBoost with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            'random_state': 42,
            'eval_metric': 'auc',
            'use_label_encoder': False
        }

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        params['scale_pos_weight'] = n_neg / n_pos

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best ROC-AUC: {study.best_value:.4f}")

    best_params = study.best_params
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    best_params['scale_pos_weight'] = n_neg / n_pos
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'auc'
    best_params['use_label_encoder'] = False

    return xgb.XGBClassifier(**best_params)


def optimize_lightgbm(X_train, y_train, n_trials=30):
    """Optimize LightGBM hyperparameters with Optuna"""
    if not HAS_OPTUNA or not HAS_LGB:
        # Return default model
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        return lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=n_neg/n_pos,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
            metric='auc'
        )

    print(f"\nOptimizing LightGBM with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            'random_state': 42,
            'verbose': -1,
            'metric': 'auc'
        }

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        params['scale_pos_weight'] = n_neg / n_pos

        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Best ROC-AUC: {study.best_value:.4f}")

    best_params = study.best_params
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    best_params['scale_pos_weight'] = n_neg / n_pos
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    best_params['metric'] = 'auc'

    return lgb.LGBMClassifier(**best_params)


def train_catboost(X_train, y_train):
    """Train CatBoost with strong regularization"""
    if not HAS_CAT:
        return None, 0.0

    print("\n" + "="*60)
    print("Training CatBoost")
    print("="*60)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    cat_model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3.0,  # Strong regularization
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False,
        eval_metric='AUC'
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cat_model, X_train, y_train, cv=cv, scoring='roc_auc')

    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    cat_model.fit(X_train, y_train)
    train_proba = cat_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Train ROC-AUC: {train_auc:.4f}")

    return cat_model, cv_scores.mean()


def feature_selection(X_train, y_train, X_test):
    """Select important features to reduce overfitting"""
    print("\nPerforming feature selection...")

    # Use LightGBM for feature importance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    selector_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        scale_pos_weight=n_neg/n_pos,
        random_state=42,
        verbose=-1
    )

    selector_model.fit(X_train, y_train)

    # Select features with importance threshold
    selector = SelectFromModel(selector_model, threshold='median', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f"Features: {X_train.shape[1]} -> {X_train_selected.shape[1]}")

    return X_train_selected, X_test_selected


def main():
    """Main pipeline with advanced optimization"""
    # Load data
    train_df, test_df = load_data()

    # Preprocess
    X_train, y_train, X_test = preprocess_data(train_df, test_df)

    # Apply SMOTE (optional - can help with imbalanced data)
    # X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    # For now, let's not use SMOTE to avoid potential overfitting

    # Feature selection to reduce overfitting
    if HAS_LGB:
        X_train_selected, X_test_selected = feature_selection(X_train, y_train, X_test)
    else:
        X_train_selected, X_test_selected = X_train, X_test

    # Train models with optimization
    models = {}
    scores = {}

    # 1. Optimized XGBoost
    if HAS_XGB:
        xgb_model = optimize_xgboost(X_train_selected, y_train, n_trials=30)

        print("\n" + "="*60)
        print("Training Optimized XGBoost")
        print("="*60)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb_model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        xgb_model.fit(X_train_selected, y_train)
        train_proba = xgb_model.predict_proba(X_train_selected)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        print(f"Train ROC-AUC: {train_auc:.4f}")

        models['XGBoost'] = xgb_model
        scores['XGBoost'] = cv_scores.mean()

    # 2. Optimized LightGBM
    if HAS_LGB:
        lgb_model = optimize_lightgbm(X_train_selected, y_train, n_trials=30)

        print("\n" + "="*60)
        print("Training Optimized LightGBM")
        print("="*60)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(lgb_model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        lgb_model.fit(X_train_selected, y_train)
        train_proba = lgb_model.predict_proba(X_train_selected)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        print(f"Train ROC-AUC: {train_auc:.4f}")

        models['LightGBM'] = lgb_model
        scores['LightGBM'] = cv_scores.mean()

    # 3. CatBoost
    cat_model, cat_score = train_catboost(X_train_selected, y_train)
    if cat_model is not None:
        models['CatBoost'] = cat_model
        scores['CatBoost'] = cat_score

    # 4. Random Forest with strong regularization
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    rf_model.fit(X_train_selected, y_train)
    train_proba = rf_model.predict_proba(X_train_selected)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Train ROC-AUC: {train_auc:.4f}")

    models['RandomForest'] = rf_model
    scores['RandomForest'] = cv_scores.mean()

    # Model comparison
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    for model_name in sorted(scores, key=scores.get, reverse=True):
        score = scores[model_name]
        points = 100 * max(min((score - 0.8) / 0.08, 1), 0)
        print(f"{model_name:20s}: {score:.4f} (~{points:.1f} points)")

    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    best_score = scores[best_model_name]

    print(f"\n🏆 Best: {best_model_name} (CV ROC-AUC: {best_score:.4f})")

    # Calibrate best model
    print("\nCalibrating probabilities...")
    calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    cv_scores_cal = cross_val_score(calibrated, X_train_selected, y_train,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                     scoring='roc_auc')
    print(f"Calibrated CV ROC-AUC: {cv_scores_cal.mean():.4f} (+/- {cv_scores_cal.std():.4f})")

    if cv_scores_cal.mean() > best_score:
        print(f"✅ Calibration improved score: {cv_scores_cal.mean():.4f}")
        calibrated.fit(X_train_selected, y_train)
        final_model = calibrated
        final_score = cv_scores_cal.mean()
    else:
        print(f"❌ Calibration didn't help, using original")
        final_model = best_model
        final_score = best_score

    # Generate predictions
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)

    probabilities = final_model.predict_proba(X_test_selected)[:, 1]
    print(f"Prob stats: min={probabilities.min():.4f}, max={probabilities.max():.4f}, mean={probabilities.mean():.4f}")

    # Save best model predictions
    output_df = pd.DataFrame({'target': probabilities})
    output_df.to_csv('answers.csv', index=False)
    print(f"✅ Best model predictions → answers.csv")

    # Create weighted ensemble
    if len(models) > 1:
        print("\nCreating weighted ensemble...")
        ensemble_probs = []
        weights = []
        for model_name, model in models.items():
            probs = model.predict_proba(X_test_selected)[:, 1]
            weight = scores[model_name]
            ensemble_probs.append(probs * weight)
            weights.append(weight)
            print(f"  {model_name}: weight={weight:.4f}")

        ensemble_probs = np.array(ensemble_probs)
        final_probs = ensemble_probs.sum(axis=0) / sum(weights)

        ensemble_df = pd.DataFrame({'target': final_probs})
        ensemble_df.to_csv('answers_ensemble.csv', index=False)
        print(f"✅ Ensemble predictions → answers_ensemble.csv")

        # Simple average
        avg_probs = ensemble_probs.mean(axis=0) / np.array(weights).mean()
        avg_df = pd.DataFrame({'target': avg_probs})
        avg_df.to_csv('answers_avg.csv', index=False)
        print(f"✅ Average ensemble → answers_avg.csv")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    expected_points = 100 * max(min((final_score - 0.8) / 0.08, 1), 0)
    print(f"Best single: {best_model_name} (CV ROC-AUC: {final_score:.4f})")
    print(f"Expected score: ~{expected_points:.1f}/100 points")
    print(f"Files: answers.csv, answers_ensemble.csv, answers_avg.csv")
    print(f"\n💡 Test all three files to find the best!")


if __name__ == "__main__":
    main()
