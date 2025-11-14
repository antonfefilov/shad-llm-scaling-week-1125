#!/usr/bin/env python3
"""
Advanced Binary Classification Pipeline for Task B - Version 5
Focus: Reduce overfitting while keeping all features
Target: ROC-AUC >= 0.88 (100 points)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
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


def load_data():
    """Load training and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Training: {train_df.shape}, Test: {test_df.shape}")
    print(f"Target distribution: {dict(train_df['target'].value_counts(normalize=True).round(3))}")

    return train_df, test_df


def create_conservative_features(df, is_train=True):
    """Create features with focus on generalization"""
    df = df.copy()

    # 1. Missing value indicator
    df['E_missing'] = df['E'].isnull().astype(int)

    # Fill missing values
    median_E = df['E'].median() if is_train else 2.0
    df['E'] = df['E'].fillna(median_E)

    # 2. Most important interactions only (from previous analysis)
    df['A_E'] = df['A'] * df['E']
    df['A_G'] = df['A'] * df['G']
    df['A_H'] = df['A'] * df['H']
    df['G_H'] = df['G'] * df['H']

    # 3. Polynomial features (squared only)
    df['A_squared'] = df['A'] ** 2
    df['G_squared'] = df['G'] ** 2
    df['H_squared'] = df['H'] ** 2

    # 4. Log transforms
    df['log_A'] = np.log1p(df['A'])
    df['log_G'] = np.log1p(df['G'] + 10)

    # 5. Key ratios
    df['A_D_ratio'] = df['A'] / (df['D'] + 0.001)
    df['G_I_ratio'] = df['G'] / (np.abs(df['I']) + 1)

    # 6. Aggregations
    df['sum_GHI'] = df['G'] + df['H'] + df['I']

    return df


def preprocess_data(train_df, test_df):
    """Preprocess features"""
    print("\nPreprocessing with conservative features...")

    # Separate target
    X_train = train_df.drop('target', axis=1).copy()
    y_train = train_df['target'].copy()
    X_test = test_df.copy()

    # Encode categorical
    X_train['C'] = (X_train['C'] == '+').astype(int)
    X_test['C'] = (X_test['C'] == '+').astype(int)

    # Create features
    X_train = create_conservative_features(X_train, is_train=True)
    X_test = create_conservative_features(X_test, is_train=False)

    # Fill any remaining missing values
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    print(f"Features after engineering: {X_train.shape[1]}")

    return X_train, y_train, X_test


def train_models_conservative(X_train, y_train):
    """Train models with strong regularization to prevent overfitting"""
    models = {}
    scores = {}

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    print(f"Class imbalance: {n_neg} class 0, {n_pos} class 1 (ratio: {scale_pos_weight:.2f})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. XGBoost with very strong regularization
    if HAS_XGB:
        print("\n" + "="*60)
        print("XGBoost (Strong Regularization)")
        print("="*60)

        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,  # Shallow trees
            learning_rate=0.03,  # Slow learning
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,  # Strong regularization
            gamma=0.5,
            reg_alpha=2.0,  # Strong L1
            reg_lambda=5.0,  # Strong L2
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )

        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        xgb_model.fit(X_train, y_train)
        train_proba = xgb_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        print(f"Train ROC-AUC: {train_auc:.4f}")
        print(f"Overfit gap: {train_auc - cv_scores.mean():.4f}")

        models['XGBoost'] = xgb_model
        scores['XGBoost'] = cv_scores.mean()

    # 2. LightGBM with very strong regularization
    if HAS_LGB:
        print("\n" + "="*60)
        print("LightGBM (Strong Regularization)")
        print("="*60)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            num_leaves=15,  # Few leaves
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=30,  # Strong regularization
            reg_alpha=2.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            metric='auc'
        )

        cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        lgb_model.fit(X_train, y_train)
        train_proba = lgb_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        print(f"Train ROC-AUC: {train_auc:.4f}")
        print(f"Overfit gap: {train_auc - cv_scores.mean():.4f}")

        models['LightGBM'] = lgb_model
        scores['LightGBM'] = cv_scores.mean()

    # 3. CatBoost with strong regularization
    if HAS_CAT:
        print("\n" + "="*60)
        print("CatBoost (Strong Regularization)")
        print("="*60)

        cat_model = CatBoostClassifier(
            iterations=400,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=False,
            eval_metric='AUC'
        )

        cv_scores = cross_val_score(cat_model, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        cat_model.fit(X_train, y_train)
        train_proba = cat_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        print(f"Train ROC-AUC: {train_auc:.4f}")
        print(f"Overfit gap: {train_auc - cv_scores.mean():.4f}")

        models['CatBoost'] = cat_model
        scores['CatBoost'] = cv_scores.mean()

    # 4. Random Forest with strong regularization
    print("\n" + "="*60)
    print("Random Forest (Strong Regularization)")
    print("="*60)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    rf_model.fit(X_train, y_train)
    train_proba = rf_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Overfit gap: {train_auc - cv_scores.mean():.4f}")

    models['RandomForest'] = rf_model
    scores['RandomForest'] = cv_scores.mean()

    # 5. Gradient Boosting with strong regularization
    print("\n" + "="*60)
    print("Gradient Boosting (Strong Regularization)")
    print("="*60)

    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_split=30,
        min_samples_leaf=15,
        subsample=0.7,
        max_features='sqrt',
        random_state=42
    )

    cv_scores = cross_val_score(gb_model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    gb_model.fit(X_train, y_train)
    train_proba = gb_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Overfit gap: {train_auc - cv_scores.mean():.4f}")

    models['GradientBoosting'] = gb_model
    scores['GradientBoosting'] = cv_scores.mean()

    return models, scores


def create_stacking_ensemble(models, X_train, y_train):
    """Create stacking ensemble with strong regularization"""
    print("\n" + "="*60)
    print("Stacking Ensemble (Conservative)")
    print("="*60)

    # Use only top models for stacking
    estimators = [(name, model) for name, model in models.items()]

    # Use simple logistic regression as meta-learner with strong regularization
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=0.01,  # Strong regularization
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        cv=5,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stacking, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    stacking.fit(X_train, y_train)
    train_proba = stacking.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Overfit gap: {train_auc - cv_scores.mean():.4f}")

    return stacking, cv_scores.mean()


def main():
    """Main pipeline"""
    # Load data
    train_df, test_df = load_data()

    # Preprocess
    X_train, y_train, X_test = preprocess_data(train_df, test_df)

    # Train models with strong regularization
    models, scores = train_models_conservative(X_train, y_train)

    # Create stacking ensemble
    stacking, stacking_score = create_stacking_ensemble(models, X_train, y_train)
    models['Stacking'] = stacking
    scores['Stacking'] = stacking_score

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

    # Try multiple calibration methods
    print("\n" + "="*60)
    print("Testing Calibration Methods")
    print("="*60)

    calibration_methods = ['sigmoid', 'isotonic']
    calibrated_models = {}
    calibrated_scores = {}

    for method in calibration_methods:
        print(f"\nTrying {method} calibration...")
        calibrated = CalibratedClassifierCV(best_model, method=method, cv=3)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_cal = cross_val_score(calibrated, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"  CV ROC-AUC: {cv_scores_cal.mean():.4f} (+/- {cv_scores_cal.std():.4f})")

        calibrated_models[method] = calibrated
        calibrated_scores[method] = cv_scores_cal.mean()

    # Choose best calibration
    best_cal_method = max(calibrated_scores, key=calibrated_scores.get)
    best_cal_score = calibrated_scores[best_cal_method]

    if best_cal_score > best_score:
        print(f"\n✅ Best calibration: {best_cal_method} ({best_cal_score:.4f})")
        final_model = calibrated_models[best_cal_method]
        final_model.fit(X_train, y_train)
        final_score = best_cal_score
    else:
        print(f"\n❌ Calibration didn't help, using original")
        final_model = best_model
        final_score = best_score

    # Generate predictions
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)

    probabilities = final_model.predict_proba(X_test)[:, 1]
    print(f"Prob stats: min={probabilities.min():.4f}, max={probabilities.max():.4f}, mean={probabilities.mean():.4f}")

    # Save best model predictions
    output_df = pd.DataFrame({'target': probabilities})
    output_df.to_csv('answers.csv', index=False)
    print(f"✅ Best model predictions → answers.csv")

    # Create weighted ensemble (excluding stacking to avoid meta-overfitting)
    base_models = {k: v for k, v in models.items() if k != 'Stacking'}
    if len(base_models) > 1:
        print("\nCreating weighted ensemble (base models only)...")
        ensemble_probs = []
        weights = []
        for model_name, model in base_models.items():
            probs = model.predict_proba(X_test)[:, 1]
            weight = scores[model_name]
            ensemble_probs.append(probs * weight)
            weights.append(weight)
            print(f"  {model_name}: weight={weight:.4f}")

        ensemble_probs = np.array(ensemble_probs)
        final_probs = ensemble_probs.sum(axis=0) / sum(weights)

        ensemble_df = pd.DataFrame({'target': final_probs})
        ensemble_df.to_csv('answers_ensemble.csv', index=False)
        print(f"✅ Weighted ensemble → answers_ensemble.csv")

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
    print(f"Best model: {best_model_name}")
    print(f"CV ROC-AUC: {final_score:.4f}")
    print(f"Expected score: ~{expected_points:.1f}/100 points")
    print(f"\nFiles generated:")
    print(f"  - answers.csv (best single model)")
    print(f"  - answers_ensemble.csv (weighted ensemble)")
    print(f"  - answers_avg.csv (average ensemble)")
    print(f"\n💡 Recommendation: Try answers.csv first!")


if __name__ == "__main__":
    main()
