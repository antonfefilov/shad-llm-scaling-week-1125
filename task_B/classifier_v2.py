#!/usr/bin/env python3
"""
Improved Binary Classification Pipeline for Task B
Optimized for ROC-AUC metric (outputs probabilities, not binary predictions)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost/LightGBM
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


def load_data():
    """Load training and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTarget distribution:")
    print(train_df['target'].value_counts(normalize=True))

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess features"""
    print("\nPreprocessing data...")

    # Separate features and target
    X_train = train_df.drop('target', axis=1).copy()
    y_train = train_df['target'].copy()
    X_test = test_df.copy()

    # Encode categorical feature C: '+' -> 1, '-' -> 0
    X_train['C'] = (X_train['C'] == '+').astype(int)
    X_test['C'] = (X_test['C'] == '+').astype(int)

    # Handle missing values in feature E with median
    median_E = X_train['E'].median()
    X_train['E'] = X_train['E'].fillna(median_E)
    X_test['E'] = X_test['E'].fillna(median_E)

    # Fill any other missing values
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    print(f"Final training features shape: {X_train.shape}")
    print(f"Final test features shape: {X_test.shape}")

    return X_train, y_train, X_test


def train_random_forest(X_train, y_train, n_splits=5):
    """Train Random Forest optimized for ROC-AUC"""
    print("\n" + "="*60)
    print("Training Random Forest Classifier (optimized for ROC-AUC)")
    print("="*60)

    # Calculate class weights for imbalanced data
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    scale_pos_weight = n_neg / n_pos

    print(f"Class imbalance: {n_neg} class 0, {n_pos} class 1 (ratio: {scale_pos_weight:.2f})")

    # Random Forest with balanced class weights
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation with ROC-AUC
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')

    print(f"Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    rf_model.fit(X_train, y_train)

    # Get probabilities for ROC-AUC
    train_proba = rf_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Training ROC-AUC: {train_auc:.4f}")

    return rf_model, cv_scores.mean()


def train_xgboost(X_train, y_train, n_splits=5):
    """Train XGBoost optimized for ROC-AUC"""
    if not HAS_XGB:
        return None, 0.0

    print("\n" + "="*60)
    print("Training XGBoost Classifier (optimized for ROC-AUC)")
    print("="*60)

    # Calculate scale_pos_weight for imbalanced data
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    scale_pos_weight = n_neg / n_pos

    print(f"Class imbalance: {n_neg} class 0, {n_pos} class 1 (ratio: {scale_pos_weight:.2f})")

    # XGBoost optimized for ROC-AUC
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )

    # Cross-validation with ROC-AUC
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')

    print(f"Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    xgb_model.fit(X_train, y_train)

    # Get probabilities for ROC-AUC
    train_proba = xgb_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Training ROC-AUC: {train_auc:.4f}")

    return xgb_model, cv_scores.mean()


def train_lightgbm(X_train, y_train, n_splits=5):
    """Train LightGBM optimized for ROC-AUC"""
    if not HAS_LGB:
        return None, 0.0

    print("\n" + "="*60)
    print("Training LightGBM Classifier (optimized for ROC-AUC)")
    print("="*60)

    # Calculate scale_pos_weight for imbalanced data
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    scale_pos_weight = n_neg / n_pos

    print(f"Class imbalance: {n_neg} class 0, {n_pos} class 1 (ratio: {scale_pos_weight:.2f})")

    # LightGBM optimized for ROC-AUC
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        metric='auc'
    )

    # Cross-validation with ROC-AUC
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='roc_auc')

    print(f"Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    lgb_model.fit(X_train, y_train)

    # Get probabilities for ROC-AUC
    train_proba = lgb_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Training ROC-AUC: {train_auc:.4f}")

    return lgb_model, cv_scores.mean()


def main():
    """Main pipeline"""
    # Load data
    train_df, test_df = load_data()

    # Preprocess
    X_train, y_train, X_test = preprocess_data(train_df, test_df)

    # Train multiple models
    models = {}
    scores = {}

    # Random Forest (always available)
    rf_model, rf_score = train_random_forest(X_train, y_train)
    models['RandomForest'] = rf_model
    scores['RandomForest'] = rf_score

    # XGBoost (if available)
    xgb_model, xgb_score = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        models['XGBoost'] = xgb_model
        scores['XGBoost'] = xgb_score

    # LightGBM (if available)
    lgb_model, lgb_score = train_lightgbm(X_train, y_train)
    if lgb_model is not None:
        models['LightGBM'] = lgb_model
        scores['LightGBM'] = lgb_score

    # Compare models
    print("\n" + "="*60)
    print("Model Comparison (ROC-AUC)")
    print("="*60)
    for model_name, score in scores.items():
        print(f"{model_name}: {score:.4f}")

    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    best_score = scores[best_model_name]

    print(f"\n🏆 Best model: {best_model_name} with CV ROC-AUC: {best_score:.4f}")

    # Calculate expected score
    if best_score >= 0.88:
        expected_points = 100
    elif best_score >= 0.80:
        expected_points = 100 * (best_score - 0.8) / 0.08
    else:
        expected_points = 0

    print(f"Expected score: {expected_points:.1f}/100 points")

    # Generate PROBABILITY predictions (important for ROC-AUC!)
    print("\n" + "="*60)
    print("Generating Probability Predictions")
    print("="*60)

    # Get probabilities for class 1
    probabilities = best_model.predict_proba(X_test)[:, 1]

    print(f"Probability statistics:")
    print(f"  Min: {probabilities.min():.4f}")
    print(f"  Max: {probabilities.max():.4f}")
    print(f"  Mean: {probabilities.mean():.4f}")
    print(f"  Median: {np.median(probabilities):.4f}")

    # Save probabilities to answers.csv
    output_df = pd.DataFrame({'target': probabilities})
    output_df.to_csv('answers.csv', index=False)
    print(f"\n✅ Probability predictions saved to 'answers.csv'")
    print(f"   This file contains probabilities for ROC-AUC calculation")

    # Create ensemble predictions from all models
    if len(models) > 1:
        print("\n" + "="*60)
        print("Creating Weighted Ensemble Predictions")
        print("="*60)

        ensemble_probs = []
        weights = []
        for model_name, model in models.items():
            probs = model.predict_proba(X_test)[:, 1]
            weight = scores[model_name]
            ensemble_probs.append(probs * weight)
            weights.append(weight)
            print(f"{model_name}: weight = {weight:.4f}")

        # Weighted average
        ensemble_probs = np.array(ensemble_probs)
        final_probs = ensemble_probs.sum(axis=0) / sum(weights)

        print(f"\nEnsemble probability statistics:")
        print(f"  Min: {final_probs.min():.4f}")
        print(f"  Max: {final_probs.max():.4f}")
        print(f"  Mean: {final_probs.mean():.4f}")

        # Calculate potential score improvement
        # Ensembles often improve by 0.01-0.03 in practice
        potential_improvement = 0.02
        ensemble_estimated_score = min(best_score + potential_improvement, 1.0)

        if ensemble_estimated_score >= 0.88:
            ensemble_points = 100
        elif ensemble_estimated_score >= 0.80:
            ensemble_points = 100 * (ensemble_estimated_score - 0.8) / 0.08
        else:
            ensemble_points = 0

        print(f"\nEstimated ensemble ROC-AUC: ~{ensemble_estimated_score:.4f}")
        print(f"Estimated ensemble score: ~{ensemble_points:.1f}/100 points")

        # Save ensemble probabilities
        ensemble_df = pd.DataFrame({'target': final_probs})
        ensemble_df.to_csv('answers_ensemble.csv', index=False)
        print(f"\n✅ Ensemble predictions saved to 'answers_ensemble.csv'")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✅ Best single model: {best_model_name} (CV ROC-AUC: {best_score:.4f})")
    print(f"✅ Predictions saved to: answers.csv")
    if len(models) > 1:
        print(f"✅ Ensemble predictions saved to: answers_ensemble.csv")
    print(f"\n💡 Try both files and see which gives better ROC-AUC on test set!")


if __name__ == "__main__":
    main()
