#!/usr/bin/env python3
"""
Binary Classification Pipeline for Task B
Trains models on train.csv and predicts on test.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost/LightGBM, fall back to RandomForest if not available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, will use Random Forest only")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available")


def load_data():
    """Load training and test datasets"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTraining data info:")
    print(train_df.info())
    print(f"\nMissing values in training data:")
    print(train_df.isnull().sum())
    print(f"\nTarget distribution:")
    print(train_df['target'].value_counts())

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess features: encode categorical, handle missing values"""
    print("\nPreprocessing data...")

    # Separate features and target
    X_train = train_df.drop('target', axis=1).copy()
    y_train = train_df['target'].copy()
    X_test = test_df.copy()

    # Encode categorical feature C: '+' -> 1, '-' -> 0
    X_train['C'] = (X_train['C'] == '+').astype(int)
    X_test['C'] = (X_test['C'] == '+').astype(int)

    # Handle missing values in feature E
    # For tree-based models, we can either:
    # 1. Fill with median/mode
    # 2. Create a special category (-1)
    # Let's use median imputation
    median_E = X_train['E'].median()
    X_train['E'] = X_train['E'].fillna(median_E)
    X_test['E'] = X_test['E'].fillna(median_E)

    # Check for any other missing values
    if X_train.isnull().sum().sum() > 0:
        print(f"Warning: Still have missing values in training data")
        print(X_train.isnull().sum())
        # Fill any remaining missing values with median
        X_train = X_train.fillna(X_train.median())

    if X_test.isnull().sum().sum() > 0:
        print(f"Warning: Still have missing values in test data")
        print(X_test.isnull().sum())
        # Fill any remaining missing values with median from training data
        X_test = X_test.fillna(X_train.median())

    print(f"Preprocessed training features shape: {X_train.shape}")
    print(f"Preprocessed test features shape: {X_test.shape}")

    return X_train, y_train, X_test


def train_random_forest(X_train, y_train, n_splits=5):
    """Train Random Forest classifier with cross-validation"""
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)

    # Random Forest with good default parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    rf_model.fit(X_train, y_train)
    train_pred = rf_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")

    # Feature importance
    feature_names = X_train.columns
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature importances:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    return rf_model, cv_scores.mean()


def train_xgboost(X_train, y_train, n_splits=5):
    """Train XGBoost classifier with cross-validation"""
    if not HAS_XGB:
        return None, 0.0

    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)

    # XGBoost with good default parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    xgb_model.fit(X_train, y_train)
    train_pred = xgb_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")

    # Feature importance
    feature_names = X_train.columns
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature importances:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    return xgb_model, cv_scores.mean()


def train_lightgbm(X_train, y_train, n_splits=5):
    """Train LightGBM classifier with cross-validation"""
    if not HAS_LGB:
        return None, 0.0

    print("\n" + "="*60)
    print("Training LightGBM Classifier")
    print("="*60)

    # LightGBM with good default parameters
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    lgb_model.fit(X_train, y_train)
    train_pred = lgb_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")

    # Feature importance
    feature_names = X_train.columns
    importances = lgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature importances:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    return lgb_model, cv_scores.mean()


def main():
    """Main pipeline"""
    # Load data
    train_df, test_df = load_data()

    # Preprocess
    X_train, y_train, X_test = preprocess_data(train_df, test_df)

    # Train multiple models and compare
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

    # Select best model based on CV score
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    for model_name, score in scores.items():
        print(f"{model_name}: {score:.4f}")

    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} with CV accuracy: {scores[best_model_name]:.4f}")

    # Generate predictions on test data
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)
    predictions = best_model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Class {val}: {count} ({count/len(predictions)*100:.1f}%)")

    # Save predictions
    output_df = pd.DataFrame({'target': predictions})
    output_df.to_csv('predictions.csv', index=False)
    print(f"\nPredictions saved to 'predictions.csv'")
    print(f"Output shape: {output_df.shape}")

    # Also try ensemble prediction (average of all models) if we have multiple
    if len(models) > 1:
        print("\n" + "="*60)
        print("Ensemble Prediction (Voting)")
        print("="*60)
        ensemble_preds = []
        for model_name, model in models.items():
            preds = model.predict(X_test)
            ensemble_preds.append(preds)

        # Majority voting
        ensemble_preds = np.array(ensemble_preds)
        final_preds = np.round(ensemble_preds.mean(axis=0)).astype(int)

        print(f"Ensemble predictions distribution:")
        unique, counts = np.unique(final_preds, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  Class {val}: {count} ({count/len(final_preds)*100:.1f}%)")

        # Save ensemble predictions
        ensemble_df = pd.DataFrame({'target': final_preds})
        ensemble_df.to_csv('predictions_ensemble.csv', index=False)
        print(f"Ensemble predictions saved to 'predictions_ensemble.csv'")


if __name__ == "__main__":
    main()
