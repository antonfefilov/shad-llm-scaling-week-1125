import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Load training data
train_data = pd.read_csv('train_weights.csv')

X_train = train_data[['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9']]
y_train = train_data['MSE']

print("=" * 70)
print("CHECKING FOR OVERFITTING/UNDERFITTING")
print("=" * 70)

# 1. Train on full training set and check performance
model = LinearRegression()
model.fit(X_train, y_train)

# Training error
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\n" + "-" * 70)
print("Full Training Set Performance:")
print("-" * 70)
print(f"Training MSE: {train_mse:.3f}")
print(f"Training R²:  {train_r2:.4f}")

# 2. Cross-validation to check generalization
print("\n" + "-" * 70)
print("Cross-Validation (5-Fold):")
print("-" * 70)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kfold,
                            scoring='neg_mean_squared_error')
cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold,
                               scoring='r2')

cv_mse = -cv_scores.mean()
cv_mse_std = cv_scores.std()
cv_r2 = cv_r2_scores.mean()
cv_r2_std = cv_r2_scores.std()

print(f"CV MSE:       {cv_mse:.3f} (±{cv_mse_std:.3f})")
print(f"CV R²:        {cv_r2:.4f} (±{cv_r2_std:.4f})")

# 3. Analyze results
print("\n" + "-" * 70)
print("Analysis:")
print("-" * 70)

gap = abs(train_mse - cv_mse)
relative_gap = gap / train_mse * 100

print(f"\nGap between training and CV MSE: {gap:.3f} ({relative_gap:.1f}%)")

if train_r2 > 0.9 and cv_r2 < 0.7:
    print("\n⚠️  HIGH OVERFITTING detected!")
    print("   → Training R² is much higher than CV R²")
    print("   → Model memorizing training data instead of learning patterns")
elif train_r2 < 0.5 and cv_r2 < 0.5:
    print("\n⚠️  UNDERFITTING detected!")
    print("   → Both training and CV R² are low")
    print("   → Model is too simple to capture the patterns")
    print("\n   Suggestions:")
    print("   - Try polynomial features")
    print("   - Try more complex models (Random Forest, Gradient Boosting)")
    print("   - Add interaction terms")
elif relative_gap < 10:
    print("\n✓  Model looks good - minimal overfitting")
    print("   → Training and CV performance are similar")
else:
    print("\n⚠️  MODERATE OVERFITTING detected")
    print("   → Some gap between training and CV performance")
    print("\n   Suggestions:")
    print("   - Try regularization (Ridge, Lasso)")
    print("   - Collect more training data")

# 4. Check individual fold performance
print("\n" + "-" * 70)
print("Individual Fold Performance:")
print("-" * 70)
for i, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    fold_model = LinearRegression()
    fold_model.fit(X_fold_train, y_fold_train)

    fold_train_mse = mean_squared_error(y_fold_train, fold_model.predict(X_fold_train))
    fold_val_mse = mean_squared_error(y_fold_val, fold_model.predict(X_fold_val))

    print(f"Fold {i}: Train MSE = {fold_train_mse:.3f}, Val MSE = {fold_val_mse:.3f}, Gap = {abs(fold_train_mse - fold_val_mse):.3f}")

# 5. Summary
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

if cv_r2 < 0.3:
    print("\n❌ Poor Model Performance (R² < 0.3)")
    print("\nRecommendations:")
    print("1. Try polynomial features (degree 2 or 3)")
    print("2. Try ensemble methods (Random Forest, XGBoost)")
    print("3. Add feature interactions (e.g., W0*W1, W0²)")
    print("4. Check if there's a non-linear relationship")
elif cv_r2 < 0.6:
    print("\n⚠️  Moderate Model Performance (0.3 < R² < 0.6)")
    print("\nRecommendations:")
    print("1. Try polynomial features")
    print("2. Try regularization (Ridge/Lasso)")
    print("3. Try tree-based models")
else:
    print("\n✓ Good Model Performance (R² > 0.6)")
    print("\nThe model generalizes reasonably well.")

print("\n" + "=" * 70)
