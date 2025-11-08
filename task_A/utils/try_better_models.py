import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Load data
train_data = pd.read_csv('train_weights.csv')
test_data = pd.read_csv('test_weights.csv')

X_train = train_data[['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9']]
y_train = train_data['MSE']

print("=" * 70)
print("TESTING DIFFERENT MODELS")
print("=" * 70)

models = {}

# 1. Linear models
models['Linear Regression'] = LinearRegression()
models['Ridge (α=1)'] = Ridge(alpha=1.0)
models['Ridge (α=10)'] = Ridge(alpha=10.0)
models['Lasso (α=0.1)'] = Lasso(alpha=0.1)

# 2. Polynomial features
models['Poly-2 + Linear'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])
models['Poly-2 + Ridge'] = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0))
])

# 3. Tree-based models
models['Random Forest (100)'] = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
models['Random Forest (200)'] = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)

print("\nCross-Validation Results (5-fold):")
print("-" * 70)

best_model = None
best_score = float('inf')
best_name = None

results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5,
                            scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    std_mse = scores.std()

    results.append({
        'name': name,
        'mse': mean_mse,
        'std': std_mse
    })

    if mean_mse < best_score:
        best_score = mean_mse
        best_model = model
        best_name = name

# Sort by MSE
results = sorted(results, key=lambda x: x['mse'])

for r in results:
    print(f"{r['name']:25s} | CV MSE: {r['mse']:7.3f} (±{r['std']:.3f})")

print("=" * 70)
print(f"\nBest Model: {best_name}")
print(f"Best CV MSE: {best_score:.3f}")
print("=" * 70)

# Train best model on full training set
print(f"\nTraining {best_name} on full dataset...")
best_model.fit(X_train, y_train)

# Make predictions on test set
predictions = best_model.predict(test_data)

# Save results
results = []
for i in range(len(test_data)):
    result_dict = {}
    for col in test_data.columns:
        result_dict[col] = test_data.iloc[i][col]
    result_dict['MSE'] = round(predictions[i], 3)
    results.append(result_dict)

with open('result.txt', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✓ Predictions saved to result.txt using {best_name}")
print("\nPrediction statistics:")
print(f"  Min:  {predictions.min():.3f}")
print(f"  Max:  {predictions.max():.3f}")
print(f"  Mean: {predictions.mean():.3f}")
print(f"  Std:  {predictions.std():.3f}")
