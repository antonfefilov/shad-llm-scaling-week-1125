import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train_data = pd.read_csv('train_weights.csv')

X_train = train_data[['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9']]
y_train = train_data['MSE']

# Train polynomial model
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

model.fit(X_train, y_train)

# Predict on training data
y_pred = model.predict(X_train)

# Calculate metrics
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print("=" * 70)
print("POLYNOMIAL MODEL (degree=2) VERIFICATION")
print("=" * 70)
print(f"\nTraining MSE: {mse:.6f}")
print(f"Training R²:  {r2:.6f}")

# Show some predictions vs actual
print("\n" + "-" * 70)
print("Sample Predictions vs Actual (first 10 rows):")
print("-" * 70)
print(f"{'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
print("-" * 70)

for i in range(min(10, len(y_train))):
    actual = y_train.iloc[i]
    pred = y_pred[i]
    error = abs(actual - pred)
    print(f"{actual:10.3f} {pred:10.3f} {error:10.3f}")

max_error = abs(y_train - y_pred).max()
mean_error = abs(y_train - y_pred).mean()

print("-" * 70)
print(f"Mean Absolute Error: {mean_error:.6f}")
print(f"Max Absolute Error:  {max_error:.6f}")
print("=" * 70)

if mse < 1:
    print("\n✓ Model is fitting the training data almost perfectly!")
    print("  This suggests MSE has a quadratic relationship with W0-W9.")
else:
    print(f"\n⚠️ MSE is {mse:.3f}, which is higher than expected.")
