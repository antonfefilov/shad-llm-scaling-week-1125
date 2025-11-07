import pandas as pd
import json
from sklearn.linear_model import LinearRegression

# Load training data
train_data = pd.read_csv('train_weights.csv')

# Separate features (W0-W9) and target (MSE)
X_train = train_data[['W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9']]
y_train = train_data['MSE']

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Load test data
test_data = pd.read_csv('test_weights.csv')

# Make predictions
predictions = model.predict(test_data)

# Format output as JSON
results = []
for i in range(len(test_data)):
    result_dict = {}
    for col in test_data.columns:
        result_dict[col] = test_data.iloc[i][col]
    result_dict['MSE'] = round(predictions[i], 3)
    results.append(result_dict)

# Write to result.txt
with open('result.txt', 'w') as f:
    json.dump(results, f, indent=4)

print(f"Training completed. Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")
print(f"Predictions saved to result.txt")
