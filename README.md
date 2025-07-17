# anakin
#Step 1: Install Required Libraries
#pip install numpy pandas matplotlib scikit-learn
#Step 2: Simulate Sensor Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate sensor readings (temperature, vibration)
np.random.seed(0)
time = np.arange(0, 1000)
temperature = 50 + np.random.normal(0, 1, 1000) + 0.02 * time  # Gradually increasing temp
vibration = 0.5 + np.random.normal(0, 0.1, 1000) + 0.001 * time  # Gradually increasing vibration

# Simulate failures when thresholds are crossed
failure = (temperature > 70) | (vibration > 1.0)
#failure = (temperature > 70) & (vibration > 1.0)
# Create DataFrame
df = pd.DataFrame({
    'Time': time,
    'Temperature': temperature,
    'Vibration': vibration,
    'Failure': failure.astype(int)
})

print(df.head())
#Step 3: Visualize the Sensor Data

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['Time'], df['Temperature'], label='Temperature')
plt.axhline(y=70, color='r', linestyle='--')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['Time'], df['Vibration'], label='Vibration', color='orange')
plt.axhline(y=1.0, color='r', linestyle='--')
plt.legend()
plt.show()
# Step 4: Train a Machine Learning Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Features and target
X = df[['Temperature', 'Vibration']]
y = df['Failure']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#Step 5: Predict Real-Time Sensor Output

# Example of new sensor reading
new_data = pd.DataFrame({
    'Temperature': [70],
    'Vibration': [1.0]
})

prediction = model.predict(new_data)
print("Predicted Failure:", "Yes" if prediction[0] == 1 else "No")
