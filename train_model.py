# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

CSV_PATH = "dataset.csv"
MODEL_PATH = "sign_model.pkl"

if not pd.io.common.file_exists(CSV_PATH):
    print(f"Error: Missing '{CSV_PATH}'. Run collect_data.py first to create a dataset.")
    exit()

# Load dataset
df = pd.read_csv(CSV_PATH)
X = df.drop('label', axis=1)
y = df['label']

print(f"Loaded dataset with {len(df)} samples across classes: {df['label'].unique()}")

# Split Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Validation Accuracy: {acc * 100:.2f}%")

# Save the trained model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Successfully trained and saved AI classifier as '{MODEL_PATH}'! 🎉")