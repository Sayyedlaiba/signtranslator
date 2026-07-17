import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

print("Loading dataset...")
data = pd.read_csv("words_dataset.csv")

# Split features (coordinates) and labels (words)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train the model instantly
print("Training the Word Prediction Model...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Save the intelligence file
joblib.dump(model, "word_recognition_model.pkl")
print("Success! 'word_recognition_model.pkl' created and ready for action.")