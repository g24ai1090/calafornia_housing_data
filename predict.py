import joblib
from sklearn.datasets import fetch_california_housing

model = joblib.load("model.joblib")
print(" Model loaded successfully inside the container!")

data = fetch_california_housing()
X, y = data.data, data.target

sample = X[0].reshape(1, -1)
prediction = model.predict(sample)

print(f"Test Prediction (for verification): {prediction[0]:.4f}")
