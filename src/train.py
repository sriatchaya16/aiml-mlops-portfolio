
import os
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "machine_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "machine_failure_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# Features and target
X = df.drop(columns=["Machine_Failure"])
y = df["Machine_Failure"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create and train the model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\nModel Evaluation")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

# Save trained model
joblib.dump(model, MODEL_PATH)
print("\nModel saved successfully!")
print("Location:", MODEL_PATH)

# Track metrics using MLflow
mlflow.set_experiment("Machine_Failure_Prediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", 0.25)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

print("MLflow tracking completed!")
