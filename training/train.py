from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from dataset import load_dataset

# Load data
# X_train, y_train = load_dataset("data/splits/train.csv")
X_train, y_train = load_dataset("data/splits/train.csv")
X_val, y_val = load_dataset("data/splits/val.csv")

print("Training samples:", X_train.shape)


# Model
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", acc)

# Save model
joblib.dump(model, "models/logreg.pkl")
print("Model saved ✅")

# import os
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import joblib

# from dataset import load_dataset

# # Ensure model directory exists
# os.makedirs("models", exist_ok=True)

# # Load data
# X_train, y_train = load_dataset("data/splits/train.csv")
# X_val, y_val = load_dataset("data/splits/val.csv")

# print("Training samples:", X_train.shape)

# # Model
# model = LogisticRegression(
#     max_iter=1000,
#     n_jobs=-1
# )

# # Train
# model.fit(X_train, y_train)

# # Validate
# y_pred = model.predict(X_val)
# acc = accuracy_score(y_val, y_pred)

# print("Validation Accuracy:", acc)

# # Save model
# joblib.dump(model, "models/logreg.pkl")
# print("Model saved ✅")
