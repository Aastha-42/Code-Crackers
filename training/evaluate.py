# from sklearn.metrics import classification_report
# import joblib
# from dataset import load_dataset

# X_test, y_test = load_dataset("data/splits/test.csv")

# model = joblib.load("models/logreg.pkl")

# y_pred = model.predict(X_test)

# print(classification_report(y_test, y_pred, target_names=["CN", "AD"]))

from sklearn.metrics import classification_report
import joblib

from dataset import load_dataset

# Load test data
X_test, y_test = load_dataset("data/splits/test.csv")

# Load trained model
model = joblib.load("models/logreg.pkl")

# Predict
y_pred = model.predict(X_test)

# Report
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["CN", "AD"]
    )
)
