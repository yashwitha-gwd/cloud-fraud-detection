import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

# 2. Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

print("\nDataset shape:", data.shape)
print("Legitimate transactions:", (y == 0).sum())
print("Fraudulent transactions:", (y == 1).sum())

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 5. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("ROC-AUC Score:", round(roc_auc_score(y_test, probabilities), 4))

# 7. Save the trained model
joblib.dump(model, "fraud_model.joblib")

print("\nModel saved successfully as fraud_model.joblib!")

