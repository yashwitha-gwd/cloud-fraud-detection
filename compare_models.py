
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. Load dataset
print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

X = data.drop(columns=["Class"])
y = data["Class"]

print("Dataset shape:", data.shape)
print("Fraud cases:", int(y.sum()))

# 2. Split data fairly for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}
import joblib
results = []

# 4. Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start

    results.append({
        "Model": name,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
        "Training + Evaluation Time (seconds)": round(
            elapsed, 2
        )
    })

# 5. Display results
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))

#results_df.to_csv("model_comparison.csv", index=False)

#print("\nResults saved to model_comparison.csv")

results_df.to_csv("model_comparison.csv", index=False)

# Save the Random Forest model
joblib.dump(models["Random Forest"], "fraud_model_rf.joblib")

print("\nResults saved to model_comparison.csv")
print("Random Forest model saved as fraud_model_rf.joblib")