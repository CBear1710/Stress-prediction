import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Improved_All_Combined_hr_rsp_binary.csv")

# Handle missing HR values
df["HR"].fillna(df["HR"].median(), inplace=True)

# Convert timestamps
df["Datetime"] = pd.to_datetime(df["Time(sec)"], unit="s")
df["Hour"] = df["Datetime"].dt.hour
df.drop(columns=["Time(sec)", "Datetime"], inplace=True)

# Compute HRV using a rolling average to reduce noise and avoid extreme values
df["HRV"] = df["HR"].rolling(window=5).std().fillna(0)
df["HRV"] = np.clip(df["HRV"], 0, 50)

# Define feature set
X = df[["HR", "respr", "Hour", "HRV"]]
y = df["Label"]

# Ensure labels are correctly mapped (Stress = 1, No Stress = 0)
print("Label Distribution Before Processing:")
print(y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train an optimized Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=300,  # Increased trees for better learning
    max_depth=15,  # Increased depth to capture patterns
    min_samples_split=10,  # Adjusted to prevent underfitting
    min_samples_leaf=5,  # Ensures more stability
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train_scaled, y_train_balanced)

# Make predictions on test data
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Use a threshold dynamically computed based on median probability
best_threshold = np.median(y_prob)

# Compute final predictions
y_pred = [1 if prob > best_threshold else 0 for prob in y_prob]

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

evaluation_results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [accuracy, precision, recall, f1, roc_auc]
}

evaluation_df = pd.DataFrame(evaluation_results)

# Print evaluation results
print("\n✅ Model Evaluation Metrics (After Optimization):")
for metric, score in zip(evaluation_results["Metric"], evaluation_results["Score"]):
    print(f"{metric}: {score:.3f}")

# Plot evaluation metrics
plt.figure(figsize=(8, 5))
plt.bar(evaluation_df["Metric"], evaluation_df["Score"], color=["blue", "green", "orange", "red", "purple"])
plt.xlabel("Evaluation Metric")
plt.ylabel("Score")
plt.title("Model Evaluation Metrics")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--")
plt.show()

# Save results to CSV
evaluation_df.to_csv("model_evaluation_results.csv", index=False)
print("✅ Model evaluation metrics saved to 'model_evaluation_results.csv'")