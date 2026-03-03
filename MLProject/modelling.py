import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
X_train = pd.read_csv("preprocessing/X_train_processed.csv").values
X_test = pd.read_csv("preprocessing/X_test_processed.csv").values
y_train = pd.read_csv("preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("preprocessing/y_test.csv").values.ravel()

print("Data loaded:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)

print(f"✅ Test AUC: {auc:.3f}")
print(classification_report(y_test, y_pred))
print("✅ Training completed!")
