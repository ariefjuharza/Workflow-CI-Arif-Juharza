# MLProject/Modelling.py - K3 MLflow Project ✅
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# MLflow autolog
mlflow.sklearn.autolog()

# Load data
X_train = pd.read_csv("namadataset_preprocessing/X_train_processed.csv").values
X_test = pd.read_csv("namadataset_preprocessing/X_test_processed.csv").values
y_train = pd.read_csv("namadataset_preprocessing/y_train.csv").squeeze()
y_test = pd.read_csv("namadataset_preprocessing/y_test.csv").squeeze()

print("Data loaded:", X_train.shape)

# Train & log
with mlflow.start_run(run_name="ci-rf"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("test_auc", auc)

    print(f"✅ Test AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))
