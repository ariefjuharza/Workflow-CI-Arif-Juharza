import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def retrain():
    mlflow.sklearn.autolog()
    
    train_path = "breast_cancer_preprocessing/train.csv"
    test_path = "breast_cancer_preprocessing/test.csv"
    
    if not os.path.exists(train_path):
        print("Preprocessed data not found.")
        return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    with mlflow.start_run(run_name="CI_Retraining_Run"):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Retraining Accuracy: {acc:.4f}")

if __name__ == "__main__":
    retrain()
