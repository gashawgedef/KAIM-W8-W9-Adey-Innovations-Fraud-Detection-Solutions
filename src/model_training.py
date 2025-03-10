# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, roc_auc_score
# import mlflow
# import mlflow.sklearn
# import pickle
# import os

# PROCESSED_DATA_PATH = "data/processed/"
# MODEL_PATH = "../models/"

# def load_processed_data():
#     fraud_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "fraud_data_processed.csv"))
#     credit_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "creditcard_processed.csv"))
#     return fraud_data, credit_data

# def train_model(X_train, X_test, y_train, y_test, model_name, dataset_name):
#     with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)
        
#         # Predictions
#         y_pred = model.predict(X_test)
#         y_prob = model.predict_proba(X_test)[:, 1]
        
#         # Metrics
#         roc_auc = roc_auc_score(y_test, y_prob)
#         report = classification_report(y_test, y_pred, output_dict=True)
        
#         # Log metrics to MLflow
#         mlflow.log_param("model", model_name)
#         mlflow.log_metric("roc_auc", roc_auc)
#         mlflow.log_metrics({"precision": report["1"]["precision"], "recall": report["1"]["recall"]})
#         mlflow.sklearn.log_model(model, "model")
        
#         # Save model
#         with open(os.path.join(MODEL_PATH, f"{dataset_name}_rf_model.pkl"), "wb") as f:
#             pickle.dump(model, f)
        
#         print(f"{model_name} trained on {dataset_name}. ROC-AUC: {roc_auc}")
#         return model

# def main():
#     fraud_data, credit_data = load_processed_data()
    
#     # Fraud data features and target
#     fraud_features = [col for col in fraud_data.columns if col != "class"]
#     X_fraud = fraud_data[fraud_features]
#     y_fraud = fraud_data["class"]
#     X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    
#     # Credit data features and target
#     credit_features = [col for col in credit_data.columns if col != "Class"]
#     X_credit = credit_data[credit_features]
#     y_credit = credit_data["Class"]
#     X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
    
#     # Train models
#     train_model(X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "RandomForest", "fraud")
#     train_model(X_credit_train, X_credit_test, y_credit_train, y_credit_test, "RandomForest", "credit")

# if __name__ == "__main__":
#     mlflow.set_tracking_uri("http://localhost:5000")  # Run MLflow server separately
#     main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
import pickle
import os

# Paths (adjusted for running from root directory)
PROCESSED_DATA_PATH = "data/processed/"  # Changed from "../data/processed/"
MODEL_PATH = "models/"

def load_processed_data():
    fraud_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "fraud_data_processed.csv"))
    credit_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "creditcard_processed.csv"))
    return fraud_data, credit_data

def train_model(X_train, X_test, y_train, y_test, model_name, dataset_name):
    # Optional MLflow tracking (comment out if no server is running)
    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics to MLflow (optional)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metrics({"precision": report["1"]["precision"], "recall": report["1"]["recall"]})
        mlflow.sklearn.log_model(model, "model")
        
        # Save model
        os.makedirs(MODEL_PATH, exist_ok=True)
        with open(os.path.join(MODEL_PATH, f"{dataset_name}_rf_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        print(f"{model_name} trained on {dataset_name}. ROC-AUC: {roc_auc}")
        return model

def main():
    fraud_data, credit_data = load_processed_data()
    
    # Fraud data features and target
    fraud_features = [col for col in fraud_data.columns if col != "class"]
    X_fraud = fraud_data[fraud_features]
    y_fraud = fraud_data["class"]
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    
    # Credit data features and target
    credit_features = [col for col in credit_data.columns if col != "Class"]
    X_credit = credit_data[credit_features]
    y_credit = credit_data["Class"]
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
    
    # Train models
    train_model(X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "RandomForest", "fraud")
    train_model(X_credit_train, X_credit_test, y_credit_train, y_credit_test, "RandomForest", "credit")

if __name__ == "__main__":
    # Optional: Set MLflow tracking URI (uncomment and run MLflow server if desired)
    # mlflow.set_tracking_uri("http://localhost:5000")
    main()