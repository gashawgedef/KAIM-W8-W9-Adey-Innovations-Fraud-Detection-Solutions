import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Paths (adjusted for running from root directory)
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

def load_data():
    full_path = os.path.join(RAW_DATA_PATH, "Fraud_Data.csv")
    print(f"Looking for file at: {full_path}")
    fraud_data = pd.read_csv(full_path)
    ip_data = pd.read_csv(os.path.join(RAW_DATA_PATH, "IpAddress_to_Country.csv"))
    credit_data = pd.read_csv(os.path.join(RAW_DATA_PATH, "creditcard.csv"))
    return fraud_data, ip_data, credit_data

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df["signup_time"] = pd.to_datetime(df.get("signup_time", pd.NaT))
    df["purchase_time"] = pd.to_datetime(df.get("purchase_time", pd.NaT))
    return df

def merge_geolocation(fraud_data, ip_data):
    fraud_data["ip_address"] = fraud_data["ip_address"].astype(int)
    ip_data = ip_data.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"])
    ip_data["lower_bound_ip_address"] = ip_data["lower_bound_ip_address"].astype(int)
    ip_data["upper_bound_ip_address"] = ip_data["upper_bound_ip_address"].astype(int)
    fraud_data = fraud_data.sort_values("ip_address")
    ip_data = ip_data.sort_values("lower_bound_ip_address")
    merged = pd.merge_asof(
        fraud_data,
        ip_data[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )
    merged = merged[
        (merged["ip_address"] >= merged["lower_bound_ip_address"]) & 
        (merged["ip_address"] <= merged["upper_bound_ip_address"])
    ]
    merged = merged.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])
    return merged

def feature_engineering(fraud_data):
    fraud_data["hour_of_day"] = fraud_data["purchase_time"].dt.hour
    fraud_data["day_of_week"] = fraud_data["purchase_time"].dt.dayofweek
    fraud_data["time_since_signup"] = (fraud_data["purchase_time"] - fraud_data["signup_time"]).dt.total_seconds() / 3600
    freq = fraud_data.groupby("user_id")["purchase_time"].count().reset_index(name="tx_frequency")
    fraud_data = fraud_data.merge(freq, on="user_id", how="left")
    # Drop non-numeric columns after feature extraction
    fraud_data = fraud_data.drop(columns=["signup_time", "purchase_time", "user_id", "device_id"])
    return fraud_data

def preprocess_data():
    fraud_data, ip_data, credit_data = load_data()
    fraud_data = clean_data(fraud_data)
    credit_data = clean_data(credit_data)
    fraud_data = merge_geolocation(fraud_data, ip_data)
    fraud_data = feature_engineering(fraud_data)
    le = LabelEncoder()
    for col in ["source", "browser", "sex", "country"]:
        fraud_data[col] = le.fit_transform(fraud_data[col].fillna("Unknown"))
    scaler = StandardScaler()
    fraud_num_cols = ["purchase_value", "age", "time_since_signup", "tx_frequency"]
    fraud_data[fraud_num_cols] = scaler.fit_transform(fraud_data[fraud_num_cols])
    credit_num_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    credit_data[credit_num_cols] = scaler.fit_transform(credit_data[credit_num_cols])
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    fraud_data.to_csv(os.path.join(PROCESSED_DATA_PATH, "fraud_data_processed.csv"), index=False)
    credit_data.to_csv(os.path.join(PROCESSED_DATA_PATH, "creditcard_processed.csv"), index=False)
    return fraud_data, credit_data

if __name__ == "__main__":
    fraud_data, credit_data = preprocess_data()
    print("Data preprocessing completed.")