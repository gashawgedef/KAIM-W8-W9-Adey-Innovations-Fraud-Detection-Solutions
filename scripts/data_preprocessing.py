# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def handle_missing_values(df, strategy="drop", fill_value=None):
    """
    Handle missing values in the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.
        strategy (str): Strategy to handle missing values ('drop' or 'impute').
        fill_value: Value to use for imputation (if strategy is 'impute').

    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "impute":
        df = df.fillna(fill_value)
    return df


def clean_data(df):
    """
    Clean the dataset by removing duplicates and correcting data types.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.drop_duplicates()
    # Correct data types (e.g., timestamps, categorical variables)
    if "signup_time" in df.columns:
        df["signup_time"] = pd.to_datetime(df["signup_time"])
    if "purchase_time" in df.columns:
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    return df


def perform_eda(df):
    """
    Perform exploratory data analysis (univariate and bivariate).

    Parameters:
        df (pd.DataFrame): Input dataset.
    """
    # Univariate analysis
    print(df.describe())
    print(df["class"].value_counts())  # Target variable distribution

    # Bivariate analysis
    import seaborn as sns

    sns.pairplot(df, hue="class")  # Visualize relationships between features


def merge_geolocation_data(fraud_df, ip_country_df):
    """
    Merge Fraud_Data.csv with IpAddress_to_Country.csv for geolocation analysis.

    Parameters:
        fraud_df (pd.DataFrame): Fraud_Data dataset.
        ip_country_df (pd.DataFrame): IpAddress_to_Country dataset.

    Returns:
        pd.DataFrame: Merged dataset with country information.
    """
    # Convert IP addresses to integer format
    fraud_df["ip_address"] = fraud_df["ip_address"].apply(
        lambda x: int(x.replace(".", ""))
    )

    # Merge datasets
    merged_df = pd.merge(
        fraud_df,
        ip_country_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        how="left",
    )
    return merged_df


def feature_engineering(df):
    """
    Create new features for fraud detection.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with new features.
    """
    # Time-based features
    if "purchase_time" in df.columns:
        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek

    # Transaction frequency and velocity
    if "user_id" in df.columns:
        df["transaction_frequency"] = df.groupby("user_id")["purchase_time"].transform(
            "count"
        )
        df["transaction_velocity"] = (
            df.groupby("user_id")["purchase_time"].diff().dt.total_seconds()
        )

    return df


def normalize_data(df, features):
    """
    Normalize numerical features.

    Parameters:
        df (pd.DataFrame): Input dataset.
        features (list): List of numerical features to normalize.

    Returns:
        pd.DataFrame: Dataset with normalized features.
    """
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def encode_categorical_features(df, categorical_features):
    """
    Encode categorical features using one-hot encoding.

    Parameters:
        df (pd.DataFrame): Input dataset.
        categorical_features (list): List of categorical features to encode.

    Returns:
        pd.DataFrame: Dataset with encoded features.
    """
    encoder = OneHotEncoder(drop="first", sparse=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_features, columns=encoder.get_feature_names_out(categorical_features)
    )
    df = pd.concat([df.drop(categorical_features, axis=1), encoded_df], axis=1)
    return df
