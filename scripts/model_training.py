from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM
import mlflow
import numpy as np


def prepare_data(df, target_column):
    """
    Separate features and target, and split into train and test sets.

    Parameters:
        df (pd.DataFrame): Input dataset.
        target_column (str): Name of the target column.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def select_model(model_name, input_shape):
    """
    Select and initialize a machine learning model.

    Parameters:
        model_name (str): Name of the model to use.
        input_shape (tuple): Shape of the input data (required for CNN and LSTM).

    Returns:
        model: Initialized model.
    """
    if model_name == "LogisticRegression":
        return LogisticRegression()
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier()
    elif model_name == "RandomForest":
        return RandomForestClassifier()
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier()
    elif model_name == "MLP":
        return MLPClassifier()
    elif model_name == "CNN":
        model = Sequential()
        model.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                input_shape=input_shape,
            )
        )
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model
    elif model_name == "LSTM":
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a machine learning model.

    Parameters:
        model: Initialized model.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
    """
    if isinstance(model, Sequential):  # Check if the model is a Keras model
        # Reshape data for CNN and LSTM
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))


def log_experiment(model, params, metrics):
    """
    Log experiment details using MLflow.

    Parameters:
        model: Trained model.
        params (dict): Hyperparameters used.
        metrics (dict): Evaluation metrics.
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
