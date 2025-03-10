# Fraud Detection Project

This project is part of the 10 Academy Artificial Intelligence Mastery Week 8&9 Challenge. It focuses on improving fraud detection for e-commerce and bank transactions using machine learning models, geolocation analysis, and real-time deployment.

## Structure
- `data/`: Contains raw and processed datasets.
- `src/`: Source code for data preprocessing, model training, explainability, API, and dashboard.
- `models/`: Stores trained model files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `Dockerfile`: Docker configuration for deployment.
- `requirements.txt`: Project dependencies.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess data: `python src/data_preprocessing.py`
3. Train models: `python src/model_training.py`
4. Explain models: `python src/model_explainability.py`
5. Run API: `python src/serve_model.py`
6. Run Dashboard: `python src/dashboard.py`
7. Build Docker image: `docker build -t fraud-detection-model .`
8. Run Docker container: `docker run -p 5000:5000 fraud-detection-model`

## Requirements
- Python 3.8+
- Access to datasets: `Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`

## Deliverables
- Task 1: Data preprocessing and feature engineering
- Task 2: Model training with MLflow tracking
- Task 3: Model explainability using SHAP and LIME
- Task 4: Flask API deployment with Docker
- Task 5: Interactive dashboard with Dash