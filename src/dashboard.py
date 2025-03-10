# from flask import Flask, jsonify
# import dash
# from dash import dcc, html
# import plotly.express as px
# import pandas as pd

# server = Flask(__name__)
# app = dash.Dash(__name__, server=server)

# # Load data
# fraud_data = pd.read_csv("data/processed/fraud_data_processed.csv")
# fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"])

# # Dashboard layout
# app.layout = html.Div([
#     html.H1("Fraud Detection Dashboard"),
    
#     # Summary boxes
#     html.Div([
#         html.Div(f"Total Transactions: {len(fraud_data)}"),
#         html.Div(f"Fraud Cases: {fraud_data['class'].sum()}"),
#         html.Div(f"Fraud Percentage: {fraud_data['class'].mean() * 100:.2f}%"),
#     ]),
    
#     # Fraud over time
#     dcc.Graph(figure=px.line(fraud_data.groupby(fraud_data["purchase_time"].dt.date)["class"].sum().reset_index(),
#                              x="purchase_time", y="class", title="Fraud Cases Over Time")),
    
#     # Fraud by country
#     dcc.Graph(figure=px.bar(fraud_data.groupby("country")["class"].sum().reset_index(),
#                             x="country", y="class", title="Fraud by Country")),
    
#     # Fraud by device
#     dcc.Graph(figure=px.bar(fraud_data.groupby("device_id")["class"].sum().reset_index().head(10),  # Limit for readability
#                             x="device_id", y="class", title="Fraud by Device (Top 10)")),
# ])

# @server.route("/data")
# def get_data():
#     summary = {
#         "total_transactions": len(fraud_data),
#         "fraud_cases": int(fraud_data["class"].sum()),
#         "fraud_percentage": fraud_data["class"].mean() * 100
#     }
#     return jsonify(summary)

# if __name__ == "__main__":
#     app.run_server(debug=True, host="0.0.0.0", port=8050)

from flask import Flask, jsonify
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import os

# Paths
PROCESSED_DATA_PATH = "data/processed/"

# Load data
fraud_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "fraud_data_processed.csv"))

# Flask app
server = Flask(__name__)

# Dash app
app = dash.Dash(__name__, server=server)

# Summary stats
total_transactions = len(fraud_data)
fraud_cases = fraud_data["class"].sum()
fraud_percentage = (fraud_cases / total_transactions) * 100

@server.route("/data")
def get_data():
    return jsonify({
        "total_transactions": int(total_transactions),
        "fraud_cases": int(fraud_cases),
        "fraud_percentage": round(fraud_percentage, 2)
    })

# Visualizations
# Fraud by hour of day (instead of purchase_time)
fig_hour = px.histogram(fraud_data, x="hour_of_day", color="class", 
                        title="Fraud Distribution by Hour of Day",
                        labels={"class": "Fraud (1) / Non-Fraud (0)"})

# Fraud by country
fig_country = px.bar(fraud_data.groupby("country")["class"].mean().reset_index(),
                     x="country", y="class", title="Fraud Rate by Country")

# Layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    html.Div([
        html.H3(f"Total Transactions: {total_transactions}"),
        html.H3(f"Fraud Cases: {fraud_cases}"),
        html.H3(f"Fraud Percentage: {fraud_percentage:.2f}%")
    ]),
    dcc.Graph(figure=fig_hour),
    dcc.Graph(figure=fig_country)
])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)