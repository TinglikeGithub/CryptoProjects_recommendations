from flask import Flask, render_template, request
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

from src.recommendation_system import match


def service_mapping(service):
    mapping = {
        "blockchain_infrastructure": "Service_Blockchain Infrastructure",
        "blockchain_service": "Service_Blockchain Service",
        "cefi": "Service_CeFi",
        "chain": "Service_Chain",
        "defi": "Service_DeFi",
        "gamefi": "Service_GameFi",
        "social": "Service_Social",
        "stablecoin": "Service_Stablecoin"
    }
    return mapping.get(service, "")

column_weights = {
    'Total Raised': 0.5,
    'First Funding Year': 0,
    'First Funding Month': 0,
    'First Funding Day': 0,
    'Funding Round_Angel': 0.0,
    'Funding Round_Pre-Seed': 0.0,
    'Funding Round_Pre-Series A': 0.0,
    'Funding Round_Seed': 0.0,
    'Funding Round_Series A': 0.0,
    'Funding Round_Strategic': 0.0,
    'Funding Round_Undisclosed': 0.0
}


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:

            data = pd.read_csv("Data/data.csv")
            description = request.form['description']
            no_of_recommendation = request.form['no_of_recommendation']
            service = request.form['service']
            funding_round = request.form['funding_round']
            amount_raised = request.form['amount_raised']

            if amount_raised:
                min_value = data['Total Raised'].min()
                max_value = data['Total Raised'].max()
                amount_raised = (int(amount_raised) - min_value)/(max_value - min_value)

                column_weights['Total Raised'] = 1

            column_weights['Funding Round_Angel'] = 1 if funding_round=="angel" else 0.5
            column_weights['Funding Round_Pre-Seed'] = 1 if funding_round=="pre_seed" else 0.5
            column_weights['Funding Round_Pre-Series A'] = 1 if funding_round=="pre_series_a" else 0.5
            column_weights['Funding Round_Seed'] = 1 if funding_round=="seed" else 0.5
            column_weights['Funding Round_Series A'] = 1 if funding_round=="series_a" else 0.5
            column_weights['Funding Round_Strategic'] = 1 if funding_round=="strategic" else 0.5
            column_weights['Funding Round_Undisclosed'] = 1 if funding_round=="undisclosed" else 0.5

            # Perform some processing on the input text (e.g., sentiment analysis)
            # Replace this with your actual processing code

            # if service:
            #     service = service_mapping(service)
            #     data = data[data['Service']==service.replace("Service_","")]


            processed_data = data.drop(columns=["Crypto Name", "Name", "Raised Amount", "First Funding Date", "Valuation Amount", "Links"])

            processed_data.fillna(0, inplace=True)
            processed_data = pd.get_dummies(processed_data, columns= ["Service", "Funding Round"], dtype = int)
            processed_data["Inverstors_and_desc"] = processed_data["Investors"].astype(str)+" "+processed_data["Description"].astype(str)
            processed_data = processed_data.drop(columns=["Investors","Description"])
            # Initialize the MinMaxScaler
            scaler = MinMaxScaler()

            # List of numeric column names
            numeric_columns = ['Total Raised', 'First Funding Year', 'First Funding Month',
                'First Funding Day', 'Service_Blockchain Infrastructure',
                'Service_Blockchain Service', 'Service_CeFi', 'Service_Chain',
                'Service_DeFi', 'Service_GameFi', 'Service_Social',
                'Service_Stablecoin', 'Funding Round_Angel', 'Funding Round_Pre-Seed',
                'Funding Round_Pre-Series A', 'Funding Round_Seed',
                'Funding Round_Series A', 'Funding Round_Strategic',
                'Funding Round_Undisclosed']

            # Apply min-max normalization to each numeric column
            processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
            if service:
                service = service_mapping(service)
            processed_data = processed_data[processed_data[service]==1]

            top_indices = match(processed_data=processed_data,
                                top_n=no_of_recommendation,
                                column_weights=column_weights,
                                description=description,
                                amount_raised=amount_raised
                                # service=service,
                                # funding_round=funding_round
                                )

            # result = data.iloc[top_indices].values.tolist()

            result = data.iloc[top_indices].values.tolist()
        except Exception as e:
            with open("log.txt", "a") as f:
                f.write(str(e) + "\n" + amount_raised)
            pass
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

