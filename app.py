from flask import Flask, render_template, request
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os

from src.recommendation_system import match

data = pd.read_csv("Data/data.csv")

column_weights = {
    'Total Raised': 0.8,
    'First Funding Year': 0.6,
    'First Funding Month': 0.6,
    'First Funding Day': 0.6,
    'Service_Blockchain Infrastructure': 0.0,
    'Service_Blockchain Service': 0.0,
    'Service_CeFi': 0.0,
    'Service_Chain': 0.0,
    'Service_DeFi': 0.0,
    'Service_GameFi': 0.0,
    'Service_Social': 0.0,
    'Service_Stablecoin': 0.0,
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
            description = request.form['description']
            no_of_recommendation = request.form['no_of_recommendation']
            service = request.form['service']
            funding_round = request.form['funding_round']

            # weights update
            column_weights['Service_Blockchain Infrastructure'] = 1 if service=="blockchain_infrastructure" else 0.6
            column_weights['Service_Blockchain Service'] = 1 if service=="blockchain_service" else 0.6
            column_weights['Service_CeFi'] = 1 if service=="cefi" else 0.6
            column_weights['Service_Chain'] = 1 if service=="chain" else 0.6
            column_weights['Service_DeFi'] = 1 if service=="defi" else 0.6
            column_weights['Service_GameFi'] = 1 if service=="gamefi" else 0.6
            column_weights['Service_Social'] = 1 if service=="social" else 0.6
            column_weights['Service_Stablecoin'] = 1 if service=="stablecoin" else 0.6

            column_weights['Funding Round_Angel'] = 1 if funding_round=="angel" else 0.6
            column_weights['Funding Round_Pre-Seed'] = 1 if funding_round=="pre_seed" else 0.6
            column_weights['Funding Round_Pre-Series A'] = 1 if funding_round=="pre_series_a" else 0.6
            column_weights['Funding Round_Seed'] = 1 if funding_round=="seed" else 0.6
            column_weights['Funding Round_Series A'] = 1 if funding_round=="series_a" else 0.6
            column_weights['Funding Round_Strategic'] = 1 if funding_round=="strategic" else 0.6
            column_weights['Funding Round_Undisclosed'] = 1 if funding_round=="undisclosed" else 0.6

            # Perform some processing on the input text (e.g., sentiment analysis)
            # Replace this with your actual processing code
            top_indices = match(top_n=no_of_recommendation, column_weights=column_weights, description=description)

            # result = data.iloc[top_indices].values.tolist()
            result = data.iloc[top_indices].to_html(classes='data', header="true")
            print(no_of_recommendation, "\n", "\n", column_weights, "\n", "\n", description)
        except:
            pass
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

