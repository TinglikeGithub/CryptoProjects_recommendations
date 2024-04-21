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
    'Service_Blockchain Infrastructure': 1.0,
    'Service_Blockchain Service': 1.0,
    'Service_CeFi': 1.0,
    'Service_Chain': 1.0,
    'Service_DeFi': 1.0,
    'Service_GameFi': 1.0,
    'Service_Social': 1.0,
    'Service_Stablecoin': 1.0,
    'Funding Round_Angel': 1.0,
    'Funding Round_Pre-Seed': 1.0,
    'Funding Round_Pre-Series A': 1.0,
    'Funding Round_Seed': 1.0,
    'Funding Round_Series A': 1.0,
    'Funding Round_Strategic': 1.0,
    'Funding Round_Undisclosed': 1.0
}


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        top_n = request.form['text']
        # Perform some processing on the input text (e.g., sentiment analysis)
        # Replace this with your actual processing code
        top_indices = match(top_n=top_n, column_weights=column_weights)
        result = data.iloc[top_indices].values.tolist()
    return render_template('index.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=True)

