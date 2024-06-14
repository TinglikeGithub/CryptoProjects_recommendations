# Crypto Venture Scout

This repository contains Python code for matching textual descriptions with existing data using word embeddings and numerical features. The matching process involves training a Word2Vec model on textual descriptions, creating embeddings for textual data, combining embeddings with numerical features, reducing dimensionality using Singular Value Decomposition (SVD), and matching new data with existing data based on similarity.

## How to run

We have hosted our application on GCP at the following link: http://35.235.93.49/

The file src/recommendation_system.py takes different filters funding round, service, description as input from the user. Based on this filtered data, we create word embeddings for each data point using Gensim word2vec. Similarly, word embeddings are created for the remaining data points in our database, and the k nearest neighbours are returned as output, which are displayed to the user in our application.


## Dependencies:
1. Python 3.10
2. NumPy
3. Pandas==2.1.4
4. Gensim (for Word2Vec model)==4.3.2
5. scikit-learn (for SVD)==1.4.0
6. Flask==2.2.2
7. Selenium==4.18.1
8. webdriver-manager
9. werkzeug==2.2.3
10. scipy==1.9.0

## Requirements:
1. Access to processed textual description data, which we have scraped from websites like: https://cryptorank.io
2. Pre-trained Word2Vec embeddings (can be trained using Gensim or loaded from existing models).
3. Properly formatted input data (textual descriptions and numerical features) for matching.

## About the Authors

This project was developed by our team as part of the Entrepreneurship course at University of San Francisco!

Team Members:
1. Ting Pan
2. Indar Kumar
3. Rithvik Donnipadu 
4. Shrey Jain 
6. Sissi Shen




