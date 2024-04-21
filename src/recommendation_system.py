import pandas as pd
from gensim.models import Word2Vec
import numpy as np

all_data = pd.read_csv("Data/processed_data.csv")
test_data = all_data[-3:]
processed_data = all_data[:-3]

texts = [text.split() for text in processed_data['Inverstors_and_desc']]
embedding_dim = 100  # Adjust as needed
word2vec_model = Word2Vec(texts, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# Create embeddings for textual descriptions
def create_embeddings(description):
    words = description.split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)

# Create feature matrix with embeddings and numerical features
feature_matrix = []
for index, entry in processed_data.iterrows():
    embeddings = create_embeddings(entry['Inverstors_and_desc'])
    numerical_features = [
        entry['Total Raised'],
        entry['First Funding Year'],
        entry['First Funding Month'],
        entry['First Funding Day'],
        entry['Service_Blockchain Infrastructure'],
        entry['Service_Blockchain Service'],
        entry['Service_CeFi'],
        entry['Service_Chain'],
        entry['Service_DeFi'],
        entry['Service_GameFi'],
        entry['Service_Social'],
        entry['Service_Stablecoin'],
        entry['Funding Round_Angel'],
        entry['Funding Round_Pre-Seed'],
        entry['Funding Round_Pre-Series A'],
        entry['Funding Round_Seed'],
        entry['Funding Round_Series A'],
        entry['Funding Round_Strategic'],
        entry['Funding Round_Undisclosed']
    ]
    feature_vector = np.concatenate([embeddings, numerical_features])
    feature_matrix.append(feature_vector)

feature_matrix = np.array(feature_matrix)

# Perform SVD on the feature matrix
U, sigma, Vt = np.linalg.svd(feature_matrix)

# Determine the value of k based on explained variance ratio
explained_variance_ratio = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
threshold = 0.95  # Adjust as needed
k = np.argmax(explained_variance_ratio >= threshold) + 1

# Reconstruction of feature matrix
reconstructed_feature_matrix = np.dot(U[:, :k], np.dot(np.diag(sigma[:k]), Vt[:k, :]))

def match(top_n, column_weights, description):
    top_n = int(top_n)
    all_top_indices = []
    all_top_similarities = []
    important_columns_per_match = []
    embeddings = create_embeddings(description)
    numerical_features = [
        entry['Total Raised'] * column_weights['Total Raised'],
        entry['First Funding Year'] * column_weights['First Funding Year'],
        entry['First Funding Month'] * column_weights['First Funding Month'],
        entry['First Funding Day'] * column_weights['First Funding Day'],
        entry['Service_Blockchain Infrastructure'] * column_weights['Service_Blockchain Infrastructure'],
        entry['Service_Blockchain Service'] * column_weights['Service_Blockchain Service'],
        entry['Service_CeFi'] * column_weights['Service_CeFi'],
        entry['Service_Chain'] * column_weights['Service_Chain'],
        entry['Service_DeFi'] * column_weights['Service_DeFi'],
        entry['Service_GameFi'] * column_weights['Service_GameFi'],
        entry['Service_Social'] * column_weights['Service_Social'],
        entry['Service_Stablecoin'] * column_weights['Service_Stablecoin'],
        entry['Funding Round_Angel'] * column_weights['Funding Round_Angel'],
        entry['Funding Round_Pre-Seed'] * column_weights['Funding Round_Pre-Seed'],
        entry['Funding Round_Pre-Series A'] * column_weights['Funding Round_Pre-Series A'],
        entry['Funding Round_Seed'] * column_weights['Funding Round_Seed'],
        entry['Funding Round_Series A'] * column_weights['Funding Round_Series A'],
        entry['Funding Round_Strategic'] * column_weights['Funding Round_Strategic'],
        entry['Funding Round_Undisclosed'] * column_weights['Funding Round_Undisclosed']
    ]
    feature_vector = np.concatenate([embeddings, numerical_features])
    
    # Calculate similarities between the new data and existing data
    similarities = np.dot(reconstructed_feature_matrix, feature_vector.T)
    
    # Find top n similar services
    top_indices = np.argsort(similarities)[::-1][:int(top_n)]
    top_similarities = similarities[top_indices]
    
    all_top_indices.extend(top_indices)
    all_top_similarities.extend(top_similarities)

    # Find top n unique indices based on the top n similarity scores
    sorted_indices = np.argsort(all_top_similarities)[::-1]
    unique_top_indices = [all_top_indices[idx] for idx in sorted_indices][:top_n]
    
    return unique_top_indices


# Call the match function with column weights
# print("Top 5 Matches for the new data (indices):", top_indices)
