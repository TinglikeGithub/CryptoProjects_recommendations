import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.neighbors import NearestNeighbors

all_data = pd.read_csv("Data/processed_data.csv")

texts = [text.split() for text in all_data['Inverstors_and_desc']]
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

def feature_matrix(processed_data, column_weights, embeddings_user, top_n):
    # Create feature matrix with embeddings and numerical features
    feature_matrix = []
    index = []
    for index_, entry in processed_data.iterrows():
        index.append(index_)
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

    knn_model = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='auto')
    knn_model.fit(feature_matrix)

    # # Perform SVD on the feature matrix
    # U, sigma, Vt = np.linalg.svd(feature_matrix)

    # # Determine the value of k based on explained variance ratio
    # explained_variance_ratio = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
    # threshold = 0.95  # Adjust as needed
    # k = np.argmax(explained_variance_ratio >= threshold) + 1

    numerical_features = [
    # entry['Total Raised'] * column_weights['Total Raised'],
    # entry['First Funding Year'] * column_weights['First Funding Year'],
    # entry['First Funding Month'] * column_weights['First Funding Month'],
    # entry['First Funding Day'] * column_weights['First Funding Day'],
    0,0,0,0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    entry['Funding Round_Angel'] * column_weights['Funding Round_Angel'],
    entry['Funding Round_Pre-Seed'] * column_weights['Funding Round_Pre-Seed'],
    entry['Funding Round_Pre-Series A'] * column_weights['Funding Round_Pre-Series A'],
    entry['Funding Round_Seed'] * column_weights['Funding Round_Seed'],
    entry['Funding Round_Series A'] * column_weights['Funding Round_Series A'],
    entry['Funding Round_Strategic'] * column_weights['Funding Round_Strategic'],
    entry['Funding Round_Undisclosed'] * column_weights['Funding Round_Undisclosed']
    ]
    
    feature_vector = np.concatenate([embeddings_user, numerical_features])

    # Reshape the feature vector to fit the KNN model
    feature_vector = feature_vector.reshape(1, -1)


    # Find the indices of the nearest neighbors
    _, top_indices = knn_model.kneighbors(feature_vector, n_neighbors=top_n)
    
    all_top_indices = top_indices.flatten()
    
    return [index[i] for i in all_top_indices]



    # # Reconstruction of feature matrix
    # reconstructed_feature_matrix = np.dot(U[:, :k], np.dot(np.diag(sigma[:k]), Vt[:k, :]))
    # return reconstructed_feature_matrix, feature_vector



def funding_mapping(funding_round):
    mapping = {
        "angel": "Funding Round_Angel",
        "pre_seed": "Funding Round_Pre-Seed",
        "pre_series_a": "Funding Round_Pre-Series A",
        "seed": "Funding Round_Seed",
        "series_a": "Funding Round_Series A",
        "strategic": "Funding Round_Strategic",
        "undisclosed": "Funding Round_Undisclosed"
    }
    return mapping.get(funding_round, "")


def match(processed_data, top_n, column_weights, description):
    top_n = int(top_n)
    all_top_indices = []
    all_top_similarities = []
    important_columns_per_match = []
    if len(description)>5:
        embeddings = create_embeddings(description)
    else:
        embeddings = np.zeros((100))

    # if funding_round:
    #     processed_data = processed_data[processed_data[funding_round] == 1]

    all_top_indices = feature_matrix(processed_data,
                                     column_weights,
                                     embeddings_user=embeddings,
                                     top_n=top_n)
    
    # reconstructed_feature_matrix, feature_vector = feature_matrix(processed_data,
    #                                                               column_weights,
    #                                                               embeddings_user=embeddings)
    
    # # Calculate similarities between the new data and existing data
    # similarities = np.dot(reconstructed_feature_matrix, feature_vector.T)
    
    # # Find top n similar services
    # top_indices = np.argsort(similarities)[::-1][:int(top_n)]
    # top_similarities = similarities[top_indices]
    
    # all_top_indices.extend(top_indices)
    # all_top_similarities.extend(top_similarities)

    # # Find top n unique indices based on the top n similarity scores
    # sorted_indices = np.argsort(all_top_similarities)[::-1]
    # unique_top_indices = [all_top_indices[idx] for idx in sorted_indices][:top_n]
    return all_top_indices


# Call the match function with column weights
# print("Top 5 Matches for the new data (indices):", top_indices)
