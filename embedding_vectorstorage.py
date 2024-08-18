import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
import faiss
import os

# 1. Load the dataset from an Excel file
def load_data(file_path):
    return pd.read_excel(file_path)

# 2. Combine textual features for embedding creation
def prepare_text_features(log_data):
    text_features = log_data[['url', 'referrer', 'user_agent', 'method', 'cookie']].fillna('')
    log_data['timestamp_str'] = log_data['timestamp'].astype(str)  # Convert timestamp to string
    text_features['timestamp'] = log_data['timestamp_str']

    # Combine text features into a single string for each row
    return text_features.apply(lambda x: ' '.join(x), axis=1)

# 3. Generate text embeddings using a Sentence-Transformer model
def create_text_embeddings(model, text_features_combined):
    return model.encode(text_features_combined.tolist(), show_progress_bar=True)

# 4. Prepare and normalize numerical features
def prepare_numeric_features(log_data):
    numeric_features = log_data[['hour', 'status', 'size_mb']].fillna(0)
    scaler = StandardScaler()  # Normalize the numerical features
    return scaler.fit_transform(numeric_features)

# 5. One-hot encode the day column (for categorical data)
def one_hot_encode_day(log_data):
    return pd.get_dummies(log_data['day'])

# 6. Combine all embeddings (text embeddings + numerical embeddings + one-hot encoded day)
def combine_embeddings(text_embeddings, numeric_embeddings, day_one_hot):
    return np.hstack([text_embeddings, numeric_embeddings, day_one_hot.values])

# 7. Create a FAISS index and add embeddings
def create_faiss_index(embeddings):
    embeddings_float32 = embeddings.astype('float32')  # Ensure embeddings are float32 for FAISS
    embedding_size = embeddings_float32.shape[1]  # Get the size of embeddings
    index = faiss.IndexFlatL2(embedding_size)  # Use L2 (Euclidean distance) for FAISS index
    index.add(embeddings_float32)  # Add embeddings to the FAISS index
    return index

# 8. Save embeddings to a file
def save_embeddings(embeddings, file_name='embeddings.npy'):
    np.save(file_name, embeddings)  # Save embeddings as a .npy file
    print(f"Embeddings saved to {file_name}")

# 9. Save FAISS index to a file
def save_faiss_index(faiss_index, file_name='faiss_index.bin'):
    faiss.write_index(faiss_index, file_name)  # Save FAISS index to a binary file
    print(f"FAISS index saved to {file_name}")

# 10. Query the FAISS index to retrieve nearest neighbors
def query_faiss_index(query_embedding, faiss_index, top_k=5):
    query_embedding = np.array(query_embedding, dtype=np.float32)  # Ensure it's a float32 array
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D if it's 1D
    D, I = faiss_index.search(query_embedding, top_k)  # Search for the top k nearest neighbors
    return D, I

# 11. Complete pipeline function to run the entire process
def run_pipeline(file_path, model_name='sentence-transformers/all-mpnet-base-v2'):
    # Load the data
    log_data = load_data(file_path)

    # Prepare text features and generate text embeddings
    text_features_combined = prepare_text_features(log_data)
    model = SentenceTransformer(model_name)  # Load the Sentence-Transformer model
    text_embeddings = create_text_embeddings(model, text_features_combined)

    # Prepare numerical and categorical (day) features
    numeric_embeddings = prepare_numeric_features(log_data)
    day_one_hot = one_hot_encode_day(log_data)

    # Combine all embeddings
    final_embeddings = combine_embeddings(text_embeddings, numeric_embeddings, day_one_hot)
    print(f"Final embeddings shape: {final_embeddings.shape}")  # Print the shape of combined embeddings

    # Save the embeddings to a file
    save_embeddings(final_embeddings, 'final_embeddings.npy')  # Save embeddings to a file

    # Create and populate the FAISS index
    faiss_index = create_faiss_index(final_embeddings)
    print(f"Total of {faiss_index.ntotal} vectors added to FAISS.")  # Print the number of vectors in FAISS

    # Save the FAISS index to a file
    save_faiss_index(faiss_index, 'faiss_index.bin')  # Save FAISS index to a file

    # Example query - Using the first log entry as a query
    query_embedding = final_embeddings[0]  # Use the first embedding as the query (no reshape here)
    D, I = query_faiss_index(query_embedding, faiss_index)

    # Display the nearest neighbors found in the FAISS index
    print("Indices of nearest vectors found:", I)
    print("Distances to the nearest vectors:", D)

    # Display the closest log entry
    closest_index = I[0][0]  # Get the index of the closest log entry
    print("Closest log entry:", log_data.iloc[closest_index])  # Display the closest log entry

# To run the pipeline
file_path = '/content/high_quality_log.xlsx'
run_pipeline(file_path)
