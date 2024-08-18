import os
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from groq import Groq

# Set the API key for Groq
os.environ["GROQ_API_KEY"] = "enter_your_groq_token"

# 1. Load FAISS index from a file
def load_faiss_index(file_name='faiss_index.bin'):
    faiss_index = faiss.read_index(file_name)
    print(f"FAISS index loaded from {file_name}")
    return faiss_index

# 2. Load embeddings from a file
def load_embeddings(file_name='embeddings.npy'):
    embeddings = np.load(file_name)
    print(f"Embeddings loaded from {file_name}")
    return embeddings

# 3. Query the FAISS index to retrieve nearest neighbors
def query_faiss_index(query_embedding, faiss_index, top_k=5):
    query_embedding = np.array(query_embedding, dtype=np.float32)  # Ensure it's a float32 array
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)  # Reshape to 2D if it's 1D
    D, I = faiss_index.search(query_embedding, top_k)  # Search for the top k nearest neighbors
    return D, I

# 4. Prepare numeric features for the query (hour, status, size_mb)
def prepare_numeric_features_for_query(hour=12, status=200, size_mb=0.01):
    numeric_features = np.array([[hour, status, size_mb]])
    scaler = StandardScaler()
    normalized_numeric_features = scaler.fit_transform(numeric_features)
    return normalized_numeric_features

# 5. Prepare one-hot encoded 'day' feature for the query
def prepare_one_hot_day_for_query(day='Monday'):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    one_hot = np.zeros((1, len(days)))  # Create an array of zeros for one-hot encoding
    if day in days:
        day_index = days.index(day)
        one_hot[0, day_index] = 1  # Set the corresponding day to 1
    return one_hot

# 6. Combine the query embedding with numeric and one-hot encoded features
def combine_query_embedding_with_numeric_and_one_hot(query_embedding, hour, status, size_mb, day):
    # Prepare the numeric and one-hot features for the query
    numeric_features = prepare_numeric_features_for_query(hour, status, size_mb)
    one_hot_day = prepare_one_hot_day_for_query(day)

    # Combine the query embedding with numeric features and one-hot encoding
    combined_embedding = np.hstack([query_embedding, numeric_features, one_hot_day])
    return combined_embedding

# 7. Use the Sentence-Transformer model to encode the query and match the FAISS embedding size
def encode_query(query, model, expected_embedding_size, hour, status, size_mb, day):
    query_embedding = model.encode([query]).astype('float32')

    # Combine with numeric and one-hot features to match FAISS embedding size
    query_embedding = combine_query_embedding_with_numeric_and_one_hot(query_embedding, hour, status, size_mb, day)

    if query_embedding.shape[1] != expected_embedding_size:
        raise ValueError(f"Query embedding size {query_embedding.shape[1]} does not match FAISS index size {expected_embedding_size}")

    return query_embedding

# 8. Fetch and format the closest log entries
def format_log_entries(I, log_data):
    closest_logs = []
    for idx in I[0]:  # Iterate over the nearest neighbors
        log_entry = log_data.iloc[idx]
        log_content = {
            'url': log_entry['url'],
            'method': log_entry['method'],
            'timestamp': log_entry['timestamp'],
            'status': log_entry['status'],
            'referrer': log_entry['referrer'],
            'user_agent': log_entry['user_agent'],
            'cookie': log_entry['cookie']
        }
        closest_logs.append(log_content)
    return closest_logs

# 9. Initialize Groq API client
def initialize_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 10. Generate a response using the Groq API and log entries with enhanced prompt
def generate_response_with_groq(client, log_entries, user_query):
    log_content = "\n\n".join([
        f"Log {idx + 1}:\n"
        f"URL: {log_entry['url']}\n"
        f"Method: {log_entry['method']}\n"
        f"Timestamp: {log_entry['timestamp']}\n"
        f"Status: {log_entry['status']}\n"
        f"Referrer: {log_entry['referrer']}\n"
        f"User-Agent: {log_entry['user_agent']}\n"
        f"Cookie: {log_entry['cookie']}\n"
        for idx, log_entry in enumerate(log_entries)
    ])

    prompt = f"""
    You are a log analysis assistant. Your task is to help the user understand the system's behavior based on the log data below.
    The user has asked the following question: "{user_query}".

    Please analyze the log data and provide a detailed answer. For each relevant log entry, explain how it might relate to the user's query.
    Also, provide any patterns or trends you observe across the logs.

    Relevant log entries:

    {log_content}

    Please provide your analysis below:
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama3-70b-8192"
    )

    return chat_completion.choices[0].message.content

# 11. Full process: Encode query, search FAISS, fetch logs, and generate response using Groq API
def run_query_pipeline_with_groq(faiss_index, log_data, user_query, hour=12, status=200, size_mb=0.01, day='Monday', top_k=5):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    expected_embedding_size = faiss_index.d

    query_embedding = encode_query(user_query, model, expected_embedding_size, hour, status, size_mb, day)
    D, I = query_faiss_index(query_embedding, faiss_index, top_k)
    closest_logs = format_log_entries(I, log_data)

    client = initialize_groq_client()
    response = generate_response_with_groq(client, closest_logs, user_query)

    return response

# User-friendly system for queries
def user_friendly_system():
    # Load FAISS index and log data
    faiss_index = load_faiss_index('faiss_index.bin')
    log_data = pd.read_excel('/content/high_quality_log.xlsx')

    print("Welcome to the Log Analysis System!\n")

    while True:
        print("Please enter your query (type 'exit' to quit):")
        user_query = input("> ")

        if user_query.lower() == "exit":
            print("Exiting the system...")
            break

        # Ask for additional parameters, and if left blank, default values will be used
        print("Select the query hour (e.g., 14) [Default: 12] (If you don't want to specify, leave it blank):")
        hour_input = input("> ")
        hour = int(hour_input) if hour_input.strip() else 12

        print("Enter the HTTP status code (e.g., 200 or 404) [Default: 200] (If you don't want to specify, leave it blank):")
        status_input = input("> ")
        status = int(status_input) if status_input.strip() else 200

        print("Enter the request size in MB (e.g., 0.01) [Default: 0.01] (If you don't want to specify, leave it blank):")
        size_mb_input = input("> ")
        size_mb = float(size_mb_input) if size_mb_input.strip() else 0.01

        print("Select the day of the week (e.g., Monday) [Default: Monday] (If you don't want to specify, leave it blank):")
        day_input = input("> ")
        day = day_input if day_input.strip() else 'Monday'

        print("How many nearest log entries do you want to retrieve? (e.g., 5) [Default: 5] (If you don't want to specify, leave it blank):")
        top_k_input = input("> ")
        top_k = int(top_k_input) if top_k_input.strip() else 5

        # Run the query based on user input
        response = run_query_pipeline_with_groq(faiss_index, log_data, user_query, hour, status, size_mb, day, top_k)

        # Display the result to the user
        print("\nHere is the response to your query:")
        print(response)
        print("\n---------------------------\n")

# Start the CLI
if __name__ == "__main__":
    user_friendly_system()
