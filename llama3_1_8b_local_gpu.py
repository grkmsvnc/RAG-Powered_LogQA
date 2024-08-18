import os
import numpy as np
import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline
from sklearn.preprocessing import StandardScaler


os.environ["HF_TOKEN"] = "enter_your_hf_token"

hf_token = os.environ["HF_TOKEN"]  # HF Token now loaded from the environment variable

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
    """
    Numeric features (hour, status, size_mb) are passed dynamically based on the query context.
    """
    numeric_features = np.array([[hour, status, size_mb]])
    scaler = StandardScaler()
    normalized_numeric_features = scaler.fit_transform(numeric_features)
    return normalized_numeric_features

# 5. Prepare one-hot encoded 'day' feature for the query
def prepare_one_hot_day_for_query(day='Monday'):
    """
    Day feature for the query is passed dynamically based on the query context.
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    one_hot = np.zeros((1, len(days)))  # Create an array of zeros for one-hot encoding
    if day in days:
        day_index = days.index(day)
        one_hot[0, day_index] = 1  # Set the corresponding day to 1
    return one_hot

# 6. Combine the query embedding with numeric and one-hot encoded features
def combine_query_embedding_with_numeric_and_one_hot(query_embedding, hour, status, size_mb, day):
    """
    Combines text embedding from the query with numeric features (hour, status, size_mb) and
    one-hot encoded day feature.
    """
    # Prepare the numeric and one-hot features for the query
    numeric_features = prepare_numeric_features_for_query(hour, status, size_mb)
    one_hot_day = prepare_one_hot_day_for_query(day)

    # Combine the query embedding with numeric features and one-hot encoding
    combined_embedding = np.hstack([query_embedding, numeric_features, one_hot_day])
    return combined_embedding

# 7. Use the Sentence-Transformer model to encode the query and match the FAISS embedding size
def encode_query(query, model, expected_embedding_size, hour, status, size_mb, day):
    """
    Encodes the text query and combines it with the numeric and one-hot features to match
    the FAISS embedding size.
    """
    # Generate the query embedding (text-based)
    query_embedding = model.encode([query]).astype('float32')

    # Combine with numeric and one-hot features to match FAISS embedding size
    query_embedding = combine_query_embedding_with_numeric_and_one_hot(query_embedding, hour, status, size_mb, day)

    # Check if the embedding size matches FAISS index
    if query_embedding.shape[1] != expected_embedding_size:
        raise ValueError(f"Query embedding size {query_embedding.shape[1]} does not match FAISS index size {expected_embedding_size}")

    return query_embedding

# 8. Fetch and format the closest log entries
def format_log_entries(I, log_data):
    """
    Formats the log entries from the FAISS index to display in a user-friendly way.
    """
    closest_logs = []

    # Iterate through the FAISS indices and extract relevant fields for each log entry
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

# 9. Initialize the local Meta-Llama model
def initialize_llama_model():
    model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    config.rope_scaling = { "type": "linear", "factor": 8.0 }  # Adjust the factor as needed

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')

    text_generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )

    return text_generator

# 10. Generate a response using the local Meta-Llama model and log entries
def generate_response_with_llama(text_generator, log_entries, user_query):
    """
    Combines user query and relevant log entries to generate a response using the local Meta-Llama-3.1-8B-Instruct model.
    """
    # Structuring the log data into a clear, categorized format
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

    # Prompt enhancement for detailed task-oriented response
    prompt = f"""
    You are a log analysis assistant. Your task is to help the user understand the system's behavior based on the log data below.
    The user has asked the following question: "{user_query}".

    Please analyze the log data and provide a detailed answer. For each relevant log entry, explain how it might relate to the user's query.
    Also, provide any patterns or trends you observe across the logs.

    Relevant log entries:

    {log_content}

    Please provide your analysis below:
    """

    # Get response from Meta-Llama model
    sequences = text_generator(prompt)
    response_text = sequences[0]["generated_text"]

    return response_text

# 11. Full process: Encode query, search FAISS, fetch logs, and generate response using Meta-Llama model
def run_query_pipeline_with_llama(faiss_index, log_data, user_query, hour=12, status=200, size_mb=0.01, day='Monday', model_name='sentence-transformers/all-mpnet-base-v2', top_k=5):
    """
    Runs the full query pipeline: encodes the query, searches the FAISS index, fetches the closest log entries, and
    generates a response using the local Meta-Llama-3.1-8B-Instruct model.
    """
    # Load Sentence-Transformer model
    model = SentenceTransformer(model_name)

    # Get the FAISS index embedding size
    expected_embedding_size = faiss_index.d  # Dimension of embeddings in FAISS

    # Encode the query to match FAISS embedding size
    query_embedding = encode_query(user_query, model, expected_embedding_size, hour, status, size_mb, day)

    # Query FAISS for nearest log entries
    D, I = query_faiss_index(query_embedding, faiss_index, top_k)

    # Fetch and format the closest log entries
    closest_logs = format_log_entries(I, log_data)

    # Initialize Meta-Llama model for text generation using the HF Token from environment variable
    text_generator = initialize_llama_model()

    # Generate a response using the Meta-Llama model and the closest log entries
    response = generate_response_with_llama(text_generator, closest_logs, user_query)

    # Print the generated response
    print("Generated Response:\n", response)

# Example of running the query pipeline with Meta-Llama model integration
if __name__ == "__main__":
    # Load FAISS index and log data
    faiss_index = load_faiss_index('faiss_index.bin')
    log_data = pd.read_excel('/content/high_quality_log.xlsx')  # Load log data

    # Example user query
    user_query = "Why did the system return an error?"

    # Run the full query process with dynamic values for numeric features and day, and Meta-Llama model integration
    run_query_pipeline_with_llama(faiss_index, log_data, user_query, hour=14, status=404, size_mb=0.05, day='Tuesday')
