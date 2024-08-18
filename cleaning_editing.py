import re
import pandas as pd
import numpy as np

# Define the regular expression pattern for log parsing
log_pattern = (
    r'(?P<ip>\S+) - (?P<session_id>\S+) \[(?P<timestamp>.*?)\s*\] '
    r'"(?P<method>[A-Z]+) (?P<url>\S+) HTTP/1.1" '
    r'(?P<status>\d{3}) (?P<size>\d+) "(?P<referrer>.*?)" '
    r'"(?P<user_agent>.*?)" Cookie="(?P<cookie>\S+)"'
)

# Function to parse the log file into a DataFrame
def parse_log_file_from_text(log_data):
    parsed_data = []
    for entry in log_data:
        match = re.match(log_pattern, entry)
        if match:
            parsed_data.append(match.groupdict())

    df = pd.DataFrame(parsed_data)

    # Check if timestamp and referrer columns exist
    if 'timestamp' not in df.columns:
        print("Timestamp column missing in data.")
        return None

    if 'referrer' not in df.columns:
        print("Referrer column missing in data.")
        return None

    # Convert timestamp to datetime and clean it
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip(), format='%d/%b/%Y:%H:%M:%S', errors='coerce')

    return df

# Step 1: Parse the log data
file_path = '/content/diverse_synthetic_web_traffic.log'

with open(file_path, 'r') as file:
    log_data = file.readlines()

df_logs_parsed = parse_log_file_from_text(log_data)

# Display the first few rows to verify the parsing
print(df_logs_parsed.head())

# Ensure that 'size' and 'status' are numeric
df_logs_parsed['size'] = pd.to_numeric(df_logs_parsed['size'], errors='coerce')
df_logs_parsed['status'] = pd.to_numeric(df_logs_parsed['status'], errors='coerce')

# Convert timestamp to proper datetime format
df_logs_parsed['timestamp'] = pd.to_datetime(df_logs_parsed['timestamp'], errors='coerce')

# Check for missing values or inconsistencies
print(df_logs_parsed.info())

# Step 2: Save the cleaned log data into CSV and Excel formats
csv_output_path = 'cleaned_log.csv'
excel_output_path = 'cleaned_log.xlsx'

# Save as CSV
df_logs_parsed.to_csv(csv_output_path, index=False)

# Save as Excel
df_logs_parsed.to_excel(excel_output_path, index=False)

print(f"Cleaned data saved as CSV: {csv_output_path}")
print(f"Cleaned data saved as Excel: {excel_output_path}")

# Step 3: Load the cleaned data for further cleaning and quality checks
df = pd.read_excel('/content/cleaned_log.xlsx')

# 1. Handle missing values in critical columns
df_clean = df.dropna(subset=['ip', 'timestamp', 'method', 'url'])

# Fill missing values for non-critical fields with placeholders
df_clean['referrer'].fillna('Unknown', inplace=True)
df_clean['user_agent'].fillna('Unknown', inplace=True)

# 2. Validate IP addresses
def is_valid_ip(ip):
    try:
        ip_parts = ip.split('.')
        if len(ip_parts) != 4:
            return False
        return all(0 <= int(part) <= 255 for part in ip_parts)
    except ValueError:
        return False

df_clean['valid_ip'] = df_clean['ip'].apply(is_valid_ip)
df_clean = df_clean[df_clean['valid_ip']]  # Remove invalid IPs

# 3. Validate and clean timestamp
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
df_clean = df_clean.dropna(subset=['timestamp'])  # Remove rows with invalid timestamps

# 4. Validate HTTP status codes and request sizes
df_clean = df_clean[(df_clean['status'] >= 100) & (df_clean['status'] <= 599)]
df_clean['size'] = pd.to_numeric(df_clean['size'], errors='coerce')
df_clean = df_clean[(df_clean['size'] >= 0) & (df_clean['size'] < 10**9)]  # Remove unreasonable sizes

# 5. Only keep valid HTTP methods
valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']
df_clean = df_clean[df_clean['method'].isin(valid_methods)]

# 6. Extract hour and day from timestamp
df_clean['hour'] = df_clean['timestamp'].dt.hour
df_clean['day'] = df_clean['timestamp'].dt.day_name()

# Convert size to megabytes
df_clean['size_mb'] = df_clean['size'] / (1024 * 1024)

# 7. Identify and remove outliers using Z-score for size_mb
def remove_outliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = (df[column] - mean) / std
    return df[(z_scores > -3) & (z_scores < 3)]  # Keep only rows within 3 standard deviations

df_clean = remove_outliers(df_clean, 'size_mb')

# 8. Standardize HTTP methods (convert to uppercase for consistency)
df_clean['method'] = df_clean['method'].str.upper()

# 9. Reporting missing values after cleaning
missing_values_report = df_clean.isnull().sum()
print(f"Missing values after cleaning:\n{missing_values_report}")

# Report the number of rows cleaned and remaining
print(f"Cleaned row count: {len(df_clean)}")
print(f"Original row count: {len(df)}")

# Step 10: Save the final cleaned and processed data
df_clean.to_csv('high_quality_log.csv', index=False)
df_clean.to_excel('high_quality_log.xlsx', index=False)

print("Data successfully cleaned and saved.")
