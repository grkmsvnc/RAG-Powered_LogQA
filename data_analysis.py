import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file (update the file path as needed)
file_path = '/content/high_quality_log.xlsx'  # Update the file path
df = pd.read_excel(file_path)

# Convert the timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. HTTP Method Distribution
print("\n### HTTP Method Distribution ###\n")
print(df['method'].value_counts())  # Print the count of HTTP methods

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='method', palette='Set2')
plt.title('HTTP Method Distribution')
plt.xlabel('HTTP Method')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Top 10 Most Visited URLs
print("\n### Top 10 Most Visited URLs ###\n")
print(df['url'].value_counts().head(10))  # Print the top 10 most visited URLs

plt.figure(figsize=(10, 6))
df['url'].value_counts().head(10).plot(kind='bar', color='lightgreen')
plt.title('Top 10 Most Visited URLs')
plt.xlabel('URL')
plt.ylabel('Visit Count')
plt.xticks(rotation=45)  # Rotate X-axis labels by 45 degrees
plt.tight_layout()
plt.show()

# 3. Top 10 Most Frequent IP Addresses
print("\n### Top 10 Most Frequent IP Addresses ###\n")
print(df['ip'].value_counts().head(10))  # Print the top 10 IP addresses

plt.figure(figsize=(10, 6))
df['ip'].value_counts().head(10).plot(kind='bar', color='lightcoral')
plt.title('Top 10 Most Frequent IP Addresses')
plt.xlabel('IP Address')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate X-axis labels by 45 degrees
plt.tight_layout()
plt.show()

# 4. Traffic Distribution by Hour
traffic_by_hour = df.groupby('hour').size()
print("\n### Traffic Distribution by Hour ###\n")
print(traffic_by_hour)  # Print traffic distribution by hour

plt.figure(figsize=(10, 6))
traffic_by_hour.plot(kind='bar', color='skyblue')
plt.title('Traffic Distribution by Hour')
plt.xlabel('Hour')
plt.ylabel('Visit Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 5. Traffic Distribution by Day
traffic_by_day = df['day'].value_counts()
print("\n### Traffic Distribution by Day ###\n")
print(traffic_by_day)  # Print traffic distribution by day

plt.figure(figsize=(10, 6))
traffic_by_day.plot(kind='bar', color='orange')
plt.title('Traffic Distribution by Day')
plt.xlabel('Day')
plt.ylabel('Visit Count')
plt.xticks(rotation=45)  # Rotate X-axis labels by 45 degrees
plt.tight_layout()
plt.show()

# 6. Top 10 Most Used Browsers (user_agent)
print("\n### Top 10 Most Used Browsers ###\n")
print(df['user_agent'].value_counts().head(10))  # Print the top 10 user agents

# Rotate X-axis labels by 90 degrees for better readability
plt.figure(figsize=(10, 6))
df['user_agent'].value_counts().head(10).plot(kind='bar', color='purple')
plt.title('Top 10 Most Used Browsers (User-Agent)')
plt.xlabel('User-Agent')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate X-axis labels by 90 degrees
plt.tight_layout()
plt.show()

# 7. Top 10 Referring Sources (Referrers)
print("\n### Top 10 Referring Sources (Referrers) ###\n")
print(df['referrer'].value_counts().head(10))  # Print the top 10 referring sources

# Horizontal bar plot for referrers to handle long labels
plt.figure(figsize=(10, 6))
df['referrer'].value_counts().head(10).plot(kind='barh', color='teal')  # Horizontal bar chart
plt.title('Top 10 Referring Sources (Referrers)')
plt.xlabel('Count')
plt.ylabel('Referrer')
plt.tight_layout()
plt.show()
