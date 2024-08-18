import random
from datetime import datetime, timedelta
from faker import Faker

# Faker library for simulating geographic information, IP addresses, browser, and user data
fake = Faker()

# Components required for generating synthetic data
geo_ip_data = {
    "US": [f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(50)],
    "TR": [f"176.45.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(50)],
    "DE": [f"82.113.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(50)],
    "FR": [f"45.76.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(50)],
    "JP": [f"103.2.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(50)]
}

urls = ["/home", "/about", "/contact", "/product", "/login", "/dashboard", "/cart", "/checkout", "/search?q=item", "/product/view"]
http_methods = ["GET", "POST", "PUT", "DELETE"]
status_codes = [200, 301, 302, 404, 500]
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)"
]
# More diverse referrer URLs
referrers = [
    "https://google.com", "https://facebook.com", "https://twitter.com", "https://linkedin.com",
    "https://instagram.com", "https://reddit.com", "https://pinterest.com", "https://tumblr.com",
    "https://youtube.com", "https://news.ycombinator.com", "https://medium.com", "https://github.com",
    "-", "https://stackoverflow.com", "https://bing.com", "https://baidu.com",
    "https://yahoo.com", "https://quora.com", "https://t.co", "https://twitch.tv", "https://netflix.com"
]

byte_ranges = [random.randint(1000, 50000) for _ in range(1000)]

# Simulating traffic intensity (e.g., peak hours in the morning and evening)
def random_date():
    peak_hours = [9, 10, 11, 12, 18, 19, 20, 21]
    off_hours = [1, 2, 3, 4, 5]
    all_hours = peak_hours + off_hours + list(range(6, 24)) # Combine all hours
    # Adjust weights to match the number of elements in all_hours
    weights = [0.2] * len(peak_hours) + [0.1] * len(off_hours) + [0.7/len(range(6,24))] * len(range(6, 24))
    hour = random.choices(all_hours, weights=weights)[0]
    return datetime.now() - timedelta(days=random.randint(0, 30), hours=hour, minutes=random.randint(0, 59))

# Session and cookie information
def generate_session_id():
    return fake.uuid4()

def generate_cookie():
    return fake.md5()

# Generating a realistic log entry
def generate_log_entry():
    # Selecting geographic location and IP address
    country = random.choice(list(geo_ip_data.keys()))
    ip = random.choice(geo_ip_data[country])

    # HTTP request information
    method = random.choice(http_methods)
    url = random.choice(urls)
    status = random.choice(status_codes)
    user_agent = random.choice(user_agents)
    referrer = random.choice(referrers)
    byte_range = random.choice(byte_ranges)

    # Random timestamp
    timestamp = random_date().strftime('%d/%b/%Y:%H:%M:%S %z')

    # Session information
    session_id = generate_session_id()
    cookie = generate_cookie()

    # Creating a log entry in Apache/Nginx format
    log_entry = f'{ip} - {session_id} [{timestamp}] "{method} {url} HTTP/1.1" {status} {byte_range} "{referrer}" "{user_agent}" Cookie="{cookie}"'

    return log_entry

# Generate a specified number of log entries
def generate_log_file(file_name, num_entries=10000):
    log_data = [generate_log_entry() for _ in range(num_entries)]

    # Writing to file
    with open(file_name, "w") as file:
        for entry in log_data:
            file.write(entry + "\n")
    print(f"Synthetic log data saved to '{file_name}'.")

# Example usage
generate_log_file("diverse_synthetic_web_traffic.log", num_entries=10000)
