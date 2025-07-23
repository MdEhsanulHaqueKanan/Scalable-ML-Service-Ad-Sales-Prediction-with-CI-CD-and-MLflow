import requests
import json

# URL of our running Flask API
url = 'http://127.0.0.1:5001/predict'

# Sample data to send. The keys must match the column names from the *original* raw data.
# We will build a more realistic sample later.
sample_data = {
    "Campaign_Name": "Data Analytics Course",
    "Clicks": 150,
    "Impressions": 5000,
    "Cost": "$200",
    "Leads": 20,
    "Conversions": 10,
    "Ad_Date": "2024-11-25",
    "Location": "hyderabad",
    "Device": "desktop",
    "Keyword": "data analytics course"
}

# Send the POST request
response = requests.post(url, json=sample_data)

# Print the results
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")