# G:\...\test_api.py

import requests
import json

# The URL of our running Docker container's API
url = 'http://127.0.0.1:5001/predict'

# A sample of raw, uncleaned data, just like a real client would send.
# The keys must match the original CSV column names.
sample_data = {
    "Campaign_Name": "Data Analytics Course",
    "Clicks": 150,
    "Impressions": 5000,
    "Cost": "$200",
    "Leads": 20,
    "Conversions": 10,
    "Ad_Date": "2025-07-24",
    "Location": "hyderabad",
    "Device": "desktop",
    "Keyword": "data analytics course"
}

print("--- Sending request to the API ---")
print("Data:", json.dumps(sample_data, indent=2))

# Send the POST request with the JSON data
response = requests.post(url, json=sample_data)

# Print the results from the API
print("\n--- Received response from the API ---")
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")