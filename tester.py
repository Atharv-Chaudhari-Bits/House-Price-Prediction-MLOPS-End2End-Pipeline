import requests

url = "http://localhost:8000/predict"

payload = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print("✅ API responded successfully!")
    print("➡️ Input:", payload)
    print("📦 Response:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ Error communicating with the API:")
    print(e)
