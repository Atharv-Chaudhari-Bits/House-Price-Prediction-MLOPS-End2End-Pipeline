from fastapi.testclient import TestClient
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'California-Housing-Price-Prediction'))
sys.path.append(root_dir)

from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]