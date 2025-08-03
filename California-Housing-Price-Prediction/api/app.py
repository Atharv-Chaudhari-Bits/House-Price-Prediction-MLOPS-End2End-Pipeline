from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import mlflow
import os
import logging
import warnings
import time
import json
from datetime import datetime
from functools import wraps
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_format = logging.Formatter("📅 %(asctime)s | 🔖 %(levelname)s | ✏️ %(message)s", "%Y-%m-%d %H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_logs.log")
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(log_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- Prometheus Metrics ---
total_requests_counter = Counter("total_requests", "Total number of incoming requests")
total_predictions_counter = Counter("total_predictions", "Total number of model predictions")

# --- Decorator for timing ---
def log_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"🚀 Started `{func.__name__}`")
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"✅ Completed `{func.__name__}` in ⏱️ {duration:.3f}s")
        if 'predict' in func.__name__:
            logger.info(f"📤 Prediction endpoint completed in ⏱️ {duration:.3f}s")
        return result
    return wrapper

# --- Model Load ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
MLFLOW_DIR = os.path.join(BASE_DIR, 'mlruns')
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    logger.info("✅ Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model/scaler: {e}")
    raise

# --- FastAPI App ---
app = FastAPI(
    title="California Housing Price Predictor",
    description="Predict house prices using ML model with MLflow tracking",
    version="1.0.0"
)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Input Schema ---
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# --- Homepage Form (GET) ---
@app.get("/", response_class=HTMLResponse)
@log_execution_time
async def read_form(request: Request):
    total_requests_counter.inc()
    return templates.TemplateResponse("form_minimal.html", {
        "request": request,
        "prediction": None,
        "now": None
    })

# --- JSON Prediction Endpoint ---
@app.post("/predict")
@log_execution_time
async def predict(features: HousingFeatures):
    total_predictions_counter.inc()
    input_dict = features.dict()

    rooms_per_person = input_dict["AveRooms"] / input_dict["Population"] if input_dict["Population"] > 0 else 0
    bedrooms_per_room = input_dict["AveBedrms"] / input_dict["AveRooms"] if input_dict["AveRooms"] > 0 else 0

    input_array = np.array([[input_dict["MedInc"], input_dict["HouseAge"], input_dict["AveRooms"], input_dict["AveBedrms"],
                             input_dict["Population"], input_dict["AveOccup"], input_dict["Latitude"], input_dict["Longitude"],
                             rooms_per_person, bedrooms_per_room]])

    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]

    logger.info("🧾 Incoming JSON request:\n" + json.dumps(input_dict, indent=4))
    logger.info("💰 Prediction Output:\n" + json.dumps({"predicted_price": round(float(pred), 3)}, indent=4))

    return JSONResponse(content={"predicted_price": round(float(pred), 3)})

# --- HTML Form Prediction Endpoint ---
@app.post("/form", response_class=HTMLResponse)
@log_execution_time
async def predict_from_form(
    request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):
    total_requests_counter.inc()
    total_predictions_counter.inc()

    input_dict = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    rooms_per_person = AveRooms / Population if Population > 0 else 0
    bedrooms_per_room = AveBedrms / AveRooms if AveRooms > 0 else 0

    input_array = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population,
                             AveOccup, Latitude, Longitude, rooms_per_person, bedrooms_per_room]])

    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    logger.info("🧾 Form input values:\n" + json.dumps(input_dict, indent=4))
    logger.info("💰 Prediction Output:\n" + json.dumps({"predicted_price": round(float(prediction), 3)}, indent=4))

    return templates.TemplateResponse("form_minimal.html", {
        "request": request,
        "prediction": round(float(prediction), 3),
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# --- Prometheus Metrics Endpoint ---
@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)




from fastapi import UploadFile, File
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

@app.get("/retrain", response_class=HTMLResponse)
@log_execution_time
async def retrain_form(request: Request):
    return templates.TemplateResponse("retrain_upload.html", {"request": request, "message": None, "data": None})


@app.post("/retrain", response_class=HTMLResponse)
@log_execution_time
async def retrain_model(request: Request, file: UploadFile = File(...)):
    total_requests_counter.inc()
    try:
        df = pd.read_csv(file.file)
        logger.info("📦 New training data received:\n" + str(df.head(3)))

        required_cols = {"MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"}
        if not required_cols.issubset(set(df.columns)):
            return templates.TemplateResponse("retrain_upload.html", {
                "request": request,
                "message": "❌ Uploaded CSV is missing required columns.",
                "data": None
            })

        df["rooms_per_person"] = df["AveRooms"] / df["Population"].replace(0, np.nan)
        df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
        df.fillna(0, inplace=True)

        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]

        scaler_new = StandardScaler()
        X_scaled = scaler_new.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        with mlflow.start_run(run_name="model_retraining"):
            model_new = XGBRegressor(n_estimators=100, random_state=42)
            model_new.fit(X_train, y_train)

            val_score = model_new.score(X_val, y_val)
            mlflow.log_metric("val_r2", val_score)

            # Save updated model + scaler
            joblib.dump(model_new, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
            joblib.dump(scaler_new, os.path.join(MODEL_DIR, 'scaler.pkl'))
            logger.info(f"📈 Model retrained with R2 score = {val_score:.3f}")
            logger.info("💾 Model and scaler updated successfully.")

        return templates.TemplateResponse("retrain_upload.html", {
            "request": request,
            "message": f"✅ Model retrained successfully! R² on validation: {val_score:.3f}",
            "data": df.head(5).to_html(classes="table table-bordered")
        })

    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        return templates.TemplateResponse("retrain_upload.html", {
            "request": request,
            "message": "❌ Error occurred during retraining.",
            "data": None
        })
