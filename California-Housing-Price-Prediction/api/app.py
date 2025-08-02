from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import numpy as np
import joblib
import mlflow
import os
import logging
import warnings
import time
from datetime import datetime
from functools import wraps

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and structured output"""
    
    def format(self, record):
        BOX_WIDTH = 100  # Consistent width for all boxes
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        day = datetime.now().strftime('%a')
        
        # Create a box around the log message
        message = record.getMessage()
        
        if "Starting" in message:
            header = f"┌{'─' * BOX_WIDTH}┐"
            time_line = f"│ ⏰ {timestamp:<19} ({day:<3}) {' ' * (BOX_WIDTH - 29)}│"
            message_line = f"│ {message}{' ' * (BOX_WIDTH - len(message) - 1)}│"
            footer = f"└{'─' * BOX_WIDTH}┘"
            formatted_msg = f"{header}\n{time_line}\n{message_line}\n{footer}"
        
        elif "Completed" in message:
            # Ensure consistent width for completed messages
            formatted_msg = f"│ {message}{' ' * (BOX_WIDTH - len(message) - 1)}│"
        
        elif "Prediction stats" in message:
            stats_header = f"├{'─' * BOX_WIDTH}┤"
            stats_footer = f"└{'─' * BOX_WIDTH}┘"
            
            # Format prediction stats with proper alignment
            stats_lines = message.split('\n')
            formatted_lines = []
            for line in stats_lines:
                if line.strip():  # Only process non-empty lines
                    padding = ' ' * (BOX_WIDTH - len(line) - 1)
                    formatted_lines.append(f"│ {line}{padding}│")
            
            formatted_msg = f"{stats_header}\n" + "\n".join(formatted_lines) + f"\n{stats_footer}"
        
        else:
            formatted_msg = f"│ {message}{' ' * (BOX_WIDTH - len(message) - 1)}│"
        
        return formatted_msg

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

def log_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Log start with decorated box
        logger.info(f"🚀 Starting {func.__name__}")
        
        result = await func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Performance indicators with consistent emojis
        if execution_time < 0.1:
            speed_indicator = "⚡ Super fast!"
            performance_emoji = "🌟"
            status = "Excellent ⭐"
        elif execution_time < 0.5:
            speed_indicator = "🚀 Fast!"
            performance_emoji = "✨"
            status = "Great ✨"
        else:
            speed_indicator = "👌 Good"
            performance_emoji = "�"
            status = "Good 👍"
            
        # Log completion with fixed-width formatting and better spacing
        completion_msg = (
            f"✅ Completed {func.__name__:<20} │ "
            f"{speed_indicator:<15} │ "
            f"⏱️ {execution_time:.3f}s {performance_emoji}"
        )
        logger.info(completion_msg)
        
        # Log detailed stats for predictions with consistent formatting
        if 'predict' in func.__name__.lower():
            stats_msg = (
                f"� Prediction Stats\n"
                f"    🎯 Function : {func.__name__}\n"
                f"    ⚡ Speed    : {execution_time:.3f}s\n"
                f"    ✨ Status   : {status}"
            )
            logger.info(stats_msg)
            
        return result
    return wrapper

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
MLFLOW_DIR = os.path.join(BASE_DIR, 'mlruns')

# Configure MLflow
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")

try:
    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load the latest model from local storage
    model_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    logger.info("✅ Model and scaler loaded successfully")
    
except Exception as e:
    logger.error(f"❌ Error loading model or scaler: {str(e)}")
    raise

# Initialize FastAPI
app = FastAPI(
    title="California Housing Price Predictor",
    description="Predict house prices using ML model with MLflow tracking",
    version="1.0.0"
)

# Setup templates directory with absolute path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# API input schema
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group (in tens of thousands)")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average occupancy")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

@app.get("/", response_class=HTMLResponse)
@log_execution_time
async def read_form(request: Request):
    """Render the initial form page"""
    return templates.TemplateResponse("form_minimal.html", {
        "request": request,
        "prediction": None,
        "now": None
    })

@app.post("/predict")
@log_execution_time
async def predict(features: HousingFeatures):
    """API endpoint for programmatic predictions"""
    # Feature engineering
    rooms_per_person = features.AveRooms / features.Population if features.Population > 0 else 0
    bedrooms_per_room = features.AveBedrms / features.AveRooms if features.AveRooms > 0 else 0

    # Input to array
    input_data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
        rooms_per_person,
        bedrooms_per_room
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    return {"predicted_price": round(float(pred), 3)}

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
    """Form-based prediction endpoint"""
    # Feature engineering
    rooms_per_person = AveRooms / Population if Population > 0 else 0
    bedrooms_per_room = AveBedrms / AveRooms if AveRooms > 0 else 0

    # Input to array
    input_data = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude,
        rooms_per_person,
        bedrooms_per_room
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Format current time
    current_time = datetime.now()
    return templates.TemplateResponse("form_minimal.html", {
        "request": request,
        "prediction": round(float(prediction), 3),
        "now": current_time.strftime("%Y-%m-%d %H:%M:%S %Z (%a)")
    })
