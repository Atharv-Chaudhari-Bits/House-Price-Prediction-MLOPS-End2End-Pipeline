import os
import warnings
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost.sklearn')
warnings.filterwarnings('ignore', category=Warning, message='.*artifact_path.*deprecated.*')

# Set up directory structure
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PARENT_DIR, 'Models')
DATA_DIR = os.path.join(PARENT_DIR, 'Data', 'processed')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Configure MLflow tracking and model registry
MLFLOW_DIR = os.path.join(PARENT_DIR, 'mlruns')
MLFLOW_TRACKING_URI = f'file:///{MLFLOW_DIR}'
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Set up MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(os.path.join(PARENT_DIR, 'mlflow_registry.db'))

print(f"🔍 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"📦 MLflow registry DB: {os.path.join(PARENT_DIR, 'mlflow_registry.db')}")

def load_and_prepare_data():
    """Load California housing dataset and prepare features for training"""
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Create additional features
    df["RoomsPerPerson"] = df["AveRooms"] / df["Population"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Scale features and save processed data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pd.DataFrame(X_scaled, columns=X.columns).to_csv(os.path.join(DATA_DIR, "X_processed.csv"), index=False)
    pd.DataFrame(y, columns=["MedHouseVal"]).to_csv(os.path.join(DATA_DIR, "y_processed.csv"), index=False)
    
    print(f"💾 Processed data saved to: {DATA_DIR}")
    return X_scaled, y, scaler

def train_model(X, y):
    """Train XGBoost model and return performance metrics"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return model, rmse, r2

def save_model_and_scaler(model, scaler):
    """Save trained model and scaler to disk"""
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    # Save both model and scaler using joblib for consistency
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"🤖 Model saved to: {model_path}")
    print(f"⚖️  Scaler saved to: {scaler_path}")
    return model_path, scaler_path

def main():
    """Main training pipeline with MLflow tracking"""
    X, y, scaler = load_and_prepare_data()
    
    # Set up experiment
    experiment_name = "CaliforniaHousing_XGBoost"
    mlflow.set_experiment(experiment_name)
    print(f"🧪 MLflow experiment: {experiment_name}")

    # Get existing runs to determine next version number
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        existing_versions = [int(name.split("v")[-1]) 
                           for name in runs["tags.mlflow.runName"].dropna() 
                           if name.startswith("XGBRegressor_v")]
        next_version = max(existing_versions, default=0) + 1
    else:
        next_version = 1

    run_name = f"XGBRegressor_v{next_version}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"🏃 Started MLflow run: {run.info.run_id}")
        
        model, rmse, r2 = train_model(X, y)
        
        # Log parameters, metrics, and artifacts
        params = {
            "model_type": "XGBoostRegressor",
            "test_size": 0.2,
            "random_state": 42
        }
        mlflow.log_params(params)
        print(f"⚙️  Logged parameters: {params}")
        
        metrics = {"rmse": rmse, "r2": r2}
        mlflow.log_metrics(metrics)
        print(f"📊 Logged metrics - RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
        model_path, scaler_path = save_model_and_scaler(model, scaler)
        
        # Log the model using MLflow
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model",
            input_example=X[:5]
        )
        print(f"🤖 Logged model: {model_info.model_uri}")
        
        # Log processed data
        mlflow.log_artifacts(DATA_DIR, artifact_path="data")
        print(f"� Logged artifacts to: {mlflow.get_artifact_uri()}")
        print(f"� View in UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main()
