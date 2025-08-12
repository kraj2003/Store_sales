from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import numpy as np

# Global variable to store the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code - load the model
    global model
    try:
        model = joblib.load('./models/best_model.pkl')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"No model found. Train the model first. Error: {e}")
    
    yield  # This is where the app runs
    
    # Shutdown code (if needed
    print("Shutting down...")

app = FastAPI(
    title="Sales Forecasting API",
    lifespan=lifespan
)

@app.get("/")
def home():
    return {"message": "Sales Forecasting API", "status": "running"}

@app.get("/predict")
def predict_sales(
    year: int = 2023,
    month: int = 9,
    day: int = 15,
    dayofweek: int = 4,
    is_weekend: int = 0,
    sales_yesterday: float = 50000,
    sales_last_week: float = 48000
):
    """Predict sales for given features"""
    if model is None:
        return {"error": "Model not loaded"}
    
    # Create feature array
    features = np.array([[year, month, day, dayofweek, is_weekend, 
                         sales_yesterday, sales_last_week]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    prediction = max(prediction, 0)  # No negative sales
    
    return {
        "predicted_sales": round(prediction, 2),
        "date_info": f"{year}-{month:02d}-{day:02d}",
        "is_weekend": bool(is_weekend)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)