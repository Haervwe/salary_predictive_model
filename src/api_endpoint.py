from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd

# Import your inference functions from src/inference.py
from src.inference import (
    make_inference_nn, 
    preprocess_inference_data, 
    get_unique_job_titles
)

# Initialize the FastAPI app
app = FastAPI()

# Global variables to hold models and preprocessors
scaler = None
te = None
model_nn = None
unique_job_titles = []
prefix = ""  # You can set a specific prefix if needed

# Event handler to load models and preprocessors at startup
@app.on_event("startup")
def load_models():
    global scaler, te, model_nn, unique_job_titles, prefix

    # Load the scaler, target encoder, and model within the inference module
    from src.inference import load_scaler, load_target_encoder, load_model_nn

    # Load the scaler
    try:
        scaler = load_scaler(prefix=prefix)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        scaler = None

    # Load the target encoder
    try:
        te = load_target_encoder(prefix=prefix)
    except Exception as e:
        print(f"Error loading target encoder: {e}")
        te = None

    # Load the trained Neural Network model
    try:
        model_nn = load_model_nn(prefix=prefix)
    except Exception as e:
        print(f"Error loading neural network model: {e}")
        model_nn = None

    # Load unique job titles
    try:
        unique_job_titles = get_unique_job_titles(prefix=prefix)
    except Exception as e:
        print(f"Error loading job titles: {e}")
        unique_job_titles = []

# Pydantic model for request validation
class InputData(BaseModel):
    age: int
    education_level: str
    job_title: str
    years_of_experience: float

    @field_validator('education_level')
    def validate_education_level(cls, v):
        if v not in ["Bachelor's", "Master's", "PhD"]:
            raise ValueError("Invalid education level")
        return v

# Endpoint to perform prediction
@app.post("/predict")
async def predict_salary(input_data: InputData):
    # Validate job title
    if unique_job_titles and input_data.job_title not in unique_job_titles:
        raise HTTPException(status_code=400, detail=f"Invalid job title: {input_data.job_title}")

    # Create a DataFrame from the input data
    data = pd.DataFrame({
        "Age": [input_data.age],
        "Education Level": [input_data.education_level],
        "Job Title": [input_data.job_title],
        "Years of Experience": [input_data.years_of_experience],
    })
    
    try:
        # Make prediction
        prediction = make_inference_nn(
            input_data=data,
            prefix=prefix,
            scaler=scaler,
            te=te,
            model_nn=model_nn
        )
        predicted_salary = prediction[0][0]
        # Return the prediction as JSON
        return {"predicted_salary": round(float(predicted_salary), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get the list of available job titles
@app.get("/job_titles")
async def get_job_titles():
    return {"job_titles": unique_job_titles}