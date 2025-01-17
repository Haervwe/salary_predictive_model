from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, field_validator
import pandas as pd
import uvicorn
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import os

# Import your inference functions from src/inference.py
from src.inference import (
    make_inference_nn, 
    get_unique_job_titles,
    load_scaler,
    load_target_encoder,
    load_model_nn
)

from src.database import engine, get_db, SQLALCHEMY_DATABASE_URL
from src.db_model import Base, Prediction

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


# Function to check and create database if not exists
def ensure_database_exists():
    # For SQLite, check if the database file exists
    if 'sqlite:///' in SQLALCHEMY_DATABASE_URL:
        db_path = SQLALCHEMY_DATABASE_URL.replace('sqlite:///', '')
        if not os.path.exists(db_path):
            # Create the database and tables
            Base.metadata.create_all(bind=engine)
            print(f"Database created at {db_path}")
    else:
        # For other database types, create tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created")

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    ensure_database_exists()
    
    # Startup: Load models and other resources
    prefix = ""
    app.state.scaler = load_scaler(prefix=prefix)
    app.state.te = load_target_encoder(prefix=prefix)
    app.state.model_nn = load_model_nn(prefix=prefix)
    app.state.unique_job_titles = get_unique_job_titles(prefix=prefix)
    yield
    # Cleanup: Release resources if needed
    app.state.scaler = None
    app.state.te = None
    app.state.model_nn = None
    app.state.unique_job_titles = None

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict_salary(input_data: InputData, request: Request, db: Session = Depends(get_db)):
    # Validate job title
    if input_data.job_title not in request.app.state.unique_job_titles:
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
            scaler=request.app.state.scaler,
            te=request.app.state.te,
            model_nn=request.app.state.model_nn
        )
        predicted_salary = prediction[0][0]

        # Store prediction in database
        db_prediction = Prediction(
            age=input_data.age,
            education_level=input_data.education_level,
            job_title=input_data.job_title,
            years_of_experience=input_data.years_of_experience,
            predicted_salary=predicted_salary
        )
        db.add(db_prediction)
        db.commit()

        return {"predicted_salary": round(float(predicted_salary), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job_titles")
async def get_job_titles(request: Request):
    return {"job_titles": request.app.state.unique_job_titles}

@app.get("/predictions")
async def get_predictions(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 10
):
    predictions = db.query(Prediction)\
        .order_by(Prediction.timestamp.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return predictions

@app.get("/health_check")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9988)