from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    education_level = Column(String)
    job_title = Column(String)
    years_of_experience = Column(Float)
    predicted_salary = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)