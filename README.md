# Salary Predictive Model

## Overview
This project implements a machine learning pipeline for salary prediction based on features like education level, job title, age and years of experience. It provides both a Gradio web interface and a FastAPI backend for making predictions.

## Quick Start

1. Clone and setup:
    ```sh
    git clone https://github.com/yourusername/salary_predictive_model.git
    cd salary_predictive_model
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. To make predictions, use either:

    a. Web Interface (Recommended):
    ```sh
    python app.py
    ```
    Then open http://localhost:7860 in your browser

    b. REST API:
    ```sh 
    python api.py
    ```
    The API will be available at http://localhost:9988

## Project Structure

### Documentation & Analysis
- **analysis.ipynb**: Jupyter notebook containing:
  - Data exploration and visualization
  - Model training process
  - Performance evaluation
  - Feature importance analysis
  - Comparative analysis of different models

### Core Modules
- **llm_dataset_filler.py**: Handles missing value imputation using a local LLM (Hermes 3) via Ollama
- **feature_engineering.py**: Implements feature scaling, encoding and selection
- **modeling.py**: Contains model training code for both Random Forest and Neural Network approaches
- **evaluation.py**: Provides metrics calculation and visualization functions
- **inference.py**: Core inference logic used by both UI and API

### Interfaces
- **app.py**: Gradio web interface for making predictions
- **api.py**: FastAPI backend service
- **inference_jupyter_form.py**: Interactive form widget for Jupyter

### Database
- **database.py**: SQLAlchemy database setup
- **db_model.py**: SQLAlchemy models
- **init_db.py**: Database initialization

## Key Features

- Missing value imputation using local LLM
- Feature engineering pipeline
- Model experimentation workflow
- Interactive web interface
- REST API endpoint
- Model performance analysis
- Prediction history tracking
- Comprehensive documentation

## Model Performance

The Neural Network model achieved:

- MSE: 417409892.441 (95% CI: [192168452.152, 798936099.324])
- MAE: 13728.422 (95% CI: [10844.958, 17243.518])
- R2: 0.834 (95% CI: [0.724, 0.911])

Outperforming both Random Forest and baseline models.

## API Documentation

Once running, API documentation is available at:
- Swagger UI: http://localhost:9988/docs
- ReDoc: http://localhost:9988/redoc

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
