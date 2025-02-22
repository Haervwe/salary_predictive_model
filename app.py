import gradio as gr
import requests
import json
import uvicorn
import multiprocessing
import time

# FastAPI endpoint URL
API_URL = "http://localhost:9988"

# Import your FastAPI app
from api import app  # Assuming your FastAPI script is named fastapi_app.py

# FastAPI endpoint URL
API_URL = "http://localhost:9988"

def run_fastapi_server():
    """Function to run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=9988)

# Get unique job titles from the API
def get_job_titles():
    response = requests.get(f"{API_URL}/job_titles")
    if response.status_code == 200:
        return response.json()["job_titles"]
    return []

#wait helper function to wait for the server to start
def wait_for_server(url, timeout=10):
    """Wait until the server at the specified URL is responsive."""
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        if time.time() - start_time > timeout:
            raise Exception(f"Server {url} did not start within {timeout} seconds.")
        time.sleep(1.0)

# Define the education levels
education_levels = ["Bachelor's", "Master's", "PhD"]


def predict_salary(age, education_level, job_title, years_of_experience):
    # Frontend validation
    if age < 18 or age > 65:
        return f"Error: Age must be between 18 and 65."

    max_experience = age - 18
    if years_of_experience < 0:
        return f"Error: Years of Experience cannot be negative."
    if years_of_experience > max_experience:
        return f"Error: Years of Experience cannot exceed {max_experience} years for the given age."

    # Prepare the request data
    input_data = {
        "age": age,
        "education_level": education_level,
        "job_title": job_title,
        "years_of_experience": years_of_experience
    }

    try:
        # Make prediction using the FastAPI endpoint
        response = requests.post(f"{API_URL}/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            return f"${result['predicted_salary']:,.2f}"
        else:
            return f"Error: {response.json()['detail']}"
    except Exception as e:
        return f"Error during prediction: {e}"

def get_recent_predictions():
    try:
        response = requests.get(f"{API_URL}/predictions?limit=5")
        if response.status_code == 200:
            predictions = response.json()
            return [[
                str(p["timestamp"]),
                p["age"],
                p["education_level"],
                p["job_title"],
                p["years_of_experience"],
                f"${p['predicted_salary']:,.2f}"
            ] for p in predictions]
    except Exception as e:
        return [[f"Error fetching predictions: {str(e)}"]]

def create_gradio_interface(unique_job_titles):
    with gr.Blocks() as demo:
        gr.Markdown("# Salary Prediction App")
        gr.Markdown("Enter your details to predict the expected salary.")

        with gr.Row():
            age_input = gr.Number(
                label="Age",
                value=30,
                precision=0,
                interactive=True,
                minimum=18,
                maximum=65
            )

            education_level_input = gr.Dropdown(
                label="Education Level",
                choices=education_levels,
                value=education_levels[0],
                interactive=True,
            )

        with gr.Row():
            job_title_input = gr.Dropdown(
                label="Job Title",
                choices=unique_job_titles,
                value=unique_job_titles[0] if unique_job_titles else None,
                interactive=True,
            )

            years_of_experience_input = gr.Number(
                label="Years of Experience",
                value=5.0,
                precision=1,
                interactive=True,
                minimum=0,
            )

        predict_button = gr.Button("Predict Salary")
        output = gr.Textbox(
            label="Prediction Output",
            placeholder="The predicted salary will appear here.",
            lines=2,
            interactive=False
        )

        gr.Markdown("## Recent Predictions")
        prediction_history = gr.Dataframe(
            headers=[
                "Time", "Age", "Education Level", 
                "Job Title", "Years of Experience", 
                "Predicted Salary"
            ],
            value=get_recent_predictions(),
            row_count=5,
            interactive=False
        )

        def on_predict(age, education_level, job_title, years_of_experience):
            result = predict_salary(age, education_level, job_title, years_of_experience)
            predictions = get_recent_predictions()
            return result, predictions

        predict_button.click(
            on_predict,
            inputs=[
                age_input,
                education_level_input,
                job_title_input,
                years_of_experience_input
            ],
            outputs=[output, prediction_history]
        )

    return demo

if __name__ == "__main__":
    # Start FastAPI server in a separate process
    fastapi_process = multiprocessing.Process(target=run_fastapi_server)
    fastapi_process.start()


    # Wait for the server to start
    wait_for_server(f"{API_URL}/health_check", timeout=30)

    # Get unique job titles
    unique_job_titles = get_job_titles()
    
    try:
        # Launch Gradio interface
        demo = create_gradio_interface(unique_job_titles)
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
    finally:
        # Ensure FastAPI server is terminated
        fastapi_process.terminate()
        fastapi_process.join()