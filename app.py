import gradio as gr
import pandas as pd
import numpy as np

# Import your inference functions
from src.inference import (
    make_inference_nn,
    get_unique_job_titles,
    load_scaler,
    load_target_encoder,
    load_model_nn,
)

# Load models and preprocessors
prefix = ""  # Adjust the prefix if needed

# Load the scaler
scaler = load_scaler(prefix=prefix)

# Load the target encoder
te = load_target_encoder(prefix=prefix)

# Load the trained Neural Network model
model_nn = load_model_nn(prefix=prefix)

# Load unique job titles
unique_job_titles = get_unique_job_titles(prefix=prefix)

# Define the education levels
education_levels = ["Bachelor's", "Master's", "PhD"]

# Define the prediction function
def predict_salary(age, education_level, job_title, years_of_experience):
    # Frontend validation
    if age < 18 or age > 65:
        return f"Error: Age must be between 18 and 65."

    max_experience = age - 18
    if years_of_experience < 0:
        return f"Error: Years of Experience cannot be negative."
    if years_of_experience > max_experience:
        return f"Error: Years of Experience cannot exceed {max_experience} years for the given age."

    # Validate job title
    if job_title not in unique_job_titles:
        return f"Error: Invalid job title selected."

    # Create a DataFrame from the input data
    data = pd.DataFrame({
        "Age": [age],
        "Education Level": [education_level],
        "Job Title": [job_title],
        "Years of Experience": [years_of_experience],
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
        # Return the prediction as a formatted string
        return f"${predicted_salary:,.2f}"
    except Exception as e:
        return f"Error during prediction: {e}"

# Create the Gradio interface
def create_gradio_interface():
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

        def on_predict(
            age, education_level, job_title, years_of_experience
        ):
            result = predict_salary(
                age,
                education_level,
                job_title,
                years_of_experience
            )
            return result

        predict_button.click(
            on_predict,
            inputs=[
                age_input,
                education_level_input,
                job_title_input,
                years_of_experience_input
            ],
            outputs=output
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()