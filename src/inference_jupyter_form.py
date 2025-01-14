import ipywidgets as widgets
from IPython.display import display
from src.inference import make_inference_nn
import pandas as pd

def create_input_form(job_titles):
    """Creates an interactive input form with dropdown menus in a Jupyter Notebook."""

    form_items = [
        widgets.IntText(description="Age:", value=30),
        widgets.Dropdown(
            description="Education Level:",
            options=["Bachelor's", "Master's", "PhD"],
            value="Bachelor's",
        ),
        widgets.Dropdown(description="Job Title:", options=job_titles, value=job_titles[0]),
        widgets.FloatText(description="Years of Experience:", value=5.0),
    ]


    button = widgets.Button(description="Predict Salary")
    output = widgets.Output()
    
    def on_button_clicked(b):
        with output:
          input_data = pd.DataFrame(
              {
                  "Age": [form_items[0].value],
                  "Education Level": [form_items[1].value],
                  "Job Title": [form_items[2].value],
                  "Years of Experience": [form_items[3].value],
              }
            )

          try:
              prediction = make_inference_nn(input_data)
              print(f"Predicted Salary: ${prediction[0][0]:,.2f}")

          except Exception as e:
              print(f"Error: {e}")    

    button.on_click(on_button_clicked)
    
    form = widgets.VBox(form_items + [button, output])

    return form