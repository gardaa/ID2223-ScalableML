import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

# Log in to Hopsworks and get feature store
project = hopsworks.login()
fs = project.get_feature_store()

# Get model registry and download the wine model and the folder from the registry
mr = project.get_model_registry()
model = mr.get_model("wine_model_rf", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model_rf.pkl")
print("Model downloaded")

# 
def wine(type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    print("Calling wine function")
    wine_df = pd.DataFrame([[type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]],
                           columns=["type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol"])
    print("Predicting...")
    print(wine_df)

    # A list of predictions returned as a label
    result = model.predict(wine_df)
    print(result)

    return result

demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predicitve Analytics",
    description="Experiment with different values of variables of wine to predict the quality of the wine",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=0, label="Type (0=white, 1=red)"),
        gr.inputs.Number(default=0, label="Fixed acidity"),
        gr.inputs.Number(default=0, label="Volatile acidity"),
        gr.inputs.Number(default=0, label="Citric acid"),
        gr.inputs.Number(default=0, label="Residual sugar"),
        gr.inputs.Number(default=0, label="Chlorides"),
        gr.inputs.Number(default=0, label="Free sulfur dioxide"),
        gr.inputs.Number(default=0, label="Total sulfur dioxide"),
        gr.inputs.Number(default=0, label="Density"),
        gr.inputs.Number(default=0, label="pH"),
        gr.inputs.Number(default=0, label="Sulphates"),
        gr.inputs.Number(default=0, label="Alcohol"),
    ],
    outputs=gr.Textbox(type="text"))

demo.launch(debug=True)