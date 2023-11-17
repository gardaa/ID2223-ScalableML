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
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

# 
def wine(type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol, quality):
    print("Calling wine function")
    wine_df = pd.DataFrame([type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol, quality],
                           columns=["type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol", "quality"])
    print("Predicting...")
    print(wine_df)

    # A list of predictions returned as a label
    result = model.predict(wine_df)
    print(result)
