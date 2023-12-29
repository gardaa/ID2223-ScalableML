import hopsworks
import joblib
import gradio as gr
import pandas as pd

# Log in to Hopsworks and get feature store
HOPSWORKS_API_KEY = "zB1HFw6waUQEsgoH.zTP79bPsYXZzR1hZN8L5lF7NJmgIJNR6ji7r4HenjkeeSel2MVi6Ca61AbrzGOy8"
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Get model registry and download the wine model and the folder from the registry
mr = project.get_model_registry()
model = mr.get_model("house_price_prediction_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/house_price_prediction_model.pkl")
print("Model downloaded from Hopsworks!")

# Load the locally stored model
#model = joblib.load('local_random_forest_model.pkl')

# Function to make predictions
def predict_price(postalcode, year, area, rooms, house_type):
    # Convert postalcode and house_type to categorical variables
    input_data = {'postalcode': [postalcode], 'year': [year], 'area': [area], 'rooms': [rooms], 'type': [house_type]}
    input_df = pd.DataFrame(input_data)

    # Make prediction
    price_prediction = model.predict(input_df)[0]
    formatted_price = '{:,.0f} ISK'.format(round(price_prediction))
    return formatted_price

iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Textbox("101", label="Postal Code"),
        gr.Textbox("2020", label="Year of Construction"),
        gr.Textbox("300", label="Area"),
        gr.Textbox("5", label="Number of Rooms"),
        gr.Radio(["Apartment", "Semi-detached House", "House"], label="Type", value="Apartment", type="index")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    live=True
)

iface.launch()