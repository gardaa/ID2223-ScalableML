import joblib
import gradio as gr
import pandas as pd

# Load the locally stored model
model = joblib.load('local_random_forest_model.pkl')

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