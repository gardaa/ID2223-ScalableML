import pandas as pd
import hopsworks

# Login to Hopsworks and get the feature store handle
HOPSWORKS_API_KEY = "PLACE API KEY HERE"
project = hopsworks.login(HOPSWORKS_API_KEY)
fs = project.get_feature_store()

icelandic_house_prices_df = pd.read_csv('data/icelandic_house_prices.csv')

##### EXPLORATORY DATA ANALYSIS (EDA) #####

##### DATA PREPROCESSING #####

##### FEATURE ENGINEERING #####

##### UPLOAD TO FEATURE STORE (HOPSWORKS) #####