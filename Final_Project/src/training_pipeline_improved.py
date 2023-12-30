import os
import pandas as pd
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import time
from datetime import datetime

def confirm_data_upload():
    # Wait for the upload to complete or until a specific condition is met
    max_wait_time_seconds = 600  # Adjust the maximum wait time as needed
    wait_interval_seconds = 10  # Adjust the interval between checks as needed

    wait_start_time = time.time()

    while (not os.path.exists('upload_complete.txt') or
        datetime.now().strftime("%Y-%m") != open('upload_complete.txt', 'r').read().strip()[:7]) and \
        (time.time() - wait_start_time) < max_wait_time_seconds:
        time.sleep(wait_interval_seconds)

    if os.path.exists('upload_complete.txt'):
        # Fetch data from Hopsworks and proceed with training
        print("Data upload confirmed. Proceeding with training.")
    else:
        print("Data upload confirmation timed out or month mismatch. Exiting without training.")

# FETCH DATA FROM HOPSWORKS AS MODEL VIEW
def login_and_create_feature_view():
    #Log in to Hopsworks
    HOPSWORKS_API_KEY = "zB1HFw6waUQEsgoH.zTP79bPsYXZzR1hZN8L5lF7NJmgIJNR6ji7r4HenjkeeSel2MVi6Ca61AbrzGOy8"
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    # connection = hsfs.connection()
    fs = project.get_feature_store()
    
    # Get feature view
    feature_group = fs.get_feature_group(name="icelandic_house_prices", version=1)
    query = feature_group.select_all()
    feature_view = fs.get_or_create_feature_view(
    # feature_view = fs.create_feature_view(
        name="icelandic_house_prices", 
        version=9,
        description="Read from Icelandic house prices dataset",
        labels=["price"],
        query=query
    )

    return project, feature_view

def split_train_test(feataure_view):
    #df = pd.read_csv(csv_file_path)
    # Select features (X) and target variable (y)
    #X = df[['postalcode', 'year', 'area', 'rooms', 'type']]
    #y = df['price']

    # Split the data into training and testing sets (80% train, 20% test)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)

    return X_train, X_test, y_train, y_test

# def split_train_test(feature_view, target_feature):
#     # Assuming 'feature_view' is a FeatureView object
#     # Load data from the FeatureView
#     df = feature_view.get_data_as_dataframe()

#     # Select features (X) and target variable (y)
#     X = df.drop(target_feature, axis=1)  # Exclude the target feature
#     y = df[target_feature]

#     # Split the data into training and testing sets (80% train, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    # Create a RandomForestRegressor
    rf = RandomForestRegressor()

    # Define the hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [200],
        'max_depth': [20],
        'min_samples_split': [5],
        'min_samples_leaf': [2]
    }

    # Perform Grid Search Cross Validation to find the best hyperparameters
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train.values.ravel())
    print("Best hyperparameters:", grid_search.best_params_)

    # Get the best model
    best_rf = grid_search.best_estimator_

    return best_rf

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    return rmse

def save_model(model, filename='random_forest_model.pkl'):
    # Save the model
    joblib.dump(model, filename)
    print(f'Model saved as {filename}')

def upload_model_to_hopsworks(login, model, X_train, y_train, rmse_score):
    # Get an object for model registry from Hopsworks
    mr = login.get_model_registry()

    # Create folder to store house_price prediction model if it does not exist
    model_dir = "house_price_prediction_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save model and conusfion matrix for rf to the correct folder. Both will be uploaded to model registry in Hopsworks
    joblib.dump(model, model_dir + "/house_price_prediction_model.pkl")
    #fig_rf.savefig(model_dir + "/confusion_matrix_rf.png")

    # Specify schema of the model 
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create model in the model registry that includes the model name, metrics, schema and description
    house_price_prediction_model = mr.python.create_model(
        name="house_price_prediction_model",
        metrics={"RMSE (Root Mean Squared Error):" : rmse_score},
        model_schema=model_schema,
        description="Icelandic house price prediction model"
    )

    # Upload model to model registry with all files in the folder
    house_price_prediction_model.save(model_dir)
    print("Model saved to local file and uploaded to Hopsworks!")

def main():
    confirm_data_upload()
    login, feature_view = login_and_create_feature_view()
    #csv_file_path = '../data/kaupskra_clean.csv'
    X_train, X_test, y_train, y_test = split_train_test(feature_view)
    best_rf_model = train_random_forest(X_train, y_train)
    rmse_score = evaluate_model(best_rf_model, X_test, y_test)
    #save_model(best_rf_model, 'local_random_forest_model.pkl')
    upload_model_to_hopsworks(login, best_rf_model, X_train, y_train, rmse_score)

main()