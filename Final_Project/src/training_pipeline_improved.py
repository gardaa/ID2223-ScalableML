import pandas as pd
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
import joblib

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

def split_train_test(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Select features (X) and target variable (y)
    X = df[['postalcode', 'year', 'area', 'rooms', 'type']]
    y = df['price']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    grid_search.fit(X_train, y_train)
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

def save_model(model, filename='random_forest_model.pkl'):
    # Save the model
    joblib.dump(model, filename)
    print(f'Model saved as {filename}')

# Example usage:
# Assuming 'df' is your DataFrame
# X_train, X_test, y_train, y_test = split_train_test(df)
login, feature_view = login_and_create_feature_view()
csv_file_path = 'Final_Project/data/kaupskra_clean.csv'
X_train, X_test, y_train, y_test = split_train_test(csv_file_path)
best_rf_model = train_random_forest(X_train, y_train)
evaluate_model(best_rf_model, X_test, y_test)
save_model(best_rf_model, 'local_random_forest_model.pkl')