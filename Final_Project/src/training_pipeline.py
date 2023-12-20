import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# FETCH DATA FROM HOPSWORKS AS MODEL VIEW
def login_and_create_feature_view():
    #Log in to Hopsworks
    HOPSWORKS_API_KEY = "PLACE API KEY HERE"
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Get feature view
    feature_group = fs.get_feature_group(name="icelandic_house_prices", version=1)
    query = feature_group.select_all()
    feature_view = fs.get_or_create_feature_view(
        name="icelandic_house_prices", 
        version=8,
        description="Read from Icelandic house prices dataset",
        labels=["price"],
        query=query
    )

    return feature_view

def train_and_evaluate_models():
    return 0
# SPLIT DATA

# TRAIN BEST MODEL

# EVALUATE AND FIND BEST MODEL

# HYPERTUNE MODEL

# UPLOAD MODEL TO HOPSWORKS

def main():
    feature_view = login_and_create_feature_view()
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
    print("X_train:", X_train, "\nX_test:", X_test, "\ny_train:", y_train, "\ny_test:", y_test)

    # A dictionary of models to train and later add corresponding RMSE scores to each model
    models_and_metrics = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "ElasticNet": ElasticNet(),
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }

    # Iterate over the models and train them and evaluate them using cross-validation
    for name, model in models_and_metrics.items():
        model.fit(X_train, y_train.values.ravel())
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = (scores * -1) ** 0.5  # Convert negative MSE to RMSE
        avg_rmse = rmse_scores.mean()
        
        models_and_metrics[name] = [model, avg_rmse]
    
    print(models_and_metrics)
    print("Best model:", min(models_and_metrics, key=models_and_metrics.get))

if __name__ == "__main__":
    main()