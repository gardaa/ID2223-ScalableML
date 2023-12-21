import time
import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# FETCH DATA FROM HOPSWORKS AS MODEL VIEW
def login_and_create_feature_view():
    #Log in to Hopsworks
    HOPSWORKS_API_KEY = "zB1HFw6waUQEsgoH.zTP79bPsYXZzR1hZN8L5lF7NJmgIJNR6ji7r4HenjkeeSel2MVi6Ca61AbrzGOy8"
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

def model_selection(X_train, y_train):
    # A dictionary of models to train and later add corresponding RMSE scores to each model
    models_and_metrics = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "ElasticNet": ElasticNet(),
        "Lasso": Lasso()
        # "Ridge": Ridge()
    }

    # Iterate over the models and train them and evaluate them using cross-validation
    for name, model in models_and_metrics.items():
        print("Training", name)
        # Use ravel to flatten y_train to 1D array to avoid warning
        model.fit(X_train, y_train.values.ravel())
        scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error', verbose=2)
        # Convert negative MSE to RMSE
        rmse_scores = (scores * -1) ** 0.5
        avg_rmse = rmse_scores.mean()
        
        # Add the average RMSE to the dictionary
        models_and_metrics[name] = [model, avg_rmse]
        print("Average RMSE for", name, ":", avg_rmse, "\n")
    
    print(models_and_metrics)
    sorted_models = sorted(models_and_metrics.items(), key=lambda x: x[1][1])
    three_best_models = sorted_models[:3]
    top_three_models_dict = {name: model_info[0] for name, model_info in three_best_models}
    print("Top three models:", top_three_models_dict)
    return top_three_models_dict

def tune_best_models(best_models, X_val, y_val):
    # Parameter grid for random forest
    # Best hyperparameters: {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
    random_forest_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    # Parameter grid for KNN
    # Best hyperparameters: {'n_neighbors': 10, 'p': 1, 'weights': 'distance'}
    knn_params = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }
    # Parameter grid for ElasticNet
    # Not part of my top 3!
    elastic_net_params = {
        'alpha': [0.1, 0.5, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9],
        'max_iter': [1000, 2000, 3000],
        'tol': [1e-4, 1e-5, 1e-6]
    }

    # Dict to store the parameter grids for each model
    # Must add DecisionTreeRegressor!
    model_param_dict = {"RandomForestRegressor": random_forest_params, "KNeighborsRegressor": knn_params, "ElasticNet": elastic_net_params}

    best_rmse = {}

    # Take the time for how long it takes to tune the models
    start_time = time.time()

    # Iterate over the models and tune them using grid search with 5-fold cross-validation
    for model_name, model in best_models.items():
        print("Tuning", model_name)
        grid_search = GridSearchCV(model, model_param_dict[model_name], cv=5, scoring='neg_mean_squared_error', verbose=2)

        # Fit the grid search to the data
        grid_search.fit(X_val, y_val.values.ravel())

        # Print the best hyperparameters and corresponding RMSE, and append the RMSE to the list
        print("Best hyperparameters:", grid_search.best_params_)
        best_rmse_value = (-grid_search.best_score_) ** 0.5
        print("Best RMSE:", best_rmse_value, "\n")
        best_rmse[model_name] = best_rmse_value
        print(best_rmse)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken to tune:", total_time, "seconds")

    # Get the best model instance
    best_model_instance_after_tuning = grid_search.best_estimator_
    return best_model_instance_after_tuning

def upload_model(model):
    # # Get an object for model registry from Hopsworks
    # mr = project.get_model_registry()

    # # Create folder to store wine model if it does not exist
    # model_dir = "wine_model"
    # if os.path.isdir(model_dir) == False:
    #     os.mkdir(model_dir)

    # # Save model and conusfion matrix for rf to the correct folder. Both will be uploaded to model registry in Hopsworks
    # joblib.dump(model_rf, model_dir + "/wine_model_rf.pkl")
    # fig_rf.savefig(model_dir + "/confusion_matrix_rf.png")

    # # Specify schema of the model 
    # input_schema = Schema(X_train)
    # output_schema = Schema(y_train)
    # model_schema = ModelSchema(input_schema, output_schema)

    # # Create model in the model registry that includes the model name, metrics, schema and description
    # wine_model = mr.python.create_model(
    #     name="wine_model_rf",
    #     metrics={"accuray:" : metrics["accuracy"]},
    #     model_schema=model_schema,
    #     description="Wine Quality Predictor"
    # )

    # # Upload model to model registry with all files in the folder
    # wine_model.save(model_dir)
    return None

# SPLIT DATA

# TRAIN MODELS WITH DEFAULT HYPERPARAMETERS

# EVALUATE AND FIND BEST MODEL

# HYPERTUNE BEST MODEL

# UPLOAD MODEL TO HOPSWORKS

def main():
    feature_view = login_and_create_feature_view()
    # Split the data into training and testing sets
    #X_train, X_val, X_test, y_train, y_val, y_test = feature_view.train_validation_test_split(validation_size=0.3, test_size=0.2)
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
    print("X_train:", X_train, "\nX_test:", X_test, "\ny_train:", y_train, "\ny_test:", y_test)

    # Train and evaluate models on training data
    best_models = model_selection(X_train, y_train)

    # Tune the best model on test data
    best_model_tuned = tune_best_models(best_models, X_test, y_test)

    # Upload the best model to Hopsworks
    upload_model(best_model_tuned)

if __name__ == "__main__":
    main()