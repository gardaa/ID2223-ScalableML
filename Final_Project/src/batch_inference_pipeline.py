import os
import random
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   HOPSWORKS_API_KEY = "zB1HFw6waUQEsgoH.zTP79bPsYXZzR1hZN8L5lF7NJmgIJNR6ji7r4HenjkeeSel2MVi6Ca61AbrzGOy8"
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name(HOPSWORKS_API_KEY))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Get model and directory from Hopsworks
    # mr = project.get_model_registry()
    # model = mr.get_model("MODEL_NAME_HERE_rf", version=4)
    # model_dir = model.download()
    # model = joblib.load(model_dir + "/MODEL_NAME_HERE_rf.pkl")
    
    feature_view = fs.get_feature_view(name="icelandic_house_prices", version=1)
    batch_data = feature_view.get_batch_data()
    
    # Predict the price
    y_pred = model.predict(batch_data)
    #offset = 1
    offset =  random.randint(1, y_pred.size) # what is offset?
    price = y_pred[y_pred.size-offset]
    
    # Print price
    print("Price predicted: " + str(price))
    dataset_api = project.get_dataset_api()    
    # dataset_api.upload("./latest_wine.jpg", "Resources/images", overwrite=True) # Print price here?
   
    # Save and upload the actual quality of the wine
    icelandic_house_prices_fg = fs.get_feature_group(name="icelandic_house_prices", version=1)
    df = icelandic_house_prices_fg.read() 
    label = df.iloc[-offset]["price"] # offset?
    print("Actual price: " + str(label))
    # dataset_api.upload("./actual_quality.jpg", "Resources/images", overwrite=True) # Print price here?
    
    # Create a new feature group for monitoring the wine predictions
    monitor_fg = fs.get_or_create_feature_group(name="price_predictions",
                                                version=1,
                                                primary_key=["datetime"], # this correct?
                                                description="Icelandic house prices dataset"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") # what is this?
    data = {
        'prediction': [price],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    # Get the recent wines for the monitoring
    df_recent = history_df.tail(4)
    # dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib') # how to do this?
    # dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 7 wine qualities
    print("Number of different price predictions to date: " + str(predictions.value_counts().count()))
    # if predictions.value_counts().count() == 7:  # is heatmap applicable here?
    #     results = confusion_matrix(labels, predictions)
    
        # df_cm = pd.DataFrame(results, range(3,10), range(3,10))
    
        # cm = sns.heatmap(df_cm, annot=True)
        # fig = cm.get_figure()
        # fig.savefig("./confusion_matrix_wine.png")
        # dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    # else:
    #     print("You need 7 different price predictions to create the confusion matrix.") # is this also applicable?
        # print("Run the batch inference pipeline more times until you get 7 different wine quality predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

