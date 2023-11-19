import os
import random
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
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
    mr = project.get_model_registry()
    model = mr.get_model("wine_model_rf", version=4)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model_rf.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    
    # Predict the wine quality with our model and find the correct picture for it
    y_pred = model.predict(batch_data)
    #offset = 1
    offset =  random.randint(1, y_pred.size)
    quality = y_pred[y_pred.size-offset]
    wine_url = "https://raw.githubusercontent.com/gardaa/ID2223-ScalableML/main/Lab1/wine-quality/wine_images/" + str(quality) + ".jpg"

    # Save and upload image (overwrite it) for monitoring
    print("Wine predicted: " + str(quality))
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    img.save("./latest_wine.jpg")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.jpg", "Resources/images", overwrite=True)
   
    # Save and upload the actual quality of the wine
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    label = df.iloc[-offset]["quality"]
    label_url = "https://raw.githubusercontent.com/gardaa/ID2223-ScalableML/main/Lab1/wine-quality/wine_images/" + str(int(label)) + ".jpg"
    print("Actual quality: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_quality.jpg")
    dataset_api.upload("./actual_quality.jpg", "Resources/images", overwrite=True)
    
    # Create a new feature group for monitoring the wine predictions
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [quality],
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
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 7 wine qualities
    print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 7:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, range(3,10), range(3,10))
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    else:
        print("You need 7 different wine quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 7 different wine quality predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

