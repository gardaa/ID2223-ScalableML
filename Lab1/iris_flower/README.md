# Iris Flower Prediction
This is the dataset that has been used to make this project: [dataset](https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv)

## Files and description
The iris flower prediction project consists of 6 different files:
- iris-eda-and-backfill-feature-group.ipynb is a pipeline that prepare the data and uploads it to hopsworks as a feature group. 
- iris-training-pipeline.ipynb is a pipeline that fetches the data from Hopsworks as a feature view, and uses the data to create a model with the K-nearest-neighbors algorithm.
- huggingface-spaces-iris and huggingface-spaces-iris-monitor/app.py is the user accessible part of this project using Gradio applications hosted on Huggingface Spaces.
- iris-feature-pipeline-daily.py is a pipeline that generates a synthetic iris flower and adds it to the feature group
- iris-batch-inference-pipeline.py is a pipeline that predicts the iris flower of the synthetic flower using the model created, and monitors the predictions by uploading it as a feature group in Hopsworks. It also generates a confusion matrix and other images. 

iris_model/iris_model_rf.pkl is where the model that was trained is stored. 
