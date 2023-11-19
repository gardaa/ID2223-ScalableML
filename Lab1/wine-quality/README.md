# Wine Quality Prediction
This is the dataset that has been used to make this project: [dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv)

## Files and description
The wine quality prediction project consists of 6 different files:
- wine-eda-and-backfill-feature-group.ipynb is a pipeline that prepare the data and uploads it to hopsworks as a feature group. 
- wine-training-pipeline.ipynb is a pipeline that fetches the data from Hopsworks as a feature view, and uses the data to create a model with the RandomForest algoritm. We tested multple algorithms and RandomForest was the algorithm that gave the best accuracy.
- wine/app.py is a script that creates and hosts a UI that predicts a wine quality with the trained model based on 12 different parameters: type, fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol.
- daily-wine-feature-pipeline.py is a pipeline that generates a synthetic wine and adds it to the feature group
- wine-batch-inference-pipeline.py is a pipeline that predicts the wine quality of the synthetic wine using the model created, and monitors the predictions by uploading it as a feature group in Hopsworks. It also generates a confusion matrix and other images. 
- wine-monitoring/app.py is a script that creates and hosts a UI which can be used to seen the latest generated flower, latest predictions, confusion matrix etc. 

wine_model/wine_model_rf.pkl is where the model that was trained is being stored. 