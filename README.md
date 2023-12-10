# Lab 1
The lab is split into two different tasks, one for predicting iris flowers and another for predicting wine quality based on various parameters. 
The source code for the iris flower prediction was provided in the assignment, whereas the code for the wine prediction is implemented by ourselves with inspiration from the iris flower prediction code.

## [Task 1 - Iris Flower](Lab1/iris_flower/README.md)
Link to folder: [Iris Flower](https://github.com/gardaa/ID2223-ScalableML/tree/main/Lab1/iris_flower)

The dataset used to train model: [Dataset](https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv)

The model predicts which flower it is between Setosa, Virginica or Versicolor based on sepal_length, sepal_width, petal_length and petal_width. The model to predict it is trained using the K-nearest-neighbors algorithm.

## [Task 2 - Wine Quality](Lab1/wine-quality/README.md)
Link to folder: [Wine Quality](https://github.com/gardaa/ID2223-ScalableML/tree/main/Lab1/wine-quality)

The dataset used to train model: [Dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv)

The model predicts the quality of a wine between 3-9 based on type, fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol. 
The model to predict it is trained using the RandomForest algorithm.

# Lab 2
This lab is also split into two different tasks. The first task involves fine tuning an existing automatic speech recognition (ASR) model, Whisper, into handling Norwegian. We have followed [this](https://huggingface.co/blog/fine-tune-whisper) blog post. The second tasks involves improving the model using a model centric and/or data centric approach. 