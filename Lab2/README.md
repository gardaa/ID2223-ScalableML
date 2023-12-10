# Task 1: Fine tuning Whisper model for the Norwegian language
This is the dataset that has been used to make this project: [dataset](https://huggingface.co/datasets/google/fleurs). You can not see a preview of the dataset, so you have to load the norwegian dataset to see it.

## Files and description
The batch_inference.py (hosting) file is in its own file, while the pipelines for feature engineering and training is in a Whisper_Norwegian_1.ipynb file. 

<!-- The project consists of 3 different files:
- Danish-feature-pipeline.ipynb: This file prepares the environment, links the notebook to the Hugging Face Hub, loads the dataset abd prepares the feature extractor, tokenizer and data.
- Danish-training-pipeline.ipynb: This file defines a data collator, the evaluation metrics, defines the training configuration, and last but not least trains the model.
- Danish-interference-pipeline.py: This file builds a demo that allows the user to record a sound bite with their microphone or upload a sound file to transcribe to text. The UI also shows the output. -->

# Task 2: Improve pipeline scalability and model performance
## How to improve model performance
<!-- Describe in your README.md program ways in which you can improve
model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the
fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to
train a better model that one provided in the blog post -->
There are two different approaches to improve the performance: 
- **Model centric**
- **Data centric**

The model centric approach means that we change the code of the training pipeline, which results in a (hopefully) better trained model, which gives a better WER score. To achieve this, some changes might be: 
- Use a different pre-trained whisper model. In the tutorial, we use a small whisper model, but we can also use a medium or large model. 
- Tune the hyperparameters, such as the learning rate, batch size, dropout and number of epochs. 

The data centric approach implies that we keep the training pipeline as it is, but we find a new source of data (dataset), to train the data on. Usually, this means a bigger and better dataset, which gives the training pipeline more data to train the model on, which would make it more accurate and give it better performance. 

### Our Choice
We decided that the best approach for us would be to improve the model using a data centric approach. The reason is that the dataset in the original model was relatively small (ca. 3.9k entries for training/validation set and 500 entries for test set). Therefore, providing a bigger and better dataset would allow the model to be trained better. [This](https://huggingface.co/datasets/NbAiLab/NPSC) is the new dataset we used, which had a total of 56k entries for Norwegian Bokmål. However, we chose to downscale it to 15,6k (35% of the dataset) entries for the training set and 3,5k (65% of the test set) for the test set, as we felt like that was enough of an upgrade from the previous model's 3,9k.

## Files and description
The project is split into three different files: feature_engineering_pipeline.py which prepares the data, training_pipeline.py which trains the model and batch_inference_pipeline.py which hosts the model in a gradio environment, where the user can either speak into their microphone or upload an audio file, and it will be written in Norwegian. 