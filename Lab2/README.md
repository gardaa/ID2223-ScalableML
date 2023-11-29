# Task 1: Fine tuning Whisper model for the Danish language
This is the dataset that has been used to make this project: [dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/da)

## Files and description
The project consists of 3 different files:
- Danish-feature-pipeline.ipynb: This file prepares the environment, links the notebook to the Hugging Face Hub, loads the dataset abd prepares the feature extractor, tokenizer and data.
- Danish-training-pipeline.ipynb: This file defines a data collator, the evaluation metrics, defines the training configuration, and last but not least trains the model.
- Danish-interference-pipeline.py: This file builds a demo that allows the user to record a sound bite with their microphone or upload a sound file to transcribe to text. The UI also shows the output.

# Task 2: Improve pipeline scalability and model performance
## How to improve model performance
<!-- Describe in your README.md program ways in which you can improve
model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the
fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to
train a better model that one provided in the blog post -->