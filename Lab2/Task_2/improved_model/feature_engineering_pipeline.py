from datasets import load_dataset, DatasetDict
import huggingface_hub
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

# repo = huggingface_hub.Repository(local_dir="cloned_model_repo", clone_from = "gardaa/ASR")

# repo.git_pull()

huggingface_hub.login(token="PLACE TOKEN HERE")

data = DatasetDict()

# Only load 35% of the training data and 65% of the test data
data["train"] = load_dataset("NbAiLab/NPSC", "16K_mp3_bokmaal", split="train[:12%]", use_auth_token=True)
data["test"] = load_dataset("NbAiLab/NPSC", "16K_mp3_bokmaal", split="test[:17%]", use_auth_token=True)

print(data)

# Remove unnecessary columns
data = data.remove_columns(["sentence_id", "sentence_order", "speaker_id", "meeting_date", "speaker_name", "sentence_language_code", "text", "start_time", "end_time", "normsentence_text", "transsentence_text", "translated"])

print(data)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Norwegian", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Norwegian", task="transcribe")

print(data["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence_text"]).input_ids
    return batch

print("Mapping dataset...")
data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=2)
print("Mapping done!")

data.save_to_disk("data")
print("Saved to disk!")
