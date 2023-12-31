import huggingface_hub
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from datasets import load_from_disk

huggingface_hub.login(token="PLACE TOKEN HERE")

data = load_from_disk("data")
print("Data loaded from disk!")
print(data)
print(data["train"][0])

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
print("Before loading model...")
chosenModel = None
try:
    model = WhisperForConditionalGeneration.from_pretrained("gardaa/whisper-small-norwegian-improved-model-2")
    processor = WhisperProcessor.from_pretrained("gardaa/whisper-small-norwegian-improved-model-2", language="Norwegian", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("gardaa/whisper-small-norwegian-improved-model-2", language="Norwegian", task="transcribe")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    chosenModel = "norwegian-model"
    print("Model, processor, tokenizer and data collator loaded from norwegian-model!")
except:
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Norwegian", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Norwegian", task="transcribe")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    chosenModel = "whisper"
    print("Model, processor, tokenizer and data collator loaded from Whisper!")
    
print("After loading model...")
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model.config.forced_decoder_ids = None
model.config.dropout = 0.1
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-norwegian-improved-model-2",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-6,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    save_strategy="steps",
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
#model.save_pretrained(model)

if chosenModel == "norwegian-model":
    print("Training norwegian model...")
    trainer.train(resume_from_checkpoint=True)
else:
    print("Training Whisper model...")
    trainer.train()
print("Training done!")