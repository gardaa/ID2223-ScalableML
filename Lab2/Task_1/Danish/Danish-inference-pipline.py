import locale
locale.getpreferredencoding = lambda: "UTF-8"
from transformers import pipeline
import gradio as gr

pipe = pipeline(model="gardaa/output")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=['upload', 'microphone'], type="filepath"),
    outputs="text",
    title="Whisper Small Danish",
    description="Realtime demo for Danish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()