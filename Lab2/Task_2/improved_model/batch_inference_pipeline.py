from transformers import pipeline
import gradio as gr

pipe = pipeline(model="gardaa/whisper-small-norwegian-improved-model-2")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=['microphone', 'upload'], type="filepath"),
    outputs="text",
    title="Whisper Small Norwegian",
    description="Realtime demo for Norwegian speech recognition using a fine-tuned Whisper small model.",
)

iface.launch(share=True)