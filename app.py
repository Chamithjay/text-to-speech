import argparse
import os
import tempfile

import gradio as gr
import soundfile as sf

from synthesize import Synthesizer


def build(ckpt):
    synth = Synthesizer(ckpt)                       # load the model once
    out_dir = tempfile.mkdtemp(prefix="tts_")       # folder for the audio files
    count = {"n": 0}

    def respond(message, history):
        message = (message or "").strip()
        if not message:
            return "", history, None
        rate, wav = synth(message)                  # text -> audio
        count["n"] += 1
        path = os.path.join(out_dir, f"reply_{count['n']}.wav")
        sf.write(path, wav, rate)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": {"path": path}},
        ]
        return "", history, path

    with gr.Blocks(title="Tiny TTS") as demo:
        gr.Markdown(f"## Tiny TTS\nModel trained for **{synth.step}** steps. "
                    "Type a sentence and press Enter.")
        chat = gr.Chatbot(type="messages", height=420)
        latest = gr.Audio(label="Latest reply", autoplay=True, interactive=False)
        box = gr.Textbox(placeholder="Type something for the model to say...", show_label=False)
        box.submit(respond, [box, chat], [box, chat, latest])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    args = parser.parse_args()
    build(args.ckpt).launch()