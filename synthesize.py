import argparse
import re

import numpy as np
import soundfile as sf
import torch

from tts import config as C
from tts.audio import mel_to_wav
from tts.model import Tacotron
from tts.text import clean_text, text_to_ids


class Synthesizer:
    def __init__(self, ckpt="checkpoints/latest.pt", vocoder="hifigan"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Tacotron().to(self.device)
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.step = state.get("step", 0)
        self.vocoder_name = vocoder
        if vocoder == "hifigan":
            from tts.vocoder import HiFiGAN
            self.vocoder = HiFiGAN(self.device)
        else:
            self.vocoder = mel_to_wav          # Griffin-Lim

    def __call__(self, text):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if clean_text(s)]
        pause = np.zeros(int(0.25 * C.SAMPLE_RATE), dtype=np.float32)
        pieces = []
        for sentence in sentences:
            ids = torch.LongTensor([text_to_ids(sentence)]).to(self.device)
            max_steps = min(450, 20 + 6 * ids.size(1))
            mel, _ = self.model.inference(ids, max_steps=max_steps)
            pieces += [self.vocoder(mel.cpu().numpy()), pause]
        if not pieces:
            return C.SAMPLE_RATE, np.zeros(1, dtype=np.float32)
        wav = np.concatenate(pieces)
        wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.95    # same loudness for every reply
        return C.SAMPLE_RATE, wav.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--out", default="out.wav")
    parser.add_argument("--vocoder", choices=["hifigan", "griffinlim"], default="hifigan")
    parser.add_argument("--iters", type=int, default=C.GL_ITERS, help="Griffin-Lim only")
    args = parser.parse_args()

    C.GL_ITERS = args.iters
    synth = Synthesizer(args.ckpt, args.vocoder)
    rate, wav = synth(args.text)
    sf.write(args.out, wav, rate)
    print(f"trained {synth.step} steps | vocoder {args.vocoder} -> wrote {args.out} ({len(wav) / rate:.1f} s)")


if __name__ == "__main__":
    main()