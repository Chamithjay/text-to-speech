import torch
from torch.utils.data import DataLoader
from tts.dataset import TTSDataset, collate
from tts.model import Tacotron

loader = DataLoader(TTSDataset(), batch_size=4, shuffle=True, collate_fn=collate)
batch = next(iter(loader))

model = Tacotron()
print("total parameters:", sum(p.numel() for p in model.parameters()))

# --- training path ---
mel_out, mel_post, stops, aligns = model(batch["ids"], batch["ids_len"], batch["mels"])
print("\nTRAINING path")
print("real grid      :", tuple(batch["mels"].shape))
print("rough grid     :", tuple(mel_out.shape))
print("polished grid  :", tuple(mel_post.shape))
print("stop signals   :", tuple(stops.shape))
print("attention      :", tuple(aligns.shape), " (batch, steps, letters)")

# --- speaking path (untrained, so just checking that it runs) ---
model.eval()
one = batch["ids"][0:1, : int(batch["ids_len"][0])]
mel, align = model.inference(one, max_steps=20)
print("\nSPEAKING path")
print("mel   :", tuple(mel.shape), " (at most 40 rows = 20 steps x 2)")
print("attention:", tuple(align.shape))