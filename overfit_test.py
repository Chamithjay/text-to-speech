import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import torch
from tts.dataset import TTSDataset, collate
from tts.losses import tacotron_loss, guided_attention_loss
from tts.model import Tacotron

torch.manual_seed(0)

# 4 very short clips, so each step is fast on a normal CPU
ds = TTSDataset()
short = [i for i, (_, text) in enumerate(ds.items) if len(text) < 35][:4]
batch = collate([ds[i] for i in short])
print("training on these 4 sentences:")
for i in short:
    print("  ", ds.items[i][1])
print("batch shapes: ids", tuple(batch["ids"].shape), " mels", tuple(batch["mels"].shape))

model = Tacotron()
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1, 201):
    mel_out, mel_post, stop_logits, aligns = model(batch["ids"], batch["ids_len"], batch["mels"])
    mel_loss, stop_loss = tacotron_loss(mel_out, mel_post, stop_logits, batch)
    ga_loss = guided_attention_loss(aligns, batch["ids_len"], batch["mel_len"])
    loss = mel_loss + 0.5 * stop_loss + ga_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stops one update from being too huge
    opt.step()

    if step == 1 or step % 20 == 0:
        print(f"step {step:3d} | loss {loss.item():.3f} | sound {mel_loss.item():.3f} "
              f"| stop {stop_loss.item():.3f} | attention {ga_loss.item():.3f}")

# draw the attention for the first clip
S = math.ceil(int(batch["mel_len"][0]) / 2)
N = int(batch["ids_len"][0])
plt.imshow(aligns[0, :S, :N].detach().numpy().T, aspect="auto", origin="lower")
plt.xlabel("time step"); plt.ylabel("letter"); plt.title("attention after 200 steps")
plt.savefig("samples/overfit_attention.png", dpi=100)
print("saved samples/overfit_attention.png")