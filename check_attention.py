import torch
from torch.utils.data import DataLoader
from tts import config as C
from tts.dataset import TTSDataset, collate
from tts.model import Encoder, LocationSensitiveAttention, sequence_mask

loader = DataLoader(TTSDataset(), batch_size=4, shuffle=True, collate_fn=collate)
batch = next(iter(loader))
B, T = batch["ids"].shape

enc = Encoder()
att = LocationSensitiveAttention()

memory = enc(batch["ids"], batch["ids_len"])                  # letter cards
proc_memory = att.memory(memory)                              # computed once
mask = ~sequence_mask(batch["ids_len"], T)                    # True where padding

query = torch.randn(B, C.ATT_RNN_DIM)                         # a fake "decoder state"
prev_w = torch.zeros(B, 2, T)                                 # finger hasn't moved yet

context, w = att(query, memory, proc_memory, prev_w, mask)

print("context shape :", tuple(context.shape))
print("weights shape :", tuple(w.shape))
print("weights add up to:", w.sum(dim=1).tolist())
print("text lengths     :", batch["ids_len"].tolist())
print("max weight (clip 0):", round(w[0].max().item(), 4), "  vs even split 1/len =", round(1 / batch["ids_len"][0].item(), 4))
shortest = int(batch["ids_len"].argmin())
print("weight on padding (shortest clip):", w[shortest][batch["ids_len"][shortest]:].sum().item())