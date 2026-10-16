import torch
from torch.utils.data import DataLoader
from tts import config as C
from tts.dataset import TTSDataset, collate
from tts.model import Encoder, Decoder, sequence_mask

loader = DataLoader(TTSDataset(), batch_size=4, shuffle=True, collate_fn=collate)
batch = next(iter(loader))
B, T = batch["ids"].shape

enc, dec = Encoder(), Decoder()
print("decoder parameters:", sum(p.numel() for p in dec.parameters()))

memory = enc(batch["ids"], batch["ids_len"])
mask = ~sequence_mask(batch["ids_len"], T)
proc = dec.attn.memory(memory)
st = dec.init_state(memory)

prev = torch.zeros(B, C.N_MELS)          # the "go" frame: silence, to start
for i in range(5):
    mel, stop, w = dec.step(prev, st, memory, proc, mask)
    mel = mel.view(B, C.R, C.N_MELS)     # 160 numbers -> 2 columns of 80
    prev = mel[:, -1]                    # next step sees the last column we made
    print(f"step {i}: mel {tuple(mel.shape)}  stop {tuple(stop.shape)}  attention {tuple(w.shape)}")

print("attention total after 5 steps (should be about 5):", [round(x, 3) for x in st["w_cum"].sum(dim=1).tolist()])