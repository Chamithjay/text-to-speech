from torch.utils.data import DataLoader
from tts.dataset import TTSDataset, collate
from tts.model import Encoder

loader = DataLoader(TTSDataset(), batch_size=4, shuffle=True, collate_fn=collate)
batch = next(iter(loader))

enc = Encoder()
print("parameters:", sum(p.numel() for p in enc.parameters()))

out = enc(batch["ids"], batch["ids_len"])
print("input ids  :", tuple(batch["ids"].shape))
print("encoder out:", tuple(out.shape))