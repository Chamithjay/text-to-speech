from torch.utils.data import DataLoader

from tts.dataset import TTSDataset, collate

ds = TTSDataset()
print("clips:", len(ds))

# DataLoader picks 4 random clips and uses our collate to make a batch
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate)
batch = next(iter(loader))

for name, value in batch.items():
    print(name, tuple(value.shape))

print("text lengths:", batch["ids_len"].tolist())
print("mel lengths :", batch["mel_len"].tolist())
print("stop targets, first clip, last 5 steps:", batch["stop"][0][-5:].tolist())
