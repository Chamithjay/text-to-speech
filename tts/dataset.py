import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from .text import text_to_ids


class TTSDataset(Dataset):
    def __init__(self):
        with open(C.MANIFEST, encoding="utf8") as f:
            self.items = [l.rstrip("\n").split("|", 1) for l in f if l.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        utt_id, text = self.items[i]
        ids = torch.LongTensor(text_to_ids(text))
        mel = np.load(f"{C.MEL_DIR}/{utt_id}.npy").astype(np.float32)
        return ids, torch.from_numpy(mel)

def collate(batch):
# 1. Real lengths of every clip in this batch
    ids_len = torch.LongTensor([len(b[0]) for b in batch])
    mel_len = torch.LongTensor([len(b[1]) for b in batch])

     # 2. The size of the biggest clip. We pad everything to this.
    max_mel = int(mel_len.max())
    max_mel = ((max_mel + C.R - 1) // C.R) * C.R     # make it divisible by R (we need that)

    # 3. Make empty rectangles filled with zeros
    ids = torch.zeros(len(batch), int(ids_len.max()), dtype=torch.long)
    mels = torch.zeros(len(batch), max_mel, C.N_MELS)

    # 4. Copy each clip into its row (the leftover space stays zero = padding)
    for i, (t, m) in enumerate(batch):
        ids[i, :len(t)] = t
        mels[i, :len(m)] = m

    # 5. Make the "stop" answer key: 0 0 0 ... 0 1 1 1
    steps = max_mel // C.R
    valid = (mel_len + C.R - 1) // C.R
    stop = (torch.arange(steps)[None, :] >= (valid[:, None] - 1)).float()

    return dict(ids=ids, ids_len=ids_len, mels=mels, mel_len=mel_len, stop=stop)