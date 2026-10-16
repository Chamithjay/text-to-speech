import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from tts import config as C
from tts.audio import load_wav, wav_to_mel
from tts.text import clean_text

ROOT = os.path.join(C.DATA_DIR, "LJSpeech-1.1")


def process(line):
    parts = line.rstrip("\n").split("|")
    utt_id, text = parts[0], clean_text(parts[-1])    # clip name, and its cleaned text
    mel = wav_to_mel(load_wav(os.path.join(ROOT, "wavs", utt_id + ".wav")))
    if len(mel) > C.MAX_MEL_FRAMES or not text:       # skip clips that are too long or empty
        return None
    np.save(os.path.join(C.MEL_DIR, utt_id + ".npy"), mel.astype(np.float16))
    return f"{utt_id}|{text}"

if __name__ == "__main__":
    os.makedirs(C.MEL_DIR, exist_ok=True)
    with open(os.path.join(ROOT, "metadata.csv"), encoding="utf8") as f:
        lines = [line for line in f if line.strip()]
    with Pool(os.cpu_count()) as pool:
        rows = [r for r in tqdm(pool.imap(process, lines, chunksize=16), total=len(lines)) if r]
    with open(C.MANIFEST, "w", encoding="utf8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"kept {len(rows)}/{len(lines)} clips")
