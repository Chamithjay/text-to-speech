import glob
import os

import pyarrow.parquet as pq

OUT = "data/LJSpeech-1.1"
os.makedirs(f"{OUT}/wavs", exist_ok=True)

n = 0
with open(f"{OUT}/metadata.csv", "w", encoding="utf8", newline="\n") as meta:
    for f in sorted(glob.glob("data/hf_ljspeech/main/train/*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=["id", "audio", "text", "normalized_text"]):
            for row in batch.to_pylist():
                uid = row["id"]
                with open(f"{OUT}/wavs/{uid}.wav", "wb") as w:
                    w.write(row["audio"]["bytes"])   # these bytes are already a complete .wav file
                meta.write(f"{uid}|{row['text']}|{row['normalized_text']}\n")
                n += 1
        print("finished", f)
print("wrote", n, "clips")
