import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = "data/LJSpeech-1.1"

rows = [line.rstrip("\n").split("|") for line in open(f"{ROOT}/metadata.csv", encoding="utf8")]
print("clips:", len(rows))

# durations (reads only the file headers, so it's fast)
durs = np.array([sf.info(f"{ROOT}/wavs/{r[0]}.wav").duration for r in rows])
print(f"total: {durs.sum()/3600:.1f} h | min {durs.min():.1f}s | mean {durs.mean():.1f}s | max {durs.max():.1f}s")

# text
texts = [r[-1] for r in rows]
print("longest text:", max(len(t) for t in texts), "characters")
print("characters used:", "".join(sorted(set("".join(texts)))))

# look at one random clip
row = random.choice(rows)
print("\nclip:", row[0], "\ntext:", row[-1])
y, sr = librosa.load(f"{ROOT}/wavs/{row[0]}.wav", sr=22050)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
db = librosa.power_to_db(mel, ref=np.max)
print("samples:", y.shape, "| mel (bands, frames):", mel.shape)

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(y); ax[0].set_title("waveform")
ax[1].imshow(db, aspect="auto", origin="lower"); ax[1].set_title("mel spectrogram (dB)")
plt.tight_layout()
plt.savefig("samples/explore.png", dpi=100)
