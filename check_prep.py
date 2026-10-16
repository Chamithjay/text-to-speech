import numpy as np
import soundfile as sf

from tts import config as C
from tts.audio import mel_to_wav
from tts.text import SYMBOLS, text_to_ids

# take the first clip from our list
utt, text = open(C.MANIFEST, encoding="utf8").readline().strip().split("|", 1)

# 1. text -> numbers -> text again (should match the original)
ids = text_to_ids(text)
print("text:", text)
print("ids :", ids[:15], "...")
print("back:", "".join(SYMBOLS[i] for i in ids))
print("alphabet size:", len(SYMBOLS))

# 2. look at the saved grid
mel = np.load(f"{C.MEL_DIR}/{utt}.npy").astype(np.float32)
print("mel:", mel.shape, "min %.2f max %.2f mean %.2f" % (mel.min(), mel.max(), mel.mean()))

# 3. turn the grid back into sound so you can listen
sf.write("samples/griffinlim_test.wav", mel_to_wav(mel), C.SAMPLE_RATE)
