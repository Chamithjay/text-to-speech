import librosa
import numpy as np
import soundfile
import torch

print("torch:", torch.__version__)
print("librosa:", librosa.__version__)
print("GPU available:", torch.cuda.is_available())

# make 1 second of a 440 Hz tone and turn it into a mel spectrogram
sr = 22050
t = np.linspace(0, 1, sr, endpoint=False)
y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
print("mel shape (bands, frames):", mel.shape)
