import librosa
import numpy as np

from . import config as C


def load_wav(path):
    y, _ = librosa.load(path, sr=C.SAMPLE_RATE)       # read the file as a list of numbers
    y, _ = librosa.effects.trim(y, top_db=40)         # cut silence at the start and end
    return y

def wav_to_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, hop_length=C.HOP, win_length=C.WIN,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX, power=1.0)
    db = 20 * np.log10(np.maximum(mel, 1e-5)) - C.REF_DB
    norm = np.clip((db - C.MIN_DB) / (-C.MIN_DB), 0, 1)
    return norm.T.astype(np.float32)

def mel_to_wav(mel):
    mel = np.clip(mel, 0, 1).T
    db = mel * (-C.MIN_DB) + C.MIN_DB + C.REF_DB                 # undo the squeeze
    amp = np.power(10.0, db / 20.0)                              # undo the log
    stft = librosa.feature.inverse.mel_to_stft(
        amp, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, power=1.0, fmin=C.FMIN, fmax=C.FMAX)
    wav = librosa.griffinlim(stft ** C.GL_POWER, n_iter=C.GL_ITERS,
                             hop_length=C.HOP, win_length=C.WIN, n_fft=C.N_FFT)
    return (wav / (np.abs(wav).max() + 1e-8) * 0.95).astype(np.float32)