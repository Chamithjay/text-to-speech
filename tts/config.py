SAMPLE_RATE = 22050   # the wav files have 22,050 numbers per second of sound
N_FFT = 1024          # how much sound we look at at once (~46 milliseconds)
HOP = 256             # how far we slide forward each time (~11.6 ms)
WIN = 1024            # same as N_FFT, a technical detail
N_MELS = 80           # how many "pitch bands" the grid has (like 80 piano-key groups)
FMIN = 0              # lowest pitch we care about (Hz)
FMAX = 8000           # highest pitch we care about. Speech has little above this
REF_DB = 20.0         # used when we squeeze loudness into a small range (see audio.py)
MIN_DB = -100.0       # anything quieter than this counts as silence

GL_ITERS = 200         # settings for turning the grid back into sound
GL_POWER = 1.3

MAX_MEL_FRAMES = 900  # skip clips longer than this (our longest is ~870, so none are skipped)
DATA_DIR = "data"
MEL_DIR = "data/mels"          # where the saved grids go
MANIFEST = "data/train.txt"    # the list of "clip name | cleaned text"
R = 2
EMB_DIM = 256    # size of each letter's "meaning card"
ENC_DIM = 256    # size of the encoder's output for each letter
ATT_RNN_DIM = 512   # size of the decoder's "what am I doing now" memory
ATT_DIM = 128       # size used inside the attention scoring

PRENET_DIM = 256     # size inside the prenet
DEC_RNN_DIM = 512    # size of the decoder's memory

POSTNET_DIM = 256

CKPT_DIR = "checkpoints"
SAMPLE_DIR = "samples"