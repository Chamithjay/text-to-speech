# Tiny TTS

A text-to-speech system built in PyTorch. A Tacotron-2-style model, trained from scratch on the LJSpeech dataset, converts text into a mel spectrogram. A pretrained HiFi-GAN vocoder then converts the spectrogram into audio. A small Gradio web page lets you type a sentence and hear it spoken.

```
text -> encoder -> attention + decoder -> mel spectrogram -> HiFi-GAN -> audio
```

## How it works

**Text processing.** Input text is lowercased and cleaned, numbers are spelled out ("3" becomes "three"), and the result is converted to character IDs from a set of 37 symbols. Long text is split into sentences, each sentence is synthesised separately, and the pieces are joined with short pauses.

**Acoustic model.** A Tacotron-2-style model with 7.6M parameters:
- a character encoder (embedding, three convolution layers, bidirectional LSTM)
- location-sensitive attention, which decides which character to read at each moment
- an LSTM decoder with a pre-net, which predicts two mel frames per step, plus a stop-token head that decides when the sentence is finished
- a convolutional post-net that refines the predicted mel

It predicts an 80-band mel spectrogram (22,050 Hz audio, 1024-point FFT, 256-sample hop).

**Vocoder.** The mel spectrogram is turned into a waveform by the pretrained `speechbrain/tts-hifigan-ljspeech` HiFi-GAN. The model's output is converted from its own normalised decibel scale to the natural-log mel scale the vocoder expects. Griffin-Lim, a fixed algorithm that needs no training, is kept as an alternative.

**Web page.** A Gradio chat-style page loads the model once. Each message is synthesised and returned as an audio reply.

## Training

- **Data:** LJSpeech 1.1, 13,100 clips of one speaker, about 24 hours. Each clip is trimmed of silence and converted to a mel spectrogram before training.
- **Hardware:** a free Kaggle T4 GPU, about 4.4 hours per 10,000 steps at batch size 64.
- **Schedule:** 40,000 steps, run in four stages. Each stage resumed from the previous checkpoint, because free GPU sessions are time-limited.
- **Loss:** L1 and MSE on the mel before and after the post-net, a stop-token loss, and a guided-attention loss that pushes the attention towards a diagonal and is faded out by step 20,000. Padding is masked.
- **Optimiser:** Adam, learning rate 1e-3 until step 15,000 and then halved every 15,000 steps, with gradient clipping at 1.0.
- **Monitoring:** every 1,000 steps the training loop synthesises a fixed sentence and saves its attention plot and mel. The attention plot is the main check: a clean diagonal line means the model has learned to read the text in order.

## Results

Quality improved steadily with training:

| Steps | Observation |
|---|---|
| 10,000 | Clean attention diagonal. Words are correct but the audio is noisy. |
| 20,000 | Much clearer, with a long pause in the middle of a sentence. |
| 30,000 | The pause is gone and pacing is steady. |
| 40,000 | Sharpest mel spectrogram and the clearest voice. |

Replacing Griffin-Lim with the HiFi-GAN vocoder made the largest single improvement in how natural the audio sounds.

## Limitations

- One English voice only.
- The model reads characters rather than phonemes, so rare words can be mispronounced.
- Abbreviations and symbols are not expanded; only plain numbers are.
- The voice is clear but still slightly robotic.

## Credits

- Shen et al., [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) (Tacotron 2)
- Tachibana et al., [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)
- Kong et al., [HiFi-GAN](https://arxiv.org/abs/2010.05646), and the pretrained [LJSpeech model from SpeechBrain](https://huggingface.co/speechbrain/tts-hifigan-ljspeech) (Apache-2.0)
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) by Keith Ito (public domain)


