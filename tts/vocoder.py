"""Neural vocoder: mel spectrogram -> waveform (pretrained HiFi-GAN trained on LJSpeech)."""
import numpy as np

from . import config as C

LN10_OVER_20 = np.log(10.0) / 20.0


class HiFiGAN:
    def __init__(self, device="cpu"):
        # imported here so the Griffin-Lim path still works without speechbrain
        from speechbrain.inference.vocoders import HIFIGAN
        from speechbrain.utils.fetching import LocalStrategy

        self.model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="pretrained_models/tts-hifigan-ljspeech",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device},
        )

    def __call__(self, mel):
        """mel: (frames, 80) numpy in [0, 1]  ->  float32 waveform."""
        import torch

        db = np.clip(mel, 0.0, 1.0) * (-C.MIN_DB) + C.MIN_DB + C.REF_DB   # 20*log10(magnitude)
        log_mel = db * LN10_OVER_20                                       # ln(magnitude)
        x = torch.from_numpy(log_mel.T.astype(np.float32)).unsqueeze(0)   # (1, 80, frames)
        with torch.no_grad():
            wav = self.model.decode_batch(x).squeeze().cpu().numpy()
        return wav.astype(np.float32)