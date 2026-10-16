import math
import torch
import torch.nn.functional as F
from . import config as C


def tacotron_loss(mel_out, mel_post, stop_logits, batch):
    # mask: 1 for real columns, 0 for padding. Shape (batch, columns, 1)
    mask = (torch.arange(mel_out.size(1), device=mel_out.device)[None, :]
            < batch["mel_len"][:, None]).unsqueeze(-1).float()
    count = mask.sum() * C.N_MELS
    target = batch["mels"]

    def sound_loss(pred):
        diff = pred - target
        return ((diff.abs() + diff ** 2) * mask).sum() / count   # L1 + MSE, real columns only

    mel_loss = sound_loss(mel_out) + sound_loss(mel_post)

    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits, batch["stop"],
        pos_weight=torch.tensor(5.0, device=mel_out.device))
    return mel_loss, stop_loss


def guided_attention_loss(aligns, text_len, mel_len, g=0.2):
    # aligns: (batch, steps, letters)
    total = aligns.new_zeros(())
    B = aligns.size(0)
    for b in range(B):
        N = int(text_len[b])                                     # real letters
        T = int(math.ceil(int(mel_len[b]) / C.R))                # real steps
        n = torch.arange(N, device=aligns.device).float() / N    # 0..1 along the text
        t = torch.arange(T, device=aligns.device).float() / T    # 0..1 along the time
        # penalty grid: 0 on the diagonal, grows toward 1 far away from it
        penalty = 1.0 - torch.exp(-((n[None, :] - t[:, None]) ** 2) / (2 * g * g))
        total = total + (aligns[b, :T, :N] * penalty).mean()
    return total / B