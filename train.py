import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")                       # draw plots to files, no window needed
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from tts import config as C
from tts.audio import mel_to_wav
from tts.dataset import TTSDataset, collate
from tts.losses import guided_attention_loss, tacotron_loss
from tts.model import Tacotron
from tts.text import text_to_ids

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def learning_rate(step):
    if step < 15000:
        return 1e-3
    return max(1e-5, 1e-3 * 0.5 ** ((step - 15000) / 15000))


def save_sample(model, device, step):
    """Make the model say a fixed sentence, so we can see and hear its progress."""
    os.makedirs(C.SAMPLE_DIR, exist_ok=True)
    model.eval()                                            # speaking mode
    ids = torch.LongTensor([text_to_ids(SAMPLE_TEXT)]).to(device)
    mel, align = model.inference(ids, max_steps=400)
    model.train()                                           # back to training mode

    wav = mel_to_wav(mel.cpu().numpy())
    sf.write(f"{C.SAMPLE_DIR}/step_{step:06d}.wav", wav, C.SAMPLE_RATE)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].imshow(align.cpu().numpy().T, aspect="auto", origin="lower")
    ax[0].set_title("attention (we want a diagonal)")
    ax[0].set_xlabel("time step"); ax[0].set_ylabel("letter")
    ax[1].imshow(mel.cpu().numpy().T, aspect="auto", origin="lower")
    ax[1].set_title("sound grid it made")
    fig.tight_layout()
    fig.savefig(f"{C.SAMPLE_DIR}/step_{step:06d}.png", dpi=90)
    plt.close(fig)


def save_checkpoint(model, opt, step):
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    torch.save(state, f"{C.CKPT_DIR}/latest.pt")
    if step % 5000 == 0:
        torch.save(state, f"{C.CKPT_DIR}/step_{step}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    args = parser.parse_args()

    device = pick_device()
    data = TTSDataset()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=collate, num_workers=args.workers,
                        pin_memory=(device == "cuda"))

    model = Tacotron().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    print(f"device: {device} | clips: {len(data)} | parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print("resumed from step", step)

    model.train()
    timer = time.time()
    while step < args.steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            for group in opt.param_groups:
                group["lr"] = learning_rate(step)

            # forward: the model guesses
            mel_out, mel_post, stop_logits, aligns = model(
                batch["ids"], batch["ids_len"], batch["mels"])

            # loss: how wrong was it?
            mel_loss, stop_loss = tacotron_loss(mel_out, mel_post, stop_logits, batch)
            ga_loss = guided_attention_loss(aligns, batch["ids_len"], batch["mel_len"])
            ga_weight = max(0.0, 1.0 - step / 20000)          # training wheels fade out
            loss = mel_loss + 0.5 * stop_loss + ga_weight * ga_loss

            # backward + step: nudge the weights
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                per_step = (time.time() - timer) / args.log_every
                timer = time.time()
                print(f"step {step:6d} | loss {loss.item():.3f} | sound {mel_loss.item():.3f} "
                      f"| stop {stop_loss.item():.3f} | attn {ga_loss.item():.4f} "
                      f"| {per_step:.2f}s/step")

            if step % args.sample_every == 0:
                save_sample(model, device, step)
                save_checkpoint(model, opt, step)
                print(f"   saved sample + checkpoint at step {step}")

            if step >= args.steps:
                break

    save_checkpoint(model, opt, step)
    print("training finished at step", step)


if __name__ == "__main__":
    main()