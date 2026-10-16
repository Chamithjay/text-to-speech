import torch
import torch.nn as nn
from . import config as C
from .text import SYMBOLS
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. the meaning cards
        self.emb = nn.Embedding(len(SYMBOLS), C.EMB_DIM, padding_idx=0)

        # 2. three conv layers, each followed by BatchNorm, ReLU and Dropout
        layers = []
        in_ch = C.EMB_DIM
        for _ in range(3):
            layers += [
                nn.Conv1d(in_ch, C.ENC_DIM, kernel_size=5, padding=2),
                nn.BatchNorm1d(C.ENC_DIM),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
            in_ch = C.ENC_DIM
        self.convs = nn.Sequential(*layers)

        # 3. the two-direction reader (128 numbers each way = 256 total)
        self.lstm = nn.LSTM(C.ENC_DIM, C.ENC_DIM // 2, batch_first=True, bidirectional=True)

    def forward(self, ids, lens):
        x = self.emb(ids)                      # numbers -> cards, shape (batch, letters, 256)
        x = self.convs(x.transpose(1, 2))      # conv layers want (batch, 256, letters)
        x = x.transpose(1, 2)                  # flip back to (batch, letters, 256)

        # tell the LSTM which parts are real letters and which are padding blanks
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=ids.size(1))
        return out                             # (batch, letters, 256)
def sequence_mask(lens, max_len):
    # True for real letters, False for padding. e.g. lens=[3,5], max_len=5
    # -> [[T,T,T,F,F],[T,T,T,T,T]]
    return torch.arange(max_len, device=lens.device)[None, :] < lens[:, None]


class LocationSensitiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(C.ATT_RNN_DIM, C.ATT_DIM, bias=False)   # shrinks the query
        self.memory = nn.Linear(C.ENC_DIM, C.ATT_DIM, bias=False)      # shrinks the letter cards
        self.loc_conv = nn.Conv1d(2, 32, kernel_size=31, padding=15, bias=False)  # looks at past finger positions
        self.loc_dense = nn.Linear(32, C.ATT_DIM, bias=False)
        self.v = nn.Linear(C.ATT_DIM, 1, bias=False)                   # turns everything into ONE score

    def forward(self, query, memory, proc_memory, prev_w, mask):
        q = self.query(query).unsqueeze(1)                              # (batch, 1, 128)
        loc = self.loc_dense(self.loc_conv(prev_w).transpose(1, 2))     # (batch, letters, 128)
        score = self.v(torch.tanh(q + loc + proc_memory)).squeeze(-1)   # (batch, letters)
        score = score.masked_fill(mask, -1e9)                           # padding can't be chosen
        w = F.softmax(score, dim=-1)                                    # scores -> percentages that add to 1
        context = torch.bmm(w.unsqueeze(1), memory).squeeze(1)          # blend the letter cards
        return context, w

class Prenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(C.N_MELS, C.PRENET_DIM)
        self.l2 = nn.Linear(C.PRENET_DIM, C.PRENET_DIM)

    def forward(self, x):
        # training=True means dropout is ALWAYS on, even when speaking (on purpose)
        x = F.dropout(F.relu(self.l1(x)), 0.5, training=True)
        x = F.dropout(F.relu(self.l2(x)), 0.5, training=True)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenet = Prenet()
        self.attn_rnn = nn.LSTMCell(C.PRENET_DIM + C.ENC_DIM, C.ATT_RNN_DIM)
        self.attn = LocationSensitiveAttention()
        self.dec_rnn = nn.LSTMCell(C.ATT_RNN_DIM + C.ENC_DIM, C.DEC_RNN_DIM)
        self.mel_proj = nn.Linear(C.DEC_RNN_DIM + C.ENC_DIM, C.N_MELS * C.R)   # the 2 columns
        self.stop_proj = nn.Linear(C.DEC_RNN_DIM + C.ENC_DIM, 1)               # the "done?" signal

    def init_state(self, memory):
        # a fresh, empty notepad at the start of a sentence
        B, T, _ = memory.shape
        zeros = lambda size: memory.new_zeros(B, size)
        return dict(
            ah=zeros(C.ATT_RNN_DIM), ac=zeros(C.ATT_RNN_DIM),
            dh=zeros(C.DEC_RNN_DIM), dc=zeros(C.DEC_RNN_DIM),
            w=memory.new_zeros(B, T), w_cum=memory.new_zeros(B, T),
            ctx=zeros(C.ENC_DIM),
        )

    def step(self, prev_frame, st, memory, proc_memory, mask):
        # 1. fuzz the previous column and join it with the last context card
        x = torch.cat([self.prenet(prev_frame), st["ctx"]], dim=-1)

        # 2. attention RNN updates its memory -> this becomes the query
        st["ah"], st["ac"] = self.attn_rnn(x, (st["ah"], st["ac"]))
        ah = F.dropout(st["ah"], 0.1, self.training)

        # 3. move the attention finger
        prev_w = torch.stack([st["w"], st["w_cum"]], dim=1)
        ctx, w = self.attn(ah, memory, proc_memory, prev_w, mask)
        st["w_cum"] = st["w_cum"] + w
        st["w"], st["ctx"] = w, ctx

        # 4. decoder RNN combines both and decides the next sound
        st["dh"], st["dc"] = self.dec_rnn(torch.cat([ah, ctx], dim=-1), (st["dh"], st["dc"]))
        dh = F.dropout(st["dh"], 0.1, self.training)
        out = torch.cat([dh, ctx], dim=-1)
        return self.mel_proj(out), self.stop_proj(out).squeeze(-1), w

class Postnet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, ch = [], C.N_MELS
        for i in range(5):
            out = C.N_MELS if i == 4 else C.POSTNET_DIM       # last layer returns to 80 bands
            layers += [nn.Conv1d(ch, out, kernel_size=5, padding=2), nn.BatchNorm1d(out)]
            if i < 4:
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.5))
            ch = out
        self.net = nn.Sequential(*layers)

    def forward(self, mel):                                   # mel: (batch, columns, 80)
        return self.net(mel.transpose(1, 2)).transpose(1, 2)  # returns the "correction"


class Tacotron(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, ids, ids_len, mels):
        """TRAINING: teacher forcing. We feed the decoder the TRUE previous column."""
        memory = self.encoder(ids, ids_len)                       # letter cards
        mask = ~sequence_mask(ids_len, ids.size(1))               # True on padding
        proc = self.decoder.attn.memory(memory)

        B, T, _ = mels.shape
        steps = T // C.R
        go = mels.new_zeros(B, 1, C.N_MELS)                       # silence for step 0
        last_of_each_group = mels[:, C.R - 1 :: C.R]              # c1, c3, c5, ...
        prev = torch.cat([go, last_of_each_group[:, :-1]], dim=1) # shift right by one step

        st = self.decoder.init_state(memory)
        outs, stops, aligns = [], [], []
        for i in range(steps):
            mel, stop, w = self.decoder.step(prev[:, i], st, memory, proc, mask)
            outs.append(mel.view(B, C.R, C.N_MELS))
            stops.append(stop)
            aligns.append(w)

        mel_out = torch.cat(outs, dim=1)                          # rough grid
        mel_post = mel_out + self.postnet(mel_out)                # polished grid
        return mel_out, mel_post, torch.stack(stops, dim=1), torch.stack(aligns, dim=1)

    @torch.no_grad()
    def inference(self, ids, max_steps=500, stop_threshold=0.5):
        """SPEAKING: one sentence (ids shape (1, letters)). Feeds on its own output."""
        lens = torch.LongTensor([ids.size(1)]).to(ids.device)
        memory = self.encoder(ids, lens)
        mask = ~sequence_mask(lens, ids.size(1))
        proc = self.decoder.attn.memory(memory)
        st = self.decoder.init_state(memory)

        prev = memory.new_zeros(1, C.N_MELS)                      # start from silence
        outs, aligns = [], []
        for i in range(max_steps):
            mel, stop, w = self.decoder.step(prev, st, memory, proc, mask)
            mel = mel.view(1, C.R, C.N_MELS)
            outs.append(mel)
            aligns.append(w)
            prev = mel[:, -1]                                     # its OWN last column
            if i > 5 and torch.sigmoid(stop).item() > stop_threshold:
                break                                             # "I'm done"

        mel_out = torch.cat(outs, dim=1)
        mel_post = mel_out + self.postnet(mel_out)
        return mel_post[0].clamp(0, 1), torch.stack(aligns, dim=1)[0]