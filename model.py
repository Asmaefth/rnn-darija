"""
Encoder-decoder (seq2seq) model for translation.

1. The encoder reads the English sentence and summarizes it in its final
   state (h, c).
2. The decoder starts from that state and writes the Darija sentence one
   word at a time.
3. Optional attention (--attention): at each step the decoder also looks at
   ALL the encoder outputs (one per English word), not only the final state.
"""

import torch
import torch.nn as nn

from cells import LSTM
from data import PAD, SOS, EOS


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cell_type="vanilla",
                 emb_size=256, hidden_size=512, num_layers=2, proj_size=None,
                 dropout=0.3, attention=False):
        super().__init__()
        self.attention = attention
        self.src_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=PAD)
        self.encoder = LSTM(cell_type, emb_size, hidden_size, num_layers, proj_size, dropout)
        self.decoder = LSTM(cell_type, emb_size, hidden_size, num_layers, proj_size, dropout)
        self.dropout = nn.Dropout(dropout)
        size = self.decoder.output_size
        if attention:
            # mixes the decoder output with the context vector: tanh(Wc [h; context])
            self.combine = nn.Linear(2 * size, size)
        self.out = nn.Linear(size, tgt_vocab_size)

    def encode(self, src):
        mask = (src != PAD).float()
        enc_outputs, state = self.encoder(self.dropout(self.src_emb(src)), mask=mask)
        return enc_outputs, state, mask

    def attend(self, dec_outputs, enc_outputs, mask):
        """Luong (dot-product) attention.
        dec_outputs: (batch, tgt_len, size), enc_outputs: (batch, src_len, size)
        1. score = how similar each decoder step is to each English word
        2. softmax -> weights that sum to 1 (padding gets weight 0)
        3. context = weighted average of the encoder outputs"""
        scores = dec_outputs @ enc_outputs.transpose(1, 2)           # (batch, tgt_len, src_len)
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ enc_outputs                               # (batch, tgt_len, size)
        return torch.tanh(self.combine(torch.cat([dec_outputs, context], dim=-1)))

    def predict(self, dec_outputs, enc_outputs, mask):
        if self.attention:
            dec_outputs = self.attend(dec_outputs, enc_outputs, mask)
        return self.out(self.dropout(dec_outputs))

    def forward(self, src, tgt_in):
        """Training with teacher forcing: the decoder sees the correct previous
        word at every step. Returns scores (batch, tgt_len, tgt_vocab_size)."""
        enc_outputs, state, mask = self.encode(src)
        dec_outputs, _ = self.decoder(self.dropout(self.tgt_emb(tgt_in)), state)
        return self.predict(dec_outputs, enc_outputs, mask)

    @torch.no_grad()
    def translate(self, src, max_len=40):
        """Greedy decoding: at each step we keep the most likely word and give
        it back to the decoder as the next input."""
        enc_outputs, state, mask = self.encode(src)
        word = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)
        result = []
        for _ in range(max_len):
            output, state = self.decoder(self.tgt_emb(word), state)
            word = self.predict(output, enc_outputs, mask).argmax(dim=-1)  # (batch, 1)
            result.append(word)
            if (torch.cat(result, dim=1) == EOS).any(dim=1).all():
                break
        return torch.cat(result, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(model, method):
    """'default' keeps PyTorch's init. 'xavier' and 'orthogonal' change the
    weight matrices of the Linear layers (biases are set to 0)."""
    if method == "default":
        return
    for name, p in model.named_parameters():
        if "emb" in name:
            continue
        if p.dim() == 2:
            if method == "xavier":
                nn.init.xavier_uniform_(p)
            elif method == "orthogonal":
                nn.init.orthogonal_(p)
        else:
            nn.init.zeros_(p)
