"""
train_bilstm_crf_char.py
─────────────────────
BiLSTM-CRF with a character-level encoder (char-CNN; Ma and Hovy, 2016; Lample et al., 2016).
Network, training, partitions and evaluation are those of train_bilstm_crf.py; only the
word representation changes: each word vector (learnt from scratch) is concatenated with
a 50-dimensional vector computed by a CNN over its characters. A word absent from training
is therefore no longer a single shared vector, but keeps its form (suffixes and prefixes;
not case, since tokens are lowercased).

Output in $RESULTADOS/bilstm_char/ (runs, pred, epocas), with the same file names.
Usage: as train_bilstm_crf.py, e.g.
  python train_bilstm_crf_char.py reserva --lang es --niveles 20 --semillas 0,1,2,3,4
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_bilstm_crf as B  # noqa: E402
from torchcrf import CRF as TorchCRF  # noqa: E402

B.OUT = Path(os.environ.get("RESULTADOS", HERE.parent / "resultados")) / "bilstm_char"
for d in ("runs", "pred", "epocas"):
    (B.OUT / d).mkdir(parents=True, exist_ok=True)

MAX_CHARS = 20
# Fixed character inventory (independent of the data): 0 = padding, 1 = unknown
ALFABETO = "abcdefghijklmnopqrstuvwxyzáéíóúüñàèìòùâêîôûçäëïöß0123456789-'./_%&()"
CHARS = {c: i + 2 for i, c in enumerate(ALFABETO)}


def ids_chars(token):
    return [CHARS.get(c, 1) for c in token[:MAX_CHARS]]


def codificar(oraciones, vocab):
    """Each token as [word id, id of character 1, ..., id of character MAX_CHARS]."""
    datos = []
    for o in oraciones:
        X = np.zeros((len(o["tokens"]), 1 + MAX_CHARS), dtype=np.int64)
        for k, t in enumerate(o["tokens"]):
            X[k, 0] = vocab.get(t, 1)
            c = ids_chars(t)
            X[k, 1:1 + len(c)] = c
        datos.append((X, np.array([B.LABELS[l] for l in o["labels"]], dtype=np.int64)))
    return datos


def lote(datos, idx):
    largo = max(len(datos[i][0]) for i in idx)
    X = np.zeros((len(idx), largo, 1 + MAX_CHARS), dtype=np.int64)
    Y = np.zeros((len(idx), largo), dtype=np.int64)
    M = np.zeros((len(idx), largo), dtype=bool)
    for k, i in enumerate(idx):
        n = len(datos[i][0])
        X[k, :n], Y[k, :n], M[k, :n] = datos[i][0], datos[i][1], True
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


class CharBiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.char_emb = nn.Embedding(len(CHARS) + 2, 30, padding_idx=0)
        self.char_cnn = nn.Conv1d(30, 50, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128 + 50, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_tags)
        self.crf = TorchCRF(num_tags, batch_first=True)

    def representar(self, x):
        palabras, chars = x[..., 0], x[..., 1:]
        b, l, w = chars.shape
        c = self.char_emb(chars.reshape(b * l, w)).transpose(1, 2)        # (b·l, 30, w)
        c = torch.relu(self.char_cnn(c))
        c = c.masked_fill((chars.reshape(b * l, w) == 0).unsqueeze(1), -1e4).max(dim=2).values
        c = torch.where(c < -1e3, torch.zeros_like(c), c).reshape(b, l, 50)
        return torch.cat([self.embedding(palabras), c], dim=-1)

    def forward(self, x, tags=None, mask=None):
        largos = mask.sum(1).cpu()
        emb = nn.utils.rnn.pack_padded_sequence(self.representar(x), largos, batch_first=True, enforce_sorted=False)
        salida, _ = self.lstm(emb)
        salida, _ = nn.utils.rnn.pad_packed_sequence(salida, batch_first=True, total_length=x.size(1))
        emissions = self.fc(salida)
        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction=os.environ.get("REDUCCION", "mean"))
        return self.crf.decode(emissions, mask=mask)


# Replace these pieces of the base module; everything else (training, evaluation, runs) is unchanged
B.codificar = codificar
B.lote = lote
B.BiLSTMCRF = CharBiLSTMCRF

if __name__ == "__main__":
    B.main()
