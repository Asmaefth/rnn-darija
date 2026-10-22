"""
Data: load the English-Darija sentence pairs, tokenize, build vocabularies,
split into train / validation / test, and make DataLoaders.
"""

import random
import re
from collections import Counter

import torch
from torch.utils.data import DataLoader

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


def load_pairs(csv_path=None):
    """Returns a list of (english, darija) sentence pairs.
    By default we download the dataset from Hugging Face. You can also give
    a CSV file with the columns 'eng' and 'darija'."""
    if csv_path:
        import pandas as pd
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        df = load_dataset("imomayiz/darija-english", "sentences", split="sentences").to_pandas()
    print("Columns in the dataset:", list(df.columns))
    df = df.rename(columns={"english": "eng", "en": "eng"})
    df = df[["eng", "darija"]].dropna()
    df = df.drop_duplicates()
    return list(zip(df["eng"].astype(str), df["darija"].astype(str)))


def tokenize(sentence):
    """Lowercase, then split into words and punctuation.
    Digits are kept inside words because Darija in Latin letters uses them
    as letters (3, 7, 9...)."""
    return re.findall(r"\w+|[^\w\s]", sentence.lower())


class Vocab:
    def __init__(self, sentences, min_freq=2):
        counts = Counter(tok for s in sentences for tok in s)
        self.itos = SPECIALS + sorted(t for t, n in counts.items() if n >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids, skip_unk=False):
        words = []
        for i in ids:
            if i == EOS:
                break
            if i in (PAD, SOS) or (skip_unk and i == UNK):
                continue
            words.append(self.itos[i])
        return " ".join(words)


def split_data(pairs, seed=42, val_ratio=0.1, test_ratio=0.1):
    """Same seed -> same split, so every model is compared on the same data."""
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    n_test = int(len(pairs) * test_ratio)
    n_val = int(len(pairs) * val_ratio)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


def make_collate(src_vocab, tgt_vocab, max_len):
    def collate(batch):
        src_ids, tgt_ids = [], []
        for src_tokens, tgt_tokens in batch:
            src_ids.append(src_vocab.encode(src_tokens[:max_len]))
            tgt_ids.append([SOS] + tgt_vocab.encode(tgt_tokens[:max_len]) + [EOS])
        return pad(src_ids), pad(tgt_ids)
    return collate


def pad(sequences):
    longest = max(len(s) for s in sequences)
    out = torch.full((len(sequences), longest), PAD, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, :len(s)] = torch.tensor(s)
    return out


def get_data(batch_size=64, max_len=30, max_samples=None, csv_path=None, seed=42):
    pairs = load_pairs(csv_path)
    pairs = [(tokenize(en), tokenize(da)) for en, da in pairs]
    pairs = [(en, da) for en, da in pairs if en and da]
    if max_samples:
        pairs = pairs[:max_samples]
    train, val, test = split_data(pairs, seed)

    # vocabularies are built on the training set only
    src_vocab = Vocab([en for en, _ in train])
    tgt_vocab = Vocab([da for _, da in train])

    collate = make_collate(src_vocab, tgt_vocab, max_len)
    loaders = {
        "train": DataLoader(train, batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val, batch_size, collate_fn=collate),
        "test": DataLoader(test, batch_size, collate_fn=collate),
    }
    print(f"train={len(train)} val={len(val)} test={len(test)} | "
          f"English vocab={len(src_vocab)} Darija vocab={len(tgt_vocab)}")
    # the raw test pairs are also returned, to compute BLEU on the real words
    return loaders, src_vocab, tgt_vocab, test
