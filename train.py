"""
Train one model and evaluate it on the test set.

Example:
    python train.py --cell vanilla --name vanilla_baseline
    python train.py --cell cifg --lr 0.003 --name cifg_lr3e-3

Each run adds one line to results/results.csv and saves its loss curves
in results/<name>_history.json.
"""

import argparse
import csv
import json
import os
import random
import time

import sacrebleu
import torch
import torch.nn as nn

from data import PAD, get_data, pad
from model import Seq2Seq, count_parameters, init_weights


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None, help="name of the run")
    p.add_argument("--cell", default="vanilla", choices=["vanilla", "cifg", "lstmp", "wmc"])
    p.add_argument("--emb_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--proj_size", type=int, default=256, help="only for lstmp")
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "rmsprop"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--init", default="default", choices=["default", "xavier", "orthogonal"])
    p.add_argument("--max_samples", type=int, default=None, help="use less data")
    p.add_argument("--csv", default=None, help="local CSV instead of Hugging Face")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. If an optimizer is given we train, else we evaluate."""
    model.train(optimizer is not None)
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # input: <sos> w1 w2 ... / target: w1 w2 ... <eos>
        with torch.set_grad_enabled(optimizer is not None):
            scores = model(src, tgt_in)
            loss = loss_fn(scores.reshape(-1, scores.size(-1)), tgt_out.reshape(-1))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoids exploding gradients
            optimizer.step()
        n = (tgt_out != PAD).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens


def evaluate_bleu(model, test_pairs, src_vocab, tgt_vocab, device, batch_size=128):
    """BLEU and chrF on the test set.
    The references are the REAL Darija words (not the vocabulary ids), and we
    remove <unk> from the model output. Otherwise a model that only writes
    <unk> would get a high score, because the references also contain <unk>."""
    model.eval()
    hypotheses, references, sources = [], [], []
    for start in range(0, len(test_pairs), batch_size):
        batch = test_pairs[start:start + batch_size]
        src = pad([src_vocab.encode(en[:30]) for en, _ in batch]).to(device)
        pred = model.translate(src).cpu()
        for (en, da), p in zip(batch, pred.tolist()):
            sources.append(" ".join(en))
            references.append(" ".join(da))
            hypotheses.append(tgt_vocab.decode(p, skip_unk=True))
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    examples = list(zip(sources, references, hypotheses))[:10]
    return bleu, chrf, examples


def main():
    args = get_args()
    name = args.name or f"{args.cell}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{args.init}"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    loaders, src_vocab, tgt_vocab, test_pairs = get_data(args.batch_size, max_samples=args.max_samples,
                                             csv_path=args.csv, seed=args.seed)

    model = Seq2Seq(len(src_vocab), len(tgt_vocab), args.cell, args.emb_size,
                    args.hidden_size, args.num_layers, args.proj_size, args.dropout).to(device)
    init_weights(model, args.init)
    n_params = count_parameters(model)
    print(f"Run {name} | cell={args.cell} | trainable parameters={n_params:,} | device={device}")

    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    history = {"train_loss": [], "val_loss": []}
    best_val, checkpoint = float("inf"), f"checkpoints/{name}.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss = run_epoch(model, loaders["val"], loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} | val loss {val_loss:.3f}")
        if val_loss < best_val:  # keep the model with the best validation loss
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)
    train_time = time.time() - start

    # final evaluation on the test set with the best model
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss = run_epoch(model, loaders["test"], loss_fn, device)
    bleu, chrf, examples = evaluate_bleu(model, test_pairs, src_vocab, tgt_vocab, device)
    print(f"TEST | loss {test_loss:.3f} | BLEU {bleu:.2f} | chrF {chrf:.2f}")
    for en, ref, hyp in examples[:5]:
        print(f"  EN : {en}\n  REF: {ref}\n  OUT: {hyp}\n")

    with open(f"results/{name}_history.json", "w") as f:
        json.dump({**history, "args": vars(args), "examples": examples}, f, indent=2)
    row = {"name": name, "cell": args.cell, "params": n_params, "optimizer": args.optimizer,
           "lr": args.lr, "batch_size": args.batch_size, "init": args.init,
           "best_val_loss": round(best_val, 3), "test_loss": round(test_loss, 3),
           "bleu": round(bleu, 2), "chrf": round(chrf, 2), "minutes": round(train_time / 60, 1)}
    path = "results/results.csv"
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
