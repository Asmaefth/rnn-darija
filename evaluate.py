"""
Recomputes BLEU and chrF for runs that are already trained, using the saved
checkpoints (no new training). Writes results/results_fixed.csv.

Example:
    python evaluate.py
"""

import csv
import json
import os

import torch

from data import get_data
from model import Seq2Seq
from train import evaluate_bleu

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("results/results.csv") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    name = row["name"]
    with open(f"results/{name}_history.json") as f:
        history = json.load(f)
    args = history["args"]
    _, src_vocab, tgt_vocab, test_pairs = get_data(max_samples=args["max_samples"],
                                                   csv_path=args["csv"], seed=args["seed"])
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), args["cell"], args["emb_size"],
                    args["hidden_size"], args["num_layers"], args["proj_size"],
                    args["dropout"], args.get("attention", False)).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location=device))
    bleu, chrf, examples = evaluate_bleu(model, test_pairs, src_vocab, tgt_vocab, device)
    row["bleu"], row["chrf"] = round(bleu, 2), round(chrf, 2)
    history["examples"] = examples
    with open(f"results/{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"{name}: BLEU {bleu:.2f} | chrF {chrf:.2f}")

with open("results/results_fixed.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("Saved results/results_fixed.csv")
