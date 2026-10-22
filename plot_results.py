"""
Plots the validation loss of several runs on one figure.

Example:
    python plot_results.py vanilla cifg lstmp
"""

import json
import sys

import matplotlib.pyplot as plt

names = sys.argv[1:]
for name in names:
    with open(f"results/{name}_history.json") as f:
        history = json.load(f)
    plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label=name)
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.legend()
out = "results/val_loss_" + "_vs_".join(names) + ".png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
