"""
Exploratory data analysis for the report: dataset size, sentence lengths,
vocabulary sizes, most frequent words. Saves figures in results/.
"""

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt

from data import load_pairs, tokenize

os.makedirs("results", exist_ok=True)
pairs = load_pairs(sys.argv[1] if len(sys.argv) > 1 else None)
en = [tokenize(e) for e, _ in pairs]
da = [tokenize(d) for _, d in pairs]

print("Number of sentence pairs:", len(pairs))
for lang, sents in [("English", en), ("Darija", da)]:
    lengths = [len(s) for s in sents]
    counts = Counter(t for s in sents for t in s)
    print(f"\n{lang}: average length {sum(lengths) / len(lengths):.1f} words, "
          f"max {max(lengths)}, vocabulary {len(counts)} words, "
          f"words seen only once {sum(1 for n in counts.values() if n == 1)}")
    print("  most frequent:", [w for w, _ in counts.most_common(15)])

plt.hist([len(s) for s in en], bins=40, range=(0, 40), alpha=0.6, label="English")
plt.hist([len(s) for s in da], bins=40, range=(0, 40), alpha=0.6, label="Darija")
plt.xlabel("sentence length (words)")
plt.ylabel("number of sentences")
plt.legend()
plt.title("Sentence lengths")
plt.savefig("results/eda_lengths.png", dpi=150, bbox_inches="tight")
print("\nSaved results/eda_lengths.png")
