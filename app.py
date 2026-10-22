"""
Web demo: translate English to Darija in the browser with our trained models.

How to run (in the rnn-darija folder, with the venv activated):
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in your browser.

The trained models must be in checkpoints/ (vanilla.pt, cifg.pt, lstmp.pt,
and vanilla_att.pt, cifg_att.pt, lstmp_att.pt for the attention versions).
"""

import os

import torch
from flask import Flask, jsonify, render_template, request

from data import get_data, pad, tokenize
from model import Seq2Seq

# checkpoint name -> (cell type, attention?, label shown on the page)
MODELS = {
    "vanilla": ("vanilla", False, "Vanilla LSTM"),
    "cifg": ("cifg", False, "CIFG"),
    "lstmp": ("lstmp", False, "LSTMP"),
    "vanilla_att": ("vanilla", True, "Vanilla + attention"),
    "cifg_att": ("cifg", True, "CIFG + attention"),
    "lstmp_att": ("lstmp", True, "LSTMP + attention"),
}

# 1) Rebuild the vocabularies. Same data + same seed = the same vocabularies
#    as during training, so the word ids match the saved models.
_, src_vocab, tgt_vocab, _ = get_data(csv_path=os.environ.get("DATA_CSV"))

# 2) Load every trained model we can find in checkpoints/.
#    The sizes are the defaults of train.py.
models = {}
for name, (cell, attention, _) in MODELS.items():
    path = f"checkpoints/{name}.pt"
    if os.path.exists(path):
        model = Seq2Seq(len(src_vocab), len(tgt_vocab), cell_type=cell,
                        emb_size=256, hidden_size=512, num_layers=2, proj_size=256,
                        attention=attention)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[name] = model
        print(f"Loaded {path}")
if not models:
    print("No model found in checkpoints/. Download vanilla.pt, cifg.pt, lstmp.pt from Kaggle.")


def translate(sentence, name):
    tokens = tokenize(sentence)[:30]
    if not tokens:
        return ""
    src = pad([src_vocab.encode(tokens)])
    pred = models[name].translate(src)[0].tolist()
    return tgt_vocab.decode(pred, skip_unk=True)


app = Flask(__name__)


@app.route("/")
def home():
    available = {name: MODELS[name][2] for name in MODELS if name in models}
    return render_template("index.html", models=available)


@app.route("/translate", methods=["POST"])
def translate_route():
    text = request.json.get("text", "")
    names = request.json.get("models", list(models))
    result = {name: translate(text, name) for name in names if name in models}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)
