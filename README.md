# English → Darija translation with hand-made LSTM cells

Lab project, Advanced NLP 2026/2027. We translate English sentences into
Moroccan Darija (Latin letters) with an encoder-decoder (seq2seq) model.
All LSTM cells are written by hand with `nn.Linear` (no `nn.LSTM`).

Dataset: [imomayiz/darija-english](https://huggingface.co/datasets/imomayiz/darija-english) (config `sentences`).

## Files

| File | What it does |
|---|---|
| `cells.py` | The 4 cells: Vanilla LSTM, CIFG, LSTMP, WMC, plus `LSTM` which runs a cell over a sequence |
| `data.py` | Loads the data, tokenizes, builds the vocabularies, makes the 80/10/10 split |
| `model.py` | Encoder-decoder model, greedy translation, parameter count, weight init |
| `train.py` | Trains one model, keeps the best one on validation, tests it (loss, BLEU, chrF) |
| `eda.py` | Dataset statistics and a sentence-length figure for the report |
| `plot_results.py` | Validation-loss curves of several runs on one figure |
| `run_experiments.sh` | Every experiment of the project |
| `kaggle_run.ipynb` | Notebook to run everything on a free Kaggle GPU |

## How the model works

1. The English words become vectors (`nn.Embedding`).
2. The **encoder** (stacked LSTM cells) reads the sentence word by word. Its last state (h, c) is a summary of the sentence.
3. The **decoder** (same cell type) starts from that state and predicts the Darija sentence one word at a time.
4. Training uses **teacher forcing**: the decoder gets the correct previous word as input. The loss is cross-entropy.
5. At test time we use **greedy decoding**: each predicted word is fed back as the next input, until `<eos>`.

## The cells

| Cell | Idea | Difference with vanilla |
|---|---|---|
| Vanilla | 4 gates: input, forget, output, candidate | baseline |
| CIFG | input gate = 1 − forget gate | 3 gates instead of 4, fewer parameters |
| LSTMP | output is projected to a smaller size P | smaller recurrent matrices, fewer parameters |
| WMC (optional) | gates also look at the memory cell c | more parameters |

## Run it

On your computer (VS Code):
```bash
pip install -r requirements.txt
python train.py --cell vanilla --epochs 2 --max_samples 2000   # quick test
```

On Kaggle (GPU): open `kaggle_run.ipynb` there and run all cells.

Useful options of `train.py`: `--cell`, `--lr`, `--optimizer`, `--batch_size`,
`--init`, `--epochs`, `--hidden_size`, `--num_layers`, `--proj_size`, `--max_samples`.

Results go to `results/results.csv` (one line per run: parameters, losses, BLEU, chrF, time).
