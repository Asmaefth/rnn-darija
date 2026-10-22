#!/bin/bash
# All experiments of the project. Every run uses the same data split (seed 42).
set -e

# 1) Compare the cells, with the same hyperparameters
python train.py --cell vanilla --name vanilla
python train.py --cell cifg    --name cifg
python train.py --cell lstmp   --name lstmp
# python train.py --cell wmc   --name wmc      # optional 4th model

# 2) Hyperparameters, changed one at a time on the vanilla model
python train.py --cell vanilla --lr 0.003  --name vanilla_lr3e-3
python train.py --cell vanilla --lr 0.0003 --name vanilla_lr3e-4
python train.py --cell vanilla --optimizer sgd --lr 0.1 --name vanilla_sgd
python train.py --cell vanilla --optimizer rmsprop      --name vanilla_rmsprop
python train.py --cell vanilla --batch_size 32  --name vanilla_bs32
python train.py --cell vanilla --batch_size 128 --name vanilla_bs128
python train.py --cell vanilla --init xavier     --name vanilla_xavier
python train.py --cell vanilla --init orthogonal --name vanilla_orthogonal

# 3) Figures
python plot_results.py vanilla cifg lstmp
python plot_results.py vanilla vanilla_lr3e-3 vanilla_lr3e-4
python plot_results.py vanilla vanilla_sgd vanilla_rmsprop
python plot_results.py vanilla vanilla_bs32 vanilla_bs128
python plot_results.py vanilla vanilla_xavier vanilla_orthogonal
