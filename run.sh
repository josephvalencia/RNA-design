#!/usr/bin/env bash
python design_5_prime.py --checkpoint models/optimus_5prime/checkpoints/frosty-river-159-epoch=12-val_loss=0.0818.ckpt --n_iter 10000 --tol 10000 --n_samples 4 --device 0 --lr 1e-1 --n_trials 1
#python design_toeholds.py --checkpoint models/toeholds/checkpoints/rose-fog-9-epoch=11-val_loss=0.0257_val_r2=0.6112.ckpt --property "Toehold On/Off Ratio" --max_train_val 0.654823 --n_iter 1000 --tol 500 --n_samples 4 --lr 1e-1 --optimizer adam --device 1 --n_trials 2
#python design_mrna_stability.py --checkpoint models/saluki/checkpoints/young-voice-15-epoch=6-val_loss=0.6450.ckpt --n_iter 1000 --tol 1000 --n_samples 1 --lr 1e-1 --device 1
