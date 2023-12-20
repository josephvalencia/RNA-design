## RNA design using seqopt
["Extrapolative benchmarking of model-based discrete sampling methods for RNA design"](assets/MLCB_Poster.pdf), presented at the 2023 Machine Learning in Computational Biology conference. 

First, setup a virtual environment and install its dependencies, including `seqopt`
```
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/josephvalencia/RNA-design.git
```

I trained regression models of three important RNA properties, using data from prior works.
1. Mean ribosome load of 5' UTRs [(Sample et al, 2019)](https://www.nature.com/articles/s41587-019-0164-5). Ribosomal occupancy of the upstream region of RNAs indicates translational activity. This dataset consists of random UTRs of length 50 nt assayed with polysome profiling. 
2. Toehold switch ON/OFF ratio [(Valeri et al, 2020)](https://www.nature.com/articles/s41467-020-18676-2). Toehold switches are two-state riboregulators which natively sequester the translation initation site (OFF) upstream of a coding sequence but adopt an alternate secondary structure in the presence of a trigger RNA ligand, exposing the translation initiation site (ON).  
3. Messenger RNA half-life [(Agarwal and Kelley, 2022)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x). RNA degradation regulates protein expression and impacts the design of mRNA therapeutics. This dataset consists of human and mouse mRNAs with a cell-type independent measure of half-life. I omitted the auxiliary codon position and splice site tracks used in the original Saluki model, resulting in significantly lower performance but permitting unrestricted sequence design.

For simplicity, I chose prior works which used deep learning to build property models for scalar properties. I defined a common architecture inspired by Saluki, consisting of
`One-hot encoding -> bi-GRU -> ByteNet + MaxPool1D(k=2) (x N blocks) -> bi-GRU` and a final `Dense` layer on the leading token position.

# Train the weak models
These models will have the top 10% of property values excluded from training and will drive the design process. If a GPU device is unavailable, pass `--device -1`

```
# MRL
python train_5_prime.py --decile_limit 9
# Toehold switch ON/OFF ratio
python train_toeholds.py --decile_limit 9 
# mRNA half-life
python train_saluki.py --decile_limit 9
```

# Train the strong models
These models will learn from the full range of training data and be used to evaluate designs.
```
python train_5_prime.py 
python train_toeholds.py 
python train_saluki.py 
```

# Design sequences
Use a Lightning checkpoint for a 'weak' model to produce UTRs with high predicted ribosome load. 
```
python design_5_prime.py --checkpoint <CHECKPOINT>.ckpt --n_iter 1000 --tol 1000 --n_samples 4 --device 0 --lr 1e-1 --n_trials 50                                                                                          ```

