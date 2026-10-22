# RAE-Former: A Decoupled Spatio-Temporal Pipeline for EEG-Based Emotion Recognition

[![Conference](https://img.shields.io/badge/ICVGIP-2026-blue.svg)](https://icvgip.in/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper:  
**"RAE-Former: A Decoupled Spatio-Temporal Pipeline for EEG-Based Emotion Recognition"**  
*Accepted at the 17th Indian Conference on Computer Vision, Graphics, and Image Processing (ICVGIP 2026).*

**Authors:** Raj Jamwal, Jyoti Nigam, Abhishek Rathore, Ravi Kumar, and Arnav Bhavsar

---

##  Overview

Traditional deep learning models for EEG emotion recognition often suffer from *spatial-temporal entanglement*—forcing a single network to simultaneously learn spatial scalp topology and temporal sequence transitions, which may lead to subject-identity memorization and poor generalization on unseen users.

**RAE-Former** resolves this by physically decoupling spatial and temporal learning into two phases:
1. **Phase 1 (Self-Supervised Residual Autoencoder - RAE):** Compresses multi-band Differential Entropy (DE) topographical maps into filtered 256-D spatial embeddings using per-sample Instance Normalization. Crucially, during Leave-One-Subject-Out (LOSO) evaluation, the RAE is trained strictly from scratch on the N-1 training subjects in each fold to guarantee zero data leakage.
2. **Phase 2 (Temporal Transformer with Differential Optimization):** Processes 5-second sequences of spatial embeddings using Multi-Head Self-Attention while gently fine-tuning the spatial encoder.

---

##  Repository Structure

```text
├── loso/                               # Strict Leave-One-Subject-Out (LOSO) evaluation scripts
│   ├── deap_loso.py                    # 32-Fold LOSO for deap
│   └── dreamer_loso.py                 # 23-Fold LOSO for dreamer
│   └── without_DE_loso.py              # ablation without differential features
│   └── vanilla_AE_loso.py              # 32-Fold LOSO with vanilla AutoEncoder ablation
├── subject_dependent/                  # 10-Fold Trial-Wise Subject-Dependent benchmarks
│   ├── trialwise_with_ablation.py      # RAE-Former (S) variant with band ablation
│   └── trialwise_heavy.py              # RAE-Former (H) variant
│   └── entangled_ablation_trialwise.py # entangled ablation in subject dependent
├── different models/                   # reproducing different architecture on similar setting
│   ├── comparison.py                   # model comparison code
└── requirements.txt                    # Python dependencies
```

---

## Installation & Dataset Preparation

### 1. Install Dependencies
```bash
git clone https://github.com/rajjamwal/RAE-Former.git
cd RAE-Former
pip install -r requirements.txt
```

### 2. Prepare Datasets
* **DEAP Dataset:** Request access from the [official DEAP website](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and place the 32 preprocessed Python `.dat` files (`s01.dat` to `s32.dat`) inside `./data/deap/`.
* **DREAMER Dataset:** Download `DREAMER.mat` from [Zenodo](https://zenodo.org/record/546113) and place it inside `./data/dreamer/DREAMER.mat`.

*(Update `DEAP_PATH` or `DREAMER_FILE` inside the `Config` class of each script to match your local dataset path and run using `python script_name.py`).*

---

##  Key Results (Strict Leak-Free LOSO)

| Method / Pipeline | DEAP Valence (%) | DEAP Arousal (%) | DREAMER Valence (%) | DREAMER Arousal (%) |
| :--- | :---: | :---: | :---: | :---: |
| EEG-Conformer | 57.50 | 62.66 | **72.22** | 69.08 |
| EEGNet | 58.04 | 61.64 | 69.08 | 62.56 |
| FAT | 58.98 | 60.31 | 69.80 | 67.14 |
| TSception | 59.37 | 62.81 | 70.28 | 68.11 |
| Vanilla AE + Trans. (Entangled) | 50.22 ± 17.61 | 52.72 ± 16.54 | 60.87 ± 15.92 | 59.50 ± 13.53 |
| **RAE-Former (Ours)** | **64.77 ± 6.14** | **65.39 ± 7.73** | 70.05 ± 9.52 | **69.10 ± 9.34** | 

---

## Citation

If you find this repository or our paper useful in your research, please cite:

```bibtex
@inproceedings{jamwal2026raeformer,
  author    = {Jamwal, Raj and Nigam, Jyoti and Rathore, Abhishek and Kumar, Ravi and Bhavsar, Arnav},
  title     = {RAE-Former: A Decoupled Spatio-Temporal Pipeline for EEG-Based Emotion Recognition},
  booktitle = {Indian Conference on Computer Vision, Graphics, and Image Processing (ICVGIP 2026)},
  year      = {2026},
  publisher = {ACM}
}
```
