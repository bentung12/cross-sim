# SONOS Radiation Simulation & Correction

## 📌 Project Overview

This project investigates how **in-situ analog computing (SONOS)** can perform machine learning inference under radiation effects.
Radiation exposure causes **drift in conductance values**, which degrades accuracy over time.

We recreate and extend these experimental results in Python, using simulation, correction factors, and training strategies to mitigate drift.

This particular folder only contains data for the single layer alpha correction factor (ACF), while the full network training/inference is ![here](../tiny_imagenet).

---

## 📂 Repository Structure

```
├── plots.py                  # Generates accuracy/TID plots, roll-off tables
├── radiation_core.py         # Core SONOS drift simulations (no alpha)
├── radiation_core_alpha.py   # Alpha correction implementations
├── data/                     # Stores collected inference data
├── TIN_classifier_data/      # Single-layer experiment data for SONOS
├── extra_files/              # Miscellaneous, not central to experiments
└── base_models/              # Pretrained models (separate folder, not here)
```



---

## ⚙️ Setup

### Requirements

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib
* [CrossSim (pytorch branch)](https://github.com/sandialabs/cross-sim/tree/pytorch)&#x20;

### Installation

1. Clone this repository and install dependencies:

   ```bash
   git clone <your-repo-link>
   cd <your-repo>
   pip install -r requirements.txt
   ```
2. Download and set up **CrossSim** (see link above).

---

## 🚀 Usage

### 1. Drift Simulation

Run baseline SONOS drift simulations:

```bash
python radiation_core.py
```

Key functions:

* `conductance()` → Recreates SONOS drift plots
* `dot_products()` → Compares NumPy vs SONOS dot products
* `accuracy_comparison()` → Tracks accuracy drop with TID

---

### 2. Alpha Correction Factor (ACF)

Apply **alpha scaling** to correct drift:

```bash
python radiation_core_alpha.py
```

Functions:

* `dot_product_comparison()` → Pre/post correction results
* `column_count()` → Effect of column sampling on alpha
* `local_alpha()` → Column-wise alpha correction (≈1% better)
* `percent_mean_error()` → Shows mean vs. std deviation

---

### 3. Full-Network Evaluation

Use plots stored in `data` to generate graphs

```bash
python plots.py
```

* `roll_off()` → Summarizes roll-off characteristics

---

## 📊 Results
<img width="910" height="374" alt="image" src="https://github.com/user-attachments/assets/07be1e94-2b40-4ba1-b629-83eb41548243" />

* **Alpha Correction** restores dot product accuracy close to baseline.

---

## 📌 Notes

* Training files (`train_tiny_imagenet.py`, `data_loader_tiny_imagenet.py`) live ![here](../tiny_imagenet) with the models.
* The `extra_files/` folder contains exploratory work (not essential).

---
