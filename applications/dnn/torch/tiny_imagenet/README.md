# Tiny ImageNet Training with CrossSim & Alpha-in-the-Loop

## 📌 Project Overview

This repository explores methods for improving the robustness of deep learning models under **radiation-induced noise in in-situ analog computing hardware (SONOS)**.

When SONOS-based crossbar arrays are exposed to radiation, **total ionizing dose (TID)** effects cause conductance drift, which leads to errors in dot products and reduced model accuracy. To mitigate this, we experiment with three strategies:

1. **Alpha Correction Factor (ACF)** – Apply a scalar correction to the dot product output to counteract drift.

   * Requires periodic calibration.
   * Reduces mean error across radiation levels.

2. **Noise-Aware Training** – Train neural networks with simulated TID noise injected into the forward/backward pass.

   * Models adapt to noise distributions.
   * Variants include alternating TID levels or Gaussian-distributed TID.

3. **Alpha-in-the-Loop Training** – Combine both methods by training with noise **and** alpha correction during training.

   * Achieves the strongest robustness, maintaining accuracy even at high TID levels.

This folder focuses on **Noise-Aware Training** and **Alpha-in-the-Loop Training**.
See [`../tiny_imagenet_radiation`](../tiny_imagenet_radiation) for the implementation of single-layer alpha correction experiments.

Experiments are performed on **Tiny ImageNet** using a **ResNet-32 backbone** (with an additional 256×256 linear layer for SONOS hardware programming). Results show that combining alpha correction and noise-aware training improves resilience by an **order of magnitude** in tolerable TID before accuracy degradation.

---

## 📂 Repository Structure

```
├── build_tiny_imagenet.py       # Builds ResNet32 for Tiny ImageNet
├── data_loader_tiny_imagenet.py # Loads Tiny ImageNet dataset
├── train_tiny_imagenet.py       # Baseline training (no CrossSim)
├── train_TIN_noise.py           # Training with CrossSim noise injection
├── train_TIN_alpha.py           # Training with noise + alpha-in-the-loop
├── inference_tiny_imagenet.py   # Standard inference
├── full_network_alpha.py        # Inference with alpha correction (or baseline if use_alpha=False)
├── model_weights.py             # Extracts penultimate linear layer (for SONOS board deployment)
├── TID_data_0802_CrossSim.p     # SONOS chip experimental dataset (preferred)
├── TID_params_202312.p          # Alternate SONOS dataset
└── TIN_dataset/                 # Dataset preprocessing utilities
```

---

## ⚙️ Setup

### Installation

1. Clone the repository and install dependencies listed here: [CrossSim (pytorch branch)](https://github.com/sandialabs/cross-sim/tree/pytorch)
2. Download the **Tiny ImageNet dataset** into `applications/dnn/data`.
3. Preprocess the dataset:

   ```bash
   python TIN_dataset/TIN_DataSet_Parser.py
   ```

---

## 🚀 Usage

### Training

* **Baseline (no noise):**

  ```bash
  python train_tiny_imagenet.py
  ```

* **With CrossSim noise:**

  ```bash
  python train_TIN_noise.py
  ```

* **With alpha-in-the-loop:**

  ```bash
  python train_TIN_alpha.py
  ```

### Inference

* **Standard inference:**

  ```bash
  python inference_tiny_imagenet.py --model_path model.pth
  ```

* **With alpha correction:**

  ```bash
  python full_network_alpha.py --model_path model.pth --use_alpha True
  ```

### Extract Model Weights for SONOS Deployment

```bash
python model_weights.py --model_path alpha_loop_models/model.pth --output weights.pth
```

---

## 📊 Results

<img width="469" height="343" alt="image" src="https://github.com/user-attachments/assets/ec71e3fd-2cc3-4ab9-bc4f-b60396022ca9" />

* **Base model** quickly loses accuracy with increasing TID.
* **Noise training** improves tolerance to drift.
* **Alpha correction** significantly extends robustness.
* **Alpha-in-the-loop training** provides the best tradeoff, maintaining accuracy over a wider TID range.

| Training Type         | Inference Trials | 0 TID Accuracy | Peak Accuracy | Loses 5% Accuracy (TID) | Reaches 49.335% Accuracy (TID) |
| --------------------- | ---------------- | -------------- | ------------- | ----------------------- | ------------------------------ |
| Base                  | 10               | 54.335         | 54.335 (0k)   | 4.174k                  | 4.174k                         |
| 4k (Alternating)      | 3                | 54.64          | 54.64 (0k)    | 9.684k                  | 9.944k                         |
| 4e3 Alpha Noise       | 3                | 53.68          | 53.7 (3k)     | 33.95k                  | 31.23k                         |
| 10k Alpha in the Loop | 10               | 55.2           | 55.2 (4k)     | 57.3k                   | 70.3k                          |

📈 Additional plots and methodology details are available [`here`](supporting_docs/Reducing_Image_Recognition_Accuracy_Loss_from_Radiation_in_Analog_Memristive_Chips.pdf).

---

## 👥 Contributors

* [Benjamin Tung](https://github.com/bentung12)
* Patrick Xiao & Matthew Marinella (mentorship)
* Maximillian Siath (SONOS TID data collection)
* Justin Weidmann, Zachary White, Wataru Tamaki, Hoyeol Bae (experimental testing)
