# Tiny ImageNet Training with CrossSim and Alpha-in-the-Loop

This repository explores methods for improving the robustness of deep learning models under **radiation-induced noise in in-situ analog computing hardware (SONOS)**.

When SONOS-based crossbar arrays are exposed to radiation, **total ionizing dose (TID)** effects cause conductance drift, which leads to errors in dot products and reduced model accuracy. To address this, we experiment with three approaches:

1. **Alpha Correction Factor (ACF)** – Apply a scalar correction to the dot product output to counteract drift.

   * Requires periodic calibration.
   * Reduces mean error across radiation levels.

2. **Noise-Aware Training** – Train neural networks with simulated TID noise injected into the forward/backward pass.

   * Models learn to adapt to noise distributions.
   * Variants include alternating TID levels or Gaussian-distributed TID.

3. **Alpha-in-the-Loop Training** – Combine both methods by training with noise **and** alpha correction applied during training.

   * Achieves the strongest robustness, maintaining accuracy even at high TID levels.

This specific folder contains the code for **Noise-Aware Training** and **Alpha-in-the-Loop Training**. Refer to ![here](../tiny_imagenet_radiation) for the implementation of the alpha correction factor upon a single layer.

Experiments are performed on **Tiny ImageNet** using a **ResNet-32 backbone** (with an additional 256×256 linear layer intended for SONOS hardware programming). Results show that combining alpha correction and noise-aware training improves resilience by an **order of magnitude** in terms of tolerable TID before accuracy degradation.

---

## 📂 Repository Structure

* **`build_tiny_imagenet.py`** – Builds a ResNet32 architecture tailored for Tiny ImageNet.
* **`data_loader_tiny_imagenet.py`** – Prepares and loads data from the `TIN_dataset` folder.
* **`train_tiny_imagenet.py`** – Baseline training script (no CrossSim).
* **`train_TIN_noise.py`** – Training with CrossSim noise injection.
* **`train_TIN_alpha.py`** – Training with both CrossSim noise and alpha-in-the-loop correction (can also run in noise-only mode with `use_alpha=False`).
* **`inference_tiny_imagenet.py`** – Standard inference on trained models.
* **`full_network_alpha.py`** – Inference with alpha correction (can also replicate baseline inference if `use_alpha=False`).
* **`model_weights.py`** – Extracts and saves the penultimate linear layer of a model (layer used on physical SONOS board).
* **`TID_data_0802_CrossSim.p` / `TID_params_202312.p`** – SONOS experiment data used in simulations.
* 
### Folders

* **`TIN_dataset/`** – Python code that processes Tiny Imagenet data.

---

## ⚙️ Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/tinyimagenet-crosssim.git
   cd tinyimagenet-crosssim
   ```

2. Install dependencies (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

   *(If you don’t have a `requirements.txt` yet, you can generate one with `pip freeze > requirements.txt`.)*

3. Prepare the Tiny ImageNet dataset and place it inside the `TIN_dataset/` directory.

---

## 🚀 Usage

### Training

* Baseline:

  ```bash
  python train_tiny_imagenet.py
  ```
* With CrossSim noise:

  ```bash
  python train_TIN_noise.py
  ```
* With alpha-in-the-loop:

  ```bash
  python train_TIN_alpha.py
  ```

### Inference

* Standard:

  ```bash
  python inference_tiny_imagenet.py --model_path base_models/model.pth
  ```
* With alpha correction:

  ```bash
  python full_network_alpha.py --model_path alpha_loop_models/model.pth --use_alpha True
  ```

### Extract Model Weights

```bash
python model_weights.py --model_path alpha_loop_models/model.pth --output weights.pth
```

---

## 📊 Results

Trained models are stored in:

* `base_models/`
* `noise_models/`
* `alpha_loop_models/`

Comparison of accuracy, robustness, and error patterns across training modes shows that **alpha-in-the-loop consistently yields the best performance under radiation noise**.

---

## 👥 Contributors

* [Your Name](https://github.com/yourusername)
* [Partner’s Name](https://github.com/partnerusername) – contributions on dataset preparation, training scripts, and evaluation.

---

## 📜 License

Choose and add a license (e.g., MIT, Apache 2.0).

---

Do you want me to also **add one or two figures from your PowerPoint (like the drift correction graph or summary TID results)** into the README as embedded images, so it visually shows the problem and solution?
