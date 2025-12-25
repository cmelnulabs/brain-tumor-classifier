# 🧠 Brain Tumor Classifier

A deep learning project for classifying brain tumor images using PyTorch. This project includes data preprocessing, model training, evaluation, and visualization.

---

## 📌 Description

This project aims to classify brain tumors from medical images using a convolutional neural network (CNN). It includes a full pipeline from data loading and preprocessing to training and evaluation.

---

## 💻 System Compatibility

This project is compatible with:
* **Linux** (Ubuntu 20.04+)

* **Windows 10+**

Thanks to Conda, the environment is fully reproducible and automatically adapted to each system. The same setup instructions work on both platforms.

---

## ⚙️ Requirements

Before proceeding, ensure you have **Conda** installed on your system. You can install it via:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://www.anaconda.com/)

> ⚠️ **Note:** This project uses a Conda environment for dependency management.

---

## 🚀 Installation Guide: Brain Tumor Classifier


Let’s get your Brain Tumor Classifier up and running on **Windows or Linux**! Follow these steps:


### 1. 🌐 Create a New Environment (Recommended)

It's always a good idea to work in a clean environment to avoid conflicts.

```bash
conda create -n brain-tumor-classifier python=3.10 -y
conda activate brain-tumor-classifier
```

### 2. 🔬 Install Core Scientific Packages

We'll install essential libraries with compatible versions for smooth operation.

```bash
conda install -c pytorch -c conda-forge pytorch torchvision torchaudio cpuonly
matplotlib=3.8.4 pillow=9.5.0 -y
```

### 3. 🧑‍💻 Install KaggleHub

```bash
pip install kagglehub
```

## Usage

After activating the environment:

```bash
python -m main
```

What happens:
1. The dataset is downloaded automatically via KaggleHub (cached after first run).
2. Data is loaded from the extracted `Training/` and `Testing/` folders.
3. The model trains with class weighting, learning rate scheduling, and early stopping.
4. The best model (by test accuracy) is saved to `best_brain_tumor_model.pt`.

To rerun without re‑downloading, just execute the same command; KaggleHub uses a local cache.

## Dataset

Source: `masoudnickparvar/brain-tumor-mri-dataset` (Kaggle). Classes are auto-detected from directory names. Grayscale conversion + resize to 224×224 + normalization (mean=0.5, std=0.5).

## Model Architecture

Compact CNN stack (all with ReLU + BatchNorm):
`1→8→16→32→64→128→256→512` feature maps with intermittent MaxPool (stride 2). Final: Flatten → Linear (512→num_classes). One dropout layer (p=0.2) after the 64‑channel block.

## Training

Features:
* Optimizer: Adam (lr = 5e-4)
* Loss: CrossEntropy with inverse-frequency class weights
* Scheduler: ReduceLROnPlateau (factor 0.5, patience 2)
* Early stopping: patience = 3 epochs (monitors training loss)
* Progress bars: per-epoch, batch-level (tqdm)

Adjust hyperparameters in `btclassifier/config.py`.

## Evaluation

Per epoch: test loss & accuracy. After training: reload best checkpoint and report final metrics. Modify/extend in `btclassifier/evaluate.py`.

## Results

Add your benchmark table here (e.g., accuracy, per-class precision/recall). Example placeholder:

| Run | Test Acc | Notes |
|-----|----------|-------|
| 1   | 0.XX     | baseline |

## Contributing

Open an issue or submit a PR with clear description. Keep code modular (see `btclassifier/` package layout).

## License

See `LICENSE` file.

## Acknowledgments

Dataset authors and the PyTorch community.
