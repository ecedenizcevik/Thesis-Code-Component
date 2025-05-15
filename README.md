# Thesis Project: Facial Emotion Recognition Experiments
Facial Emotion Recognition on Japanese Emotion "Ahegao"

This repository contains the code and materials for my thesis project on facial emotion recognition (FER) using both deep learning and traditional machine learning methods. It includes dataset preparation scripts, a support vector machine (SVM) baseline on ResNet50 feature embeddings, a custom convolutional neural network (CNN), and a VGG16-based transfer learning model.

The raw dataset used for this thesis can be found on this website: https://data.mendeley.com/datasets/5ck5zz6f2c/2
---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Prerequisites](#prerequisites)
4. [Repository Structure](#repository-structure)
5. [Data Preparation](#data-preparation)
6. [Notebook Descriptions](#notebook-descriptions)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Contact](#contact)

---

## Project Overview

This thesis project compares three approaches to FER:

* **Baseline SVM**: A support vector machine using ResNet50 feature embeddings.
* **Custom CNN**: A convolutional neural network trained from scratch.
* **VGG-based Model**: A fine-tuned VGG16 architecture for transfer learning.

Each model is evaluated on dataset_split file to analyze differences in performance.

---

## Environment Setup

### Local (Jupyter)

Use Jupyter Lab environment for the SVM and CNN notebooks:

```bash
pip install -r requirements.txt
pip install jupyterlab
jupyter lab
```

### Google Colab

The VGG-based notebook requires more memory and is best run in Google Colab:

1. Open `notebooks/VGG16-Transfer-Learning.py` in Colab 
2. Enable GPU under Runtime → Change runtime type → T4 GPU.
3. Mount the repository or upload data as needed.

---

## Prerequisites

* Python 3.7+
* Jupyter Notebook or JupyterLab
* Google Colab account (for VGG16 notebook)
* Libraries (install via `pip install -r requirements.txt`):

  * `numpy`
  * `scikit-learn`
  * `tensorflow` / `keras`
  * `matplotlib`
  * `seaborn`
  * `os`

---

## Repository Structure

```plaintext
├── notebooks/                       # Python scripts
│   ├── Dataset_split.py             # Script to split dataset into train/test
│   ├── SVM-baseline.py              # SVM baseline implementation
│   ├── custom-CNN.py                # Custom CNN training and evaluation
│   ├── VGG16-Transfer-Learning.py   # VGG transfer learning (Colab recommended)
│   └── OAHEGA_class_distribution.py # Dataset distribution figure
├── figures/                         # Evaluation metrics, plots, and figures
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## Data Preparation

After the raw dataset is downloaded, `Dataset_split.py` does:

1. Reading raw images from the OAHEGA dataset, organized by emotion class.
2. Randomly splitting each class into 80% training and 20% test sets.
3. Creating `dataset_split/train/<class>/` and `dataset_split/test/<class>/` directories.
4. Copying images accordingly for compatibility with Keras' `ImageDataGenerator`.

*No separate validation set is created in this script but handled in the seperate model notebooks.*

---

## Notebook Descriptions

### 1. `Dataset_split.py`

* Splits the raw dataset into train (80%) and test (20%) subfolders per class.

### 2. `SVM-baseline.py`

* Uses ResNet50 to extract feature embeddings.
* Trains and evaluates an SVM classifier on the split data.
* Reports test macro-F1 score, precision/recall, confusion matrix, and learning curve.

### 3. `custom-CNN.py`

* Defines and trains a custom CNN architecture on the training split.
* Evaluates performance on the test split, reporting macro-F1 score, precision/recall, confusion matrix, loss and accuracy.

### 4. `VGG16-Transfer-Learning.py`

* Fine-tunes a pretrained VGG16/VGG19 model on the training split.
* Evaluates on the test split; designed to run in Google Colab with GPU.
* Repors macro-F1 score, precision/recall, confusion matrix, loss and accuracy.


### 5. `OAHEGA_class_distribution.py`

* Visualizes dataset distribution by each class.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Prepare the data split**

   ```bash
   jupyter notebook notebooks/Dataset_split.py
   ```

   * Run all cells to generate `dataset_split/train/` and `dataset_split/test/`.

3. **Run SVM and CNN locally**

   ```bash
   ```bash
   pip install jupyterlab
   jupyter lab
   ```

   * Execute:

     1. `SVM-baseline.py`
     2. `custom-CNN.py`

4. **Run VGG model in Colab**

   * Open and run `notebooks/VGG16-Transfer-Learning.py` in Google Colab with GPU.

---

## Results

All evaluations use the test split. Preliminary results saved in `figures/`:

* **SVM baseline**: \~0.64 macro-F1 score
* **Custom CNN**: \~0.76 macro-F1 score
* **VGG-based model**: \~0.77 macro-F1 score

Detailed figures and tables are included for thesis write-up.

---

## Contact

For questions or feedback, please contact:

**Ece Deniz Çevik**

> [e.d.cevik@tilburguniversity.edu](mailto:e.d.cevik@tilburguniversity.edu)

---

**Last updated:** 15-05-2025
