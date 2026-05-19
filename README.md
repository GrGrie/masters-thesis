# Master's Thesis: Self-Supervised Representation Geometry & Spurious-Correlation Robustness

This repository contains the codebase for the Master's Thesis: **"Linking Self-Supervised Representation Geometry to Spurious-Correlation Robustness in Vision Transformers"** by Grigory Grechkin.

## Project Overview

The goal of this project is to systematically evaluate how different self-supervised learning (SSL) pretraining objectives (Contrastive Learning (CL), Masked Image Modeling (MIM), and Hybrid CL+MIM) affect a Vision Transformer's robustness to spurious correlations.

Instead of proposing a new robust training algorithm, this project focuses on **empirical and mechanistic analysis** across layers to understand how semantic vs. spurious information is geometrically organized.

## Architecture

The project architecture is organized in a modular way to allow easy experimentation with different datasets, models, training regimes, and analysis metrics.

### Directory Structure

- `data/`: Raw and processed datasets (e.g., Waterbirds, CelebA). Data is not tracked by Git.
- `src/datasets/`: PyTorch Dataset classes and data transforms.
- `src/models/`: ViT architectures and wrappers for different SSL regimes (CL, MIM, Hybrid).
- `configs/`: YAML configuration files containing hyperparameters and paths (e.g., `train.yaml`, `analyze.yaml`).
- `src/training/`: Training and fine-tuning scripts, including custom loss functions.
- `src/evaluation/`: Scripts for evaluating average and worst-group accuracy, as well as background consistency.
- `src/analysis/`: Core code for representation analysis:
  - `attention.py`: Attention distance, spatial aggregation, and homogeneity.
  - `spectral.py`: Singular value spectra, effective rank, and UMAP/PCA visualization.
  - `probing.py`: Linear probing for object vs. background features.
- `src/utils/`: Generic helpers like configuration loading, wandb setup, and seed enforcement.
- `scripts/`: Bash scripts to execute common workflows (e.g., training sweeps).
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and plotting final results.

## Setup

1. Create a Conda environment and install dependencies:
   ```bash
   conda create -n thesis-env python=3.10
   conda activate thesis-env
   pip install -r requirements.txt
   ```

2. Download and prepare the datasets according to the instructions in `src/datasets/README.md` (to be created).

## Workflow / How to Run

1. **Configure**: Set up your experiment parameters in `configs/train.yaml` or `configs/analyze.yaml`.
2. **Train/Fine-tune**: Run the main entry point to fine-tune a model:
   ```bash
   python main.py --config configs/train.yaml --mode train
   ```
3. **Analyze**: Run the representation analysis pipeline:
   ```bash
   python main.py --config configs/analyze.yaml --mode analyze
   ```
4. **Evaluate**: Run downstream evaluation metrics:
   ```bash
   python main.py --config configs/train.yaml --mode evaluate
   ```

## Why this Architecture?

- **Modularity**: By separating models, datasets, training, and analysis, you can easily plug in a new dataset (e.g., CelebA) or a new analysis metric (e.g., a new attention metric) without breaking the rest of the pipeline.
- **Config-Driven**: `main.py` uses YAML configurations to track all hyper-parameters, making reproducibility much easier for your thesis.
- **Separation of Concerns**: The analysis is purely functional and operates on frozen features or model states. It does not mix with the training loops. This allows you to run heavy analyses (like UMAP or probing) offline on saved checkpoints.
