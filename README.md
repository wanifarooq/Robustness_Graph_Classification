# GNN Training and Analysis Framework for Noisy Graph Data

This repository provides a **comprehensive framework for training, evaluating, and analyzing Graph Neural Network (GNN) models**, with a special focus on **robustness under noisy labels and perturbations**.  
It includes multiple baseline and advanced models, custom loss functions, augmentation strategies, and detailed analysis tools such as **Dirichlet energy tracking, eigenvalue analysis, and embedding visualization**.  

The codebase integrates **Comet.ml for experiment tracking** and supports a wide range of datasets (vision datasets, TU datasets, OGB).

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Repository Structure](#repository-structure)  
3. [Getting Started](#getting-started)  
   * [Prerequisites](#prerequisites)  
   * [Installation](#installation)  
4. [Usage](#usage)  
   * [Training a Model](#training-a-model)  
   * [Command-Line Arguments](#command-line-arguments)  
   * [Examples](#examples)  
5. [Core Components](#core-components)  
   * [Supported Models](#supported-models)  
   * [Datasets](#datasets)  
   * [Loss Functions](#loss-functions)  
   * [Utilities](#utilities)  
6. [Experiment Tracking with Comet.ml](#experiment-tracking-with-cometml)  
7. [Dependencies](#dependencies)  

---

## Key Features
- **Multiple GNN Architectures**: Implementations of GCN, GIN, GAT, SGC, and Frequency-aware GNNs (F2GNN).  
- **Robustness to Noisy Labels**: Noise-robust training strategies with custom loss functions (`sopLoss`, `lossdir`, etc.).  
- **Comprehensive Dataset Support**: Includes MNIST, CIFAR-100, TU datasets (ENZYMES, PROTEINS, MUTAG, etc.), and OGB datasets.  
- **Advanced Analysis Tools**:  
  - Dirichlet energy calculation for embedding smoothness  
  - Eigenvalue/spectral analysis of model weights  
  - Graph augmentations and perturbations (edge add/drop, clustering, sparsification)  
  - Wasserstein-based distribution comparisons  
  - Visualization with t-SNE, KDE, and Comet.ml logging  
- **Modular and Extensible**: Clear separation between models, losses, datasets, and utilities for quick extension.  

---

## Repository Structure

### Top-level Training Scripts
- **`main.py`** – Entry point for standard GNN training on TU datasets.  
- **`mainMnist.py`** – Train CNN/GNN models on MNIST Superpixels dataset.  
- **`Ciafr100Main.py`** – Training on CIFAR-100.  
- **`OtherDatasets.py`** – Runner for other benchmarks.  
- **`mainf2gnn.py`** – Training with Frequency-aware GNN (F2GNN).  
- **`main_funtional.py` / `main1.py`** – Experimental variants of training pipelines.  
- **`newmethod.py`** – Prototype of new robustness strategies.  
- **`nodeTest.py`** – Node-level classification testing and sanity checks.  
- **`checkdirchiletmain.py` / `checkdirchiletmainSingleW2.py`** – Scripts for running experiments that specifically log and analyze Dirichlet energy dynamics.  
- **`classcharacteristics.py`** – Extracts and logs per-class embedding/Dirichlet statistics.  
- **`killProcess.sh`** – Utility to terminate runaway training processes.  

---

### `models/` – Core Model Implementations
- **`gnn.py` / `conv.py` / `layers.py` / `models_layer.py`** – Standard GNN layers and backbones (GCN, GAT, GIN, SGC).  
- **`MLP_modules.py`** – Baseline MLP architectures for comparison.  
- **`conv copy.py`** – Legacy/convenience copy of convolution definitions.  
- **`loss/`** – Noise-robust and experimental loss functions:  
  - `loss.py`, `loss1.py`, `loss2.py`, `loss3.py` – Variants of cross-entropy with reweighting/robustness tweaks.  
  - `lossdir.py` – Dirichlet energy–regularized loss.  
  - `sopLoss.py` – Self-organizing penalty loss for noisy labels.  

---

### `VPA/` – Variational Perturbation Analysis
- **`train.py`** – Main VPA training script.  
- **`conf/`** – YAML configs for datasets, models, logging, trainer.  
- **`src/`** – VPA implementations of GAT, GCN, GIN, SGC and supporting utilities.  
- **`environment.yaml`** – Conda environment for reproducibility.  
- **`requirements.txt`** – Python dependencies.  
- **`gnn-vpa.png`** – Visual diagram of the VPA framework.  

---

### `f2gnn/` – Frequency-aware GNN
- **`f2gnnModel.py`** – Core F2GNN model implementation.  
- **`agg_zoo.py`, `pooling_zoo.py`** – Libraries of aggregation and pooling operators.  
- **`message_passing.py`, `pyg_gnn_layer.py`** – Custom PyTorch Geometric layers.  
- **`genotypes.py`, `operations.py`** – Search/architecture definitions.  
- **`op_graph_classification.py`** – Operators specialized for graph classification.  
- **`arch.txt`** – Architecture specification.  

---

### `utils/` – General Utilities
- **`Visulaize.py`** – Tools for plotting metrics, embeddings, Dirichlet energies.  
- **`augmentGraph.py` / `addEdgesandDrop.py`** – Graph augmentation utilities.  
- **`denserCluster.py`** – Graph densification utilities.  
- **`dirchilet.py`** – Functions for computing Dirichlet energy of embeddings.  
- **`wassertein.py`** – Wasserstein distance utilities for distribution comparison.  
- **`sparse_softmax.py`** – Sparse tensor softmax implementation.  
- **`utils.py`** – Miscellaneous helpers (logging, setup, metrics).  

---

### `datasethelpers/`
- **`customdataset.py`** – Loader for custom graph datasets.  
- **`nodeDataset.py`** – Node-level dataset wrappers for PyTorch Geometric.  

---

### Other
- **`.vscode/`** – Editor settings for consistency.  
- **Top-level `README.md`** – Quick reference (this file expands it).  

---

## Getting Started

### Prerequisites
- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- PyTorch Geometric ≥ 2.0  
- (Optional) Comet.ml account for logging  

### Installation
```bash
git clone https://github.com/<your-username>/Robustness_Graph_Classification.git
cd Robustness_Graph_Classification/Robustness_Graph_Classification-main
pip install -r requirements.txt
