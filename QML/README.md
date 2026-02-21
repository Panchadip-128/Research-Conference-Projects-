# NoiseAware-QML: Noise-Aware Variational Quantum Circuits for Anomaly Detection

Official implementation and research code for **NoiseAware-QML**, a Noise-Aware Variational Quantum Circuit (NA-VQC) framework designed for parameter-efficient, robust anomaly detection on tabular data.

This repository contains:

* Full training pipeline
* Noise-aware quantum model implementation
* Experimental evaluation and ablation studies
* Reproducibility scripts
* Research paper and results

---

# Overview

NoiseAware-QML introduces a novel approach to quantum machine learning by integrating a **trainable noise parameter directly into the variational quantum circuit (VQC)**. Instead of treating noise as a fixed hardware constraint, this framework learns an optimal noise level during training.

Key objective:

> Improve robustness, calibration, and parameter efficiency of quantum models for anomaly detection.

Primary dataset:

* UNSW-NB15 Network Intrusion Dataset

---

# Key Contributions

## 1. Trainable Noise Parameter

A learnable noise parameter ( \sigma ) is embedded in the quantum circuit:

* Optimized jointly with gate parameters
* Enables adaptive noise calibration
* Improves robustness and generalization

---

## 2. Noise-Aware Loss Function

Custom loss:

L = BCE + λσ² + Ranking Loss

Benefits:

* Prevents noise overfitting
* Encourages strong anomaly separation
* Improves calibration and stability

---

## 3. Parameter-Efficient Quantum Architecture

Model uses:

* 4 qubits
* 3 variational layers
* Only **~54 trainable parameters**

Compared to classical baselines with thousands to millions of parameters.

---

## 4. Built-in Quantum Explainability

Provides interpretability via:

* Gate gradient importance
* Parameter sensitivity analysis
* Bloch sphere state evolution

---

# Architecture

Pipeline:

Classical Input → Feature Encoding → Variational Quantum Circuit → Measurement → Classical Output

Components:

* Angle embedding
* Parameterized rotation gates
* Entanglement layers
* Noise injection layer (trainable)
* Expectation measurement

---

# Results Summary

| Model            | AUC    | Parameters |
| ---------------- | ------ | ---------- |
| NoiseAware-QML   | 0.8366 | 54         |
| Random Forest    | 0.9369 | ~100,000+  |
| Isolation Forest | 0.8084 | ~10,000+   |

Key strengths:

* Extreme parameter efficiency
* Strong anomaly separation
* Stable performance under noise

---

# Repository Structure

```
QML/
 ├ Dataset/
 │  └ README.md
 ├ Images/
 │  └ (Outputs of the notebook)
 └ Model/
    └ qml-kaggle-v3.ipynb
```

---

# Usage

Run training notebook:

```
jupyter notebook qml-kaggle-v3.ipynb
```

---

# Reproducibility

Includes:

* Multi-seed training
* Bootstrap confidence intervals
* Ablation studies
* Statistical significance tests

Reproducible using provided notebook.

---

# Experimental Features

Supports:

* Trainable noise optimization
* Adaptive regularization
* Gradient-based explainability
* Noise robustness evaluation


---

# Applications

* Network intrusion detection
* Fraud detection
* Cybersecurity anomaly detection
* Financial anomaly detection
* Industrial fault detection

---

# Limitations

* Uses quantum simulator (not real hardware)
* Smaller datasets due to simulation constraints
* Quantum hardware deployment not yet tested

---

# Future Work

* Deployment on real quantum hardware
* Scaling to larger datasets
* Hybrid classical-quantum architectures
* Real-time anomaly detection
