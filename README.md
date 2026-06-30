# Remaining Useful Life Prediction Comparison Using Different Time Series Models

**Author:** Arjun Nain

## Overview

This repository contains a comparative study of deep learning models for **Remaining Useful Life (RUL) prediction** using the NASA C-MAPSS turbofan engine degradation dataset. The objective of this project is to investigate how different sequence-based neural network architectures perform on predictive maintenance tasks.

The repository is structured as an ongoing benchmark, where multiple time-series models will be implemented, trained under similar conditions, and evaluated using common regression metrics.

The first completed implementation is a Long Short-Term Memory (LSTM) network.

---

## Objectives

- Study the application of deep learning to predictive maintenance.
- Compare the performance of different time-series architectures for Remaining Useful Life estimation.
- Build reproducible implementations using PyTorch.
- Analyze the strengths and limitations of each architecture on the NASA C-MAPSS dataset.

---

## Dataset

This project currently uses the **NASA C-MAPSS FD001** dataset.

The FD001 subset contains:

- One operating condition
- One fault mode
- Multiple simulated turbofan engines
- Multivariate sensor measurements collected throughout each engine's degradation cycle

The target variable is the **Remaining Useful Life (RUL)** of each engine at every operating cycle.

---

## Current Implementation

### Model

Long Short-Term Memory (LSTM)

Architecture:

- Input Features: 17
- Sequence Length: 30
- LSTM Layers: 2
- Hidden Size: 64
- Dropout: 0.3
- Fully Connected Layers:
  - 64 → 16
  - ReLU Activation
  - 16 → 1 (RUL prediction)

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of non-informative features
- Feature selection to retain relevant sensor measurements
- Standardization using **StandardScaler**
- Sliding window sequence generation
- Sequence length of **30** time steps
- RUL clipping at **125 cycles**

---

## Training Configuration

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Huber Loss |
| Framework | PyTorch |

---

## Evaluation Metrics

The model is evaluated using standard regression metrics.

| Metric | Value |
|---------|-------:|
| MSE | 340.2420 |
| RMSE | 18.4457 |
| MAE | 12.7824 |
| R² Score | 0.6150 |

---

## Repository Structure

```
├── notebooks/
│   └── LSTM_RUL.ipynb
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── README.md
└── requirements.txt
```

---

## Future Work

This repository will be expanded to include additional time-series architectures for direct comparison, including:

- Vanilla RNN
- GRU
- Bidirectional LSTM
- Transformer-based models
- Temporal Convolutional Networks (TCN)

Each model will be trained using a consistent preprocessing pipeline and evaluated using identical metrics to provide a fair comparison.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## References

1. Saxena, A., & Goebel, K. *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Center of Excellence.

2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
