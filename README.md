# ECG Classification with a Custom Neural Network

This project builds and validates a custom neural network for ECG binary classification using NumPy. The implementation includes dense layers, ReLU activations, dropout, Adam optimization, and binary cross-entropy loss, with numerical gradient checking used to verify the backpropagation logic.

The project also includes an ECG data pipeline for preparing MIT-BIH signal data and a training/evaluation workflow for model experimentation.

## Overview

The goal of this project is to:
- implement a neural network from scratch without a deep learning framework
- validate gradient calculations numerically
- process ECG signals into training-ready features
- train and evaluate a binary classifier for abnormal rhythm detection
- test edge cases such as dropout, confident predictions, and near-zero gradients

## Features

- Custom NumPy-based neural network implementation
- Dense, ReLU, and Dropout layers
- Adam optimizer
- Binary cross-entropy with logits
- Gradient checking for numerical verification
- ECG preprocessing and feature extraction pipeline
- Patient-safe train/validation/test splitting
- Training scripts and regression tests

## Project Structure

- `nn.py` — neural network implementation, optimizer, loss, and gradient-check utilities
- `ecg_data.py` — ECG data loading, preprocessing, feature extraction, and patient grouping
- `ecg_pipeline.py` — signal-processing and inference utilities
- `train_ecg.py` — ECG training and model evaluation workflow
- `test_nn.py` — unit tests for gradient correctness and regression safety
- `test_pipeline.py` — pipeline and model validation checks

## Setup

This project is designed for Python 3.10+.

Install dependencies:

```bash
pip install numpy
