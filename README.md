# Supplementary Materials: Granger Causality with Diffusion-based Likelihood Estimation

This directory contains the supplementary experiments for event-based causal discovery using Granger causality testing combined with diffusion-based likelihood estimation.

## Overview

This bundle implements:
1. **Transformer-based Hawkes Process (THP)**: A neural temporal point process model
2. **Diffusion-based Likelihood Estimator**: Uses diffusion models to estimate event likelihoods
3. **Granger Causality Testing**: Statistical testing with controlled perturbations to infer causal structure
4. **Structural Causal Discovery**: End-to-end pipeline for discovering causal relationships from event sequences

## Files

### Main Scripts
- `train_thp.py`: Trains THP model with perturbation-based causality analysis
- `train_structural_diffusion.py`: Trains diffusion-based likelihood estimator with Granger causality testing
- `thp.py`: Basic THP model implementation (educational/demo version)

### Data Files
- `synthetic_event_sequence_hawkes_{5,10,20,30}.csv`: Synthetic Hawkes event sequences with different sizes (D=5,10,20,30)
- `synth_{5,10,20,30}_answer.csv`: Ground truth adjacency matrices
- `semiconductor_event.csv`: Real-world semiconductor dataset

### Automation
- `run_structural.sh`: Batch experiment runner for multiple sizes and seeds

## Quick Start

### 1. Install Dependencies
```bash
pip install torch pandas numpy scipy scikit-learn matplotlib seaborn
```

### 2. Run Single Experiment

#### THP
```bash
python3 train_thp.py
```

#### Ours
```bash
python3 train_structural_diffusion.py
```

### 3. Run Batch Experiments
```bash
bash run_structural.sh [DATA_DIR] [OUTPUT_DIR]
```

Example:
```bash
bash run_structural.sh
```

## Method Details

### Diffusion-based Likelihood Estimation
The model learns \( P(\text{next\_event} | \text{history}) \) using:
1. **History Encoding**: Multi-head self-attention over event sequences
2. **Diffusion Process**: Forward diffusion adds noise, denoising network predicts noise
3. **Likelihood Estimation**: Reconstruction error serves as likelihood proxy

### Granger Causality Testing
For each potential edge \( X \rightarrow Y \):
1. Compute likelihood of Y WITH X events in history
2. Compute likelihood of Y WITHOUT X events (masked)
3. Statistical test: If likelihood significantly improves with X, edge exists


## Output Format

### Key Parameters
- `--nodes`: Number of event types (5, 10, 20, 30)
- `--seed`: Random seed for reproducibility
- `--epoch`: Training epochs (default: 150 for diffusion, 10 for THP)
- `--significance`: Significance level for statistical tests (default: 0.10)
- `--threshold`: Manual threshold for binarization (THP only, default: 0.005)

## Notes

1. **Data Format**: Event sequences should have columns: `time_stamp`, `event_type`, `seq_id` (optional)
2. **Ground Truth**: Answer files are N×N adjacency matrices (CSV without headers)
3. **GPU Usage**: Automatically uses GPU if available
4. **Computational Cost**: Granger testing is expensive; each size/seed takes 5-30 minutes depending on hardware
5. **Reproducibility**: Set seeds for deterministic results

