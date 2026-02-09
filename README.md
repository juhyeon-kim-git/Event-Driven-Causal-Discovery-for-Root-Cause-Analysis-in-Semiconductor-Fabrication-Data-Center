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
- `threshold.py`: Utility script for thresholding and binarizing adjacency matrices

### Data Files
- `synthetic_event_sequence_hawkes_{5,10,20,30}.csv`: Synthetic Hawkes event sequences with different sizes (D=5,10,20,30)
- `synth_{5,10,20,30}_answer.csv`: Ground truth adjacency matrices

### Automation
- `run_structural.sh`: Batch experiment runner for multiple sizes and seeds

### Outputs
- `figures/`: Visualization results (heatmaps, bar charts)

## Quick Start

### 1. Install Dependencies
```bash
pip install torch pandas numpy scipy scikit-learn matplotlib seaborn
```

### 2. Run Single Experiment

#### Option A: THP with Perturbation
```bash
python3 train_thp.py --nodes 5 --seed 0 --epoch 10 --threshold 0.005
```

#### Option B: Diffusion + Granger Causality (Recommended)
```bash
python3 train_structural_diffusion.py --nodes 5 --seed 0 --epoch 150 --significance 0.10
```

### 3. Run Batch Experiments
```bash
bash run_structural.sh [DATA_DIR] [OUTPUT_DIR]
```

Example:
```bash
bash run_structural.sh /home/s2/juhyeonkim/samsung4/data ./results_thp_g3
```

This will:
- Run experiments on all sizes (5, 10, 20, 30)
- Execute 20 runs per size with different seeds (0-19)
- Save results to `OUTPUT_DIR/size{N}_results.csv`
- Generate summary statistics in `OUTPUT_DIR/size{N}_summary.txt`
- Create overall summary in `OUTPUT_DIR/overall_summary.csv`

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

**Criteria** (relaxed for better recall):
- Strong evidence: \( p < 0.05 \) AND Cohen's \( d > 0.3 \), OR Cohen's \( d > 0.8 \)
- Moderate evidence: \( p < 0.1 \) AND Cohen's \( d > 0.5 \) AND score \( > \) adaptive threshold
- Weak evidence: \( p < 0.05 \) AND score \( > 1.5 \times \) adaptive threshold

### Ensemble Testing
The method runs multiple tests with different random seeds and keeps edges that appear in majority of runs (≥2 out of 3).

## Output Format

### Individual Results CSV
```csv
seed,precision,recall,f1,shd,nhd
0,0.8542,0.7123,0.7778,3.0000,0.0500
1,0.9012,0.7450,0.8156,2.5000,0.0417
...
```

### Summary Statistics
```
Size | Runs | Precision        | Recall           | F1 Score         | SHD              | NHD
-----|------|------------------|------------------|------------------|------------------|------------------
   5 |   20 | 0.8123 ± 0.0543 | 0.7234 ± 0.0612 | 0.7645 ± 0.0523 | 2.8500 ± 0.5124 | 0.0475 ± 0.0085
  10 |   20 | 0.7845 ± 0.0678 | 0.6912 ± 0.0734 | 0.7342 ± 0.0612 | 8.2300 ± 1.2345 | 0.0823 ± 0.0123
...
```

## Hyperparameters

### Key Parameters
- `--nodes`: Number of event types (5, 10, 20, 30)
- `--seed`: Random seed for reproducibility
- `--epoch`: Training epochs (default: 150 for diffusion, 10 for THP)
- `--significance`: Significance level for statistical tests (default: 0.10)
- `--threshold`: Manual threshold for binarization (THP only, default: 0.005)

### Model Architecture
- Hidden dimension: 128
- Attention heads: 4
- Diffusion steps: 50
- Learning rate: 0.001 with warmup
- Batch size: 32
- Sequence length: 100

## Utility: Threshold Tool

The `threshold.py` script can binarize any adjacency matrix:

```bash
python3 threshold.py \
    --input learned_adj_matrix.csv \
    --output binary_adj_matrix.csv \
    --threshold 0.1 \
    --abs \
    --no-self-loop
```

Options:
- `--threshold`: Threshold value (values ≥ threshold become 1, else 0)
- `--abs`: Apply threshold on absolute values
- `--no-self-loop`: Force diagonal entries to 0
- `--header`: Input CSV has header row
- `--eps`: Numerical epsilon for comparison

## Notes

1. **Data Format**: Event sequences should have columns: `time_stamp`, `event_type`, `seq_id` (optional)
2. **Ground Truth**: Answer files are N×N adjacency matrices (CSV without headers)
3. **GPU Usage**: Automatically uses GPU if available
4. **Computational Cost**: Granger testing is expensive; each size/seed takes 5-30 minutes depending on hardware
5. **Reproducibility**: Set seeds for deterministic results

## Troubleshooting

### NaN/Inf in Loss
The code automatically handles NaN/Inf values through:
- Data clipping and normalization
- Gradient clipping (max norm = 1.0)
- Skip batches with invalid loss

### Memory Issues
- Reduce batch size in the code
- Use shorter sequence lengths
- Process fewer likelihood samples

### Low Recall
- Increase `--significance` (e.g., 0.15 or 0.20)
- Adjust min_support threshold (default: 15)
- Increase ensemble size

## Citation

If you use this code, please cite:
- Original THP paper: [Transformer Hawkes Process (2020)]
- Granger causality framework: [Time Series Causality Testing]
- Diffusion models: [Denoising Diffusion Probabilistic Models]

## Contact

For questions or issues, please contact the authors or open an issue in the repository.

