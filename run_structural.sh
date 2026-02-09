#!/usr/bin/env bash
#SBATCH --job-name=ours
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=unlimited
#SBATCH --mem=100GB
#SBATCH --partition=viba1
#SBATCH --cpus-per-task=4

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DATA_DIR="${1:-/home/s2/juhyeonkim/samsung4/data}"
OUTPUT_DIR="${2:-./results_thp_g3}"
SIZES=(5 10 20 30)
NUM_RUNS=20
START_SEED=0
EPOCHS=150
SIGNIFICANCE=0.10

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print header
echo ""
print_msg "======================================================================" "$BLUE"
print_msg " Granger Causality with Diffusion Likelihood Experiment Runner" "$BLUE"
print_msg "======================================================================" "$BLUE"
echo ""
echo "Data Directory:    $DATA_DIR"
echo "Output Directory:  $OUTPUT_DIR"
echo "Sizes:             ${SIZES[@]}"
echo "Runs per size:     $NUM_RUNS"
echo "Seed range:        $START_SEED - $((START_SEED + NUM_RUNS - 1))"
echo "Epochs:            $EPOCHS"
echo "Significance:      $SIGNIFICANCE"
echo ""
print_msg "======================================================================" "$BLUE"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_msg "ERROR: Data directory not found: $DATA_DIR" "$RED"
    echo ""
    echo "Usage: $0 [DATA_DIR] [OUTPUT_DIR]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Create overall summary file
OVERALL_SUMMARY="$OUTPUT_DIR/overall_summary.csv"
echo "size,runs,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std,shd_mean,shd_std,nhd_mean,nhd_std" > "$OVERALL_SUMMARY"

print_msg "Starting Granger Causality experiments at $(date)" "$GREEN"
echo ""

# Main experiment loop
for size in "${SIZES[@]}"; do
    print_msg "======================================================================" "$BLUE"
    print_msg " Processing Size = $size" "$BLUE"
    print_msg "======================================================================" "$BLUE"
    echo ""
    
    # Check data files
    event_file="$DATA_DIR/synthetic_event_sequence_hawkes_${size}.csv"
    answer_file="$DATA_DIR/synth_${size}_answer.csv"
    
    if [ ! -f "$event_file" ]; then
        print_msg "ERROR: Event file not found: $event_file" "$RED"
        echo ""
        continue
    fi
    
    if [ ! -f "$answer_file" ]; then
        print_msg "ERROR: Answer file not found: $answer_file" "$RED"
        echo ""
        continue
    fi
    
    print_msg "Found data files:" "$GREEN"
    echo "  - $event_file"
    echo "  - $answer_file"
    echo ""
    
    # Create output file for this size
    results_file="$OUTPUT_DIR/size${size}_results.csv"
    echo "seed,precision,recall,f1,shd,nhd" > "$results_file"
    
    # Run experiments for each seed
    print_msg "Running $NUM_RUNS experiments with Granger Causality + Diffusion..." "$YELLOW"
    echo ""
    
    for i in $(seq 0 $((NUM_RUNS - 1))); do
        seed=$((START_SEED + i))
        run_num=$((i + 1))
        
        echo -n "[$(date +%H:%M:%S)] Run $run_num/$NUM_RUNS (seed=$seed)... "
        
        # Run experiment
        result=$(python3 train_structural_diffusion.py \
            --nodes $size \
            --seed $seed \
            --epoch $EPOCHS \
            --significance $SIGNIFICANCE \
            2>&1 | tail -n 1)
        
        # Check if experiment succeeded
        if [[ $result == *"ERROR"* ]] || [[ $result == *"Traceback"* ]]; then
            print_msg "✗ FAILED" "$RED"
            echo "$result" >> "$OUTPUT_DIR/size${size}_errors.log"
        elif [[ $result =~ ^[0-9]+,[0-9]+,[0-9.]+,[0-9.]+,[0-9.]+,[0-9.]+,[0-9.]+$ ]]; then
            # Parse results: nodes,seed,precision,recall,f1,shd,nhd
            IFS=',' read -r nodes seed_out prec rec f1 shd nhd <<< "$result"
            
            # Save to file
            echo "$seed,$prec,$rec,$f1,$shd,$nhd" >> "$results_file"
            
            print_msg "✓ P=$prec, R=$rec, F1=$f1" "$GREEN"
        else
            print_msg "✗ FAILED: Invalid output" "$RED"
            echo "Output: $result" >> "$OUTPUT_DIR/size${size}_errors.log"
        fi
    done
    
    echo ""
    
    # Calculate statistics
    print_msg "Computing statistics..." "$YELLOW"
    
    python3 << EOF
import pandas as pd
import numpy as np
import sys

try:
    # Load results
    df = pd.read_csv('$results_file')
    
    if len(df) == 0:
        print("ERROR: No successful runs", file=sys.stderr)
        sys.exit(1)
    
    # Calculate statistics
    p_mean = df['precision'].mean()
    p_std = df['precision'].std(ddof=1) if len(df) > 1 else 0.0
    r_mean = df['recall'].mean()
    r_std = df['recall'].std(ddof=1) if len(df) > 1 else 0.0
    f_mean = df['f1'].mean()
    f_std = df['f1'].std(ddof=1) if len(df) > 1 else 0.0
    s_mean = df['shd'].mean()
    s_std = df['shd'].std(ddof=1) if len(df) > 1 else 0.0
    n_mean = df['nhd'].mean()
    n_std = df['nhd'].std(ddof=1) if len(df) > 1 else 0.0
    
    num_runs = len(df)
    
    # Print summary
    print()
    print("----------------------------------------------------------------------")
    print(f" Granger + Diffusion Summary - Size = $size ({num_runs} runs)")
    print("----------------------------------------------------------------------")
    print(f"Precision: {p_mean:.4f} ± {p_std:.4f}")
    print(f"Recall:    {r_mean:.4f} ± {r_std:.4f}")
    print(f"F1 Score:  {f_mean:.4f} ± {f_std:.4f}")
    print(f"SHD:       {s_mean:.4f} ± {s_std:.4f}")
    print(f"NHD:       {n_mean:.4f} ± {n_std:.4f}")
    print("----------------------------------------------------------------------")
    print()
    
    # Save summary
    with open('$OUTPUT_DIR/size${size}_summary.txt', 'w') as f:
        f.write("=" * 70 + "\\n")
        f.write(f"Granger Causality + Diffusion - Size = $size ({num_runs} runs)\\n")
        f.write("=" * 70 + "\\n")
        f.write(f"Precision: {p_mean:.6f} ± {p_std:.6f}\\n")
        f.write(f"Recall:    {r_mean:.6f} ± {r_std:.6f}\\n")
        f.write(f"F1 Score:  {f_mean:.6f} ± {f_std:.6f}\\n")
        f.write(f"SHD:       {s_mean:.6f} ± {s_std:.6f}\\n")
        f.write(f"NHD:       {n_mean:.6f} ± {n_std:.6f}\\n")
        f.write("=" * 70 + "\\n")
    
    # Append to overall summary
    with open('$OVERALL_SUMMARY', 'a') as f:
        f.write(f"$size,{num_runs},{p_mean:.6f},{p_std:.6f},{r_mean:.6f},{r_std:.6f},{f_mean:.6f},{f_std:.6f},{s_mean:.6f},{s_std:.6f},{n_mean:.6f},{n_std:.6f}\\n")
    
    print(f"Results saved to: $results_file")
    print(f"Summary saved to: $OUTPUT_DIR/size${size}_summary.txt")
    print()

except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        print_msg "ERROR: Failed to compute statistics for size=$size" "$RED"
    fi
    
    echo ""
done

# Print overall summary
echo ""
print_msg "======================================================================" "$GREEN"
print_msg " OVERALL SUMMARY - Granger Causality + Diffusion" "$GREEN"
print_msg "======================================================================" "$GREEN"
echo ""

python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv('$OVERALL_SUMMARY')
    
    if len(df) == 0:
        print("No results to display")
        sys.exit(0)
    
    print()
    print("Size | Runs | Precision        | Recall           | F1 Score         | SHD              | NHD")
    print("-----|------|------------------|------------------|------------------|------------------|------------------")
    
    for _, row in df.iterrows():
        size = int(row['size'])
        runs = int(row['runs'])
        p_m, p_s = row['precision_mean'], row['precision_std']
        r_m, r_s = row['recall_mean'], row['recall_std']
        f_m, f_s = row['f1_mean'], row['f1_std']
        s_m, s_s = row['shd_mean'], row['shd_std']
        n_m, n_s = row['nhd_mean'], row['nhd_std']
        
        print(f"{size:4d} | {runs:4d} | {p_m:.4f} ± {p_s:.4f} | {r_m:.4f} ± {r_s:.4f} | {f_m:.4f} ± {f_s:.4f} | {s_m:.4f} ± {s_s:.4f} | {n_m:.4f} ± {n_s:.4f}")
    
    print()
    
except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
print_msg "======================================================================" "$GREEN"
print_msg "All experiments completed at $(date)" "$GREEN"
print_msg "Results saved in: $OUTPUT_DIR" "$GREEN"
print_msg "Overall summary: $OVERALL_SUMMARY" "$GREEN"
print_msg "======================================================================" "$GREEN"
echo ""