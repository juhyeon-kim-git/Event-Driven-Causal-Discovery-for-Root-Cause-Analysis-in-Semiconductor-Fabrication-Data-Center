#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply threshold to a CSV adjacency matrix and binarize it.

Rule:
  value >= threshold -> 1
  value <  threshold -> 0

Options:
- remove self-loops (diagonal = 0)
- use absolute value
"""

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Threshold a CSV adjacency matrix and save as binary (0/1)."
    )
    parser.add_argument("--input", required=True, help="Input adjacency CSV path")
    parser.add_argument("--output", required=True, help="Output binary CSV path")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value")
    parser.add_argument(
        "--header",
        action="store_true",
        help="Set if input CSV has a header row"
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="Apply threshold on absolute values"
    )
    parser.add_argument(
        "--no-self-loop",
        action="store_true",
        help="Force diagonal entries to 0"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Numerical epsilon (default: 0.0)"
    )

    args = parser.parse_args()

    # =====================
    # Load CSV
    # =====================
    df = pd.read_csv(args.input, header=0 if args.header else None)
    mat = df.values.astype(float)

    # =====================
    # Thresholding
    # =====================
    if args.abs:
        mat = np.abs(mat)

    bin_mat = (mat >= (args.threshold - args.eps)).astype(int)

    # =====================
    # Remove self-loops
    # =====================
    if args.no_self_loop:
        np.fill_diagonal(bin_mat, 0)

    # =====================
    # Save
    # =====================
    pd.DataFrame(bin_mat).to_csv(
        args.output,
        index=False,
        header=False
    )

    print("[DONE] Thresholding complete")
    print(f"  input       : {args.input}")
    print(f"  output      : {args.output}")
    print(f"  threshold   : {args.threshold}")
    print(f"  abs         : {args.abs}")
    print(f"  no_selfloop : {args.no_self_loop}")


if __name__ == "__main__":
    main()
