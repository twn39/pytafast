#!/bin/bash
# run_comparison.sh - Cross-compare pytafast results with R TTR

# Ensure we are in the project root
cd "$(dirname "$0")/.."

DATA_FILE=${1:-"data/berkshire_1y.csv"}
export DATA_FILE

echo "=== Running comparison for: $DATA_FILE ==="

echo "--- 1. Computing Python results ---"
uv run python scripts/compute_all_py.py

echo -e "\n--- 2. Computing R results ---"
OUTPUT_FILE="temp_r_results.csv" Rscript scripts/compute_all_r.R

echo -e "\n--- 3. Final Comparison Report ---"
# Move results to current dir temporarily for final_compare.py if it expects them locally
# Or just run final_compare.py which reads them from current dir
R_FILE="temp_r_results.csv" uv run python scripts/final_compare.py

echo -e "\n--- 4. Cleaning up temporary data ---"
rm py_all_results.csv temp_r_results.csv
