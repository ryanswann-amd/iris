#!/bin/bash
# Compare Iris vs Triton Reference - Run both and show results

set -e

NPROC=${1:-8}  # Default to 8 GPUs
NAME=${2:-comparison-test}

echo "================================================================================"
echo "Iris vs Triton Comparison"
echo "================================================================================"
echo "GPUs: $NPROC"
echo "Test name: $NAME"
echo ""

# Run Iris benchmark
echo "================================================================================"
echo "Running Iris MoE Benchmark..."
echo "================================================================================"
torchrun --nproc-per-node=$NPROC bench_iris_mlp.py --tp 1 --ep $NPROC --name "iris-$NAME"

echo ""
echo "================================================================================"
echo "Running Triton Reference Benchmark..."
echo "================================================================================"
cd reference
torchrun --nproc-per-node=$NPROC bench_mlp.py --tp 1 --ep $NPROC --name "triton-$NAME"
cd ..

echo ""
echo "================================================================================"
echo "Benchmark Complete! Comparing Results..."
echo "================================================================================"

# Find the CSV files
IRIS_CSV=$(find logs/iris-$NAME -name "*.csv" | head -1)
TRITON_CSV=$(find logs/triton-$NAME -name "*.csv" | head -1)

if [ -f "$IRIS_CSV" ] && [ -f "$TRITON_CSV" ]; then
    echo ""
    echo "Iris Results:"
    echo "-------------"
    head -20 "$IRIS_CSV"
    
    echo ""
    echo "Triton Results:"
    echo "---------------"
    head -20 "$TRITON_CSV"
    
    echo ""
    echo "================================================================================"
    echo "CSV files saved:"
    echo "  Iris:   $IRIS_CSV"
    echo "  Triton: $TRITON_CSV"
    echo ""
    echo "Roofline plots saved in:"
    echo "  logs/iris-$NAME/"
    echo "  logs/triton-$NAME/"
    echo "================================================================================"
else
    echo "Warning: Could not find CSV output files"
    echo "Check logs/ directory for results"
fi

