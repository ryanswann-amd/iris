#!/usr/bin/env python3
"""
Minimal roccap wrapper for multi-rank GEMM trace capture.

Usage (run from iris directory): 
  torchrun --nproc_per_node=N examples/07_gemm_all_scatter/roccap_wrapper_gemm.py examples/07_gemm_all_scatter/benchmark.py [args...]

Example (2 ranks):
  torchrun --nproc_per_node=2 examples/07_gemm_all_scatter/roccap_wrapper_gemm.py examples/07_gemm_all_scatter/benchmark.py -m 512 -n 512 -k 512 --heap_size 268435456 --gemm_sms 256 --verbose

Example (8 ranks):
  torchrun --nproc_per_node=8 examples/07_gemm_all_scatter/roccap_wrapper_gemm.py examples/07_gemm_all_scatter/benchmark.py -m 512 -n 512 -k 512 --heap_size 268435456 --gemm_sms 256 --verbose
"""
import os
import sys

# Get rank from torchrun environment
rank = os.environ.get('RANK', '0')

# Use descriptive name for output files
script_name = "gemm_all_scatter"

# Hardcoded dispatch filter: only capture persistent_gemm_all_scatter dispatches
DISP_FILTER = "persistent_gemm_all_scatter/0-"

# Build roccap command: capture --loglevel trace --file <output> --disp <filter> python3 <script> <args>
args = [
    "capture",
    "--loglevel", "trace",
    "--file", f"{script_name}_rank_{rank}.cap",
    "--disp", DISP_FILTER,
    "python3"
] + sys.argv[1:]

# Replace current process with roccap
os.execvp("roccap", ["roccap"] + args)
