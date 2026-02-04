#!/usr/bin/env python3
"""
Minimal roccap wrapper for multi-rank trace capture.
Automatically filters out internal ROCm kernels (e.g., __amd_rocclr_copyBuffer).

Usage: torchrun --nproc_per_node=N roccap_wrapper.py script.py [args...]

Example (2 ranks):
  torchrun --nproc_per_node=2 ./scripts/roccap_wrapper.py ./examples/03_all_store/all_store_bench.py --buffer_size_min 65536 --buffer_size_max 65536 --heap_size 268435456 --num_experiments 1 --num_warmup 0 --active_ranks 2 --verbose

Example (8 ranks):
  torchrun --nproc_per_node=8 ./scripts/roccap_wrapper.py ./examples/03_all_store/all_store_bench.py --buffer_size_min 65536 --buffer_size_max 65536 --heap_size 268435456 --num_experiments 1 --num_warmup 0 --active_ranks 8 --verbose
"""
import os
import sys

# Get rank from torchrun environment
rank = os.environ.get('RANK', '0')

# Get script name for output filename
script_name = os.path.basename(sys.argv[1]).replace('.py', '') if len(sys.argv) > 1 else 'capture'

# Hardcoded dispatch filter: only capture all_store_kernel dispatches
DISP_FILTER = "all_store_kernel/0-"

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
