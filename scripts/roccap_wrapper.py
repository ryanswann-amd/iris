#!/usr/bin/env python3
import os
import sys
import argparse
import shutil

# Example:
# cd examples/29_ops_all_gather_matmul
# torchrun --nproc_per_node=4 --standalone ../../scripts/roccap_wrapper.py -k _fused_all_gather_matmul_kernel example.py -m 1024 -n 128

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kernel", type=str, default="_fused_all_gather_matmul_kernel")

# Everything else: first is the script to run, rest are passed through to it
parsed, unknown = parser.parse_known_args()
if not unknown:
    sys.exit("Usage: roccap_wrapper.py [-d DISP_FILTER] <script> [script_args...]")
child_script = unknown[0]
child_args = unknown[1:]

# Get rank from torchrun environment
rank = os.environ.get("RANK", "0")

# Hardcoded dispatch filter: only capture dispatches
DISP_FILTER = f"{parsed.kernel}/0-"

# Build roccap command: capture --loglevel trace --file <output> --disp <filter> python3 <script> [args...]
roccap_args = [
    "capture",
    "--loglevel",
    "trace",
    "--file",
    f"{parsed.kernel}_rank_{rank}.cap",
    "--disp",
    DISP_FILTER,
    "python3",
    child_script,
] + child_args

# Replace current process with roccap; set simulation env so Iris uses torch allocator
os.environ["IRIS_SIMULATION"] = "1"

roccap_path = shutil.which("roccap")
if roccap_path is None:
    sys.exit("roccap not found in PATH. Install it or activate the environment that provides it.")
print(f"Executing the command: roccap {' '.join(roccap_args)}")
os.execvp(roccap_path, ["roccap"] + roccap_args)
