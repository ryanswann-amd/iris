#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
irisrun: A launcher for distributed Iris programs.

Similar to torchrun, this tool automatically manages distributed initialization
by finding free ports and setting up the environment for multi-GPU execution.

Usage:
    irisrun --nproc_per_node=N script.py [script_args...]

Example:
    irisrun --nproc_per_node=2 examples/00_load/load_bench.py --verbose
"""

import argparse
import os
import socket
import sys


def _find_free_port():
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def _distributed_worker(local_rank, world_size, master_addr, master_port, script_path, script_args):
    """Worker function that sets up environment and runs the target script."""
    # Set environment variables for distributed training
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Set CUDA device for this process
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except ImportError:
        pass  # torch may not be installed yet, that's ok

    # Restore sys.argv to make it appear as if the script was called directly
    sys.argv = [script_path] + script_args

    # Execute the script in the current namespace
    try:
        with open(script_path, encoding="utf-8") as f:
            code = compile(f.read(), script_path, "exec")
            exec(code, {"__name__": "__main__", "__file__": script_path})
    except SystemExit as e:
        # Propagate exit code from script
        sys.exit(e.code if isinstance(e.code, int) else 1)
    except Exception as e:
        print(f"Error in worker {local_rank}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for irisrun."""
    parser = argparse.ArgumentParser(
        description="Launch distributed Iris programs with automatic port management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  irisrun --nproc_per_node=2 examples/00_load/load_bench.py --verbose
  irisrun --nproc_per_node=4 examples/01_store/store_bench.py
        """,
    )

    parser.add_argument(
        "--nproc_per_node",
        type=int,
        required=True,
        help="Number of processes to launch per node (typically number of GPUs)",
    )

    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="Master node address (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Master node port (default: auto-selected free port)",
    )

    parser.add_argument("script", type=str, help="Python script to run")

    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments for the script")

    args = parser.parse_args()

    # Find a free port if not specified
    master_port = args.master_port if args.master_port is not None else _find_free_port()
    master_addr = args.master_addr

    print(f"[irisrun] Launching {args.nproc_per_node} processes")
    print(f"[irisrun] Master address: {master_addr}:{master_port}")
    print(f"[irisrun] Script: {args.script}")
    print(f"[irisrun] Script args: {args.script_args}")

    try:
        # Import torch.multiprocessing here, after args are parsed
        import torch.multiprocessing as mp

        mp.spawn(
            _distributed_worker,
            args=(args.nproc_per_node, master_addr, master_port, args.script, args.script_args),
            nprocs=args.nproc_per_node,
            join=True,
        )
    except ImportError as e:
        print(f"[irisrun] Error: PyTorch is required to run irisrun: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[irisrun] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[irisrun] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
