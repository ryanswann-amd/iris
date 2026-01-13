#!/usr/bin/env python3
"""
Wrapper script to run a Python script under roccap capture.
This allows torchrun to spawn this wrapper, which then spawns roccap as a child process.
"""
import sys
import subprocess
import os
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper to run Python scripts under roccap capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default filename (script_name_rank_N.cap)
  roccap_wrapper.py iris/examples/00_load/load_bench.py --buffer_size 1024
  
  # Specify custom output filename
  roccap_wrapper.py --output-file load_bench_rank_0.cap iris/examples/00_load/load_bench.py --buffer_size 1024
  
  # Use filename pattern with {rank} placeholder
  roccap_wrapper.py --output-file-pattern "load_bench_rank_{rank}.cap" iris/examples/00_load/load_bench.py
        """
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output capture file name (will include rank if not in filename)"
    )
    parser.add_argument(
        "--output-file-pattern",
        type=str,
        default=None,
        help="Output capture file name pattern with {rank} placeholder (e.g., 'load_bench_rank_{rank}.cap')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for capture files (default: current directory)"
    )
    
    # Parse known args to separate wrapper args from script args
    args, remaining = parser.parse_known_args()
    
    if len(remaining) < 1:
        parser.print_help()
        print("\nError: Script path is required")
        sys.exit(1)
    
    # Find roccap command
    roccap_cmd = shutil.which("roccap")
    if not roccap_cmd:
        print("Error: roccap command not found in PATH")
        sys.exit(1)
    
    # Get the script and its arguments
    script = remaining[0]
    script_args = remaining[1:]
    
    # Preserve all environment variables (especially torchrun's RANK, WORLD_SIZE, etc.)
    env = os.environ.copy()
    rank = env.get("RANK", env.get("LOCAL_RANK", "0"))
    
    # Determine output filename
    if args.output_file_pattern:
        # Use pattern with {rank} placeholder
        output_file = args.output_file_pattern.format(rank=rank)
    elif args.output_file:
        # Use provided filename, add rank if not present
        output_file = args.output_file
        if "{rank}" in output_file:
            output_file = output_file.format(rank=rank)
        elif f"_rank_{rank}" not in output_file and f"_rank{rank}" not in output_file:
            # Add rank to filename if not already present
            base, ext = os.path.splitext(output_file)
            output_file = f"{base}_rank_{rank}{ext}"
    else:
        # Generate default filename from script name
        script_name = Path(script).stem
        output_file = f"{script_name}_rank_{rank}.cap"
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full path to output file
    output_path = output_dir / output_file
    
    # Build the command: roccap capture --file <output> --loglevel trace python3 <script> <args>
    cmd = [
        roccap_cmd,
        "capture",
        "--file", str(output_path),
        "--loglevel", "trace",
        sys.executable,  # Use the same Python interpreter
        script,
    ] + script_args
    
    print(f"[roccap_wrapper] Rank: {rank}, World Size: {env.get('WORLD_SIZE', 'N/A')}")
    print(f"[roccap_wrapper] Output file: {output_path}")
    print(f"[roccap_wrapper] Running: {' '.join(cmd)}")
    
    # Run roccap capture with the script
    try:
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n[roccap_wrapper] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[roccap_wrapper] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
