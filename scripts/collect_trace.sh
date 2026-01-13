#!/bin/bash

# Old example (commented out)
#EXAMPLE="iris/examples/00_load/load_bench.py"
#ARGS="--buffer_size 1024 --heap_size 2048 --num_experiments 1 --num_warmup 1"

# New example: all_store_bench.py with minimal args for single kernel launch
EXAMPLE="iris/examples/03_all_store/all_store_bench.py"
# Minimal args: small buffer size (64KB), single kernel launch (no warmup, 1 experiment), small heap size (256MB)
ARGS="--buffer_size_min 65536 --buffer_size_max 65536 --heap_size 268435456 --num_experiments 1 --num_warmup 0 --active_ranks 2"

NUM_RANKS=4
CMD="${EXAMPLE} ${ARGS}"
echo ${CMD}

# Normal run
# torchrun --nproc_per_node=${NUM_RANKS} ${CMD}

# Trace - use roccap_wrapper.py so roccap is spawned as child by each torchrun process
# Output files will be: all_store_bench_rank_0.cap, all_store_bench_rank_1.cap, etc.

# Option 1: Use default naming (script_name_rank_N.cap)
torchrun --nproc_per_node=${NUM_RANKS} roccap_wrapper.py ${CMD}
#torchrun --nproc_per_node=${NUM_RANKS} ${CMD}

# Option 2: Specify custom output directory and pattern
# torchrun --nproc_per_node=${NUM_RANKS} roccap_wrapper.py --output-dir ./traces --output-file-pattern "all_store_bench_rank_{rank}.cap" ${CMD}

# Option 3: Specify custom filename (rank will be appended if not in filename)
# torchrun --nproc_per_node=${NUM_RANKS} roccap_wrapper.py --output-file "all_store_bench.cap" ${CMD}


