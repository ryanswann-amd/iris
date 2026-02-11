HSA_NO_SCRATCH_RECLAIM=1 \
python3 $(pwd)/benchmark.py \
  -m 2048 \
  -n 16384 \
  -k 131072 \
  --num_ranks 8 \
  --num_xcds 8 \
  --datatype fp16 \
  --block_size_m 512 \
  --block_size_n 128 \
  --block_size_k 64 \
  --group_size_m 1 \
  --benchmark \
  --b_col_major \
  -v \
  --benchmark_pytorch