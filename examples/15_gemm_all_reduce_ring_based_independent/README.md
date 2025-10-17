

# How to run the iris benchmark

```bash overlap mode
python benchmark.py -b -m 3840 -n 3840 -k 4352 -r 8 --datatype "bf16" --benchmark_mode both
```

# how to run the benchmark with different benchmark modes

```bash gemm only mode
python benchmark.py -b -m 3840 -n 3840 -k 4352 -r 8 --datatype "bf16" --benchmark_mode gemm
```

```bash comm only mode
python benchmark.py -b -m 3840 -n 3840 -k 4352 -r 8 --datatype "bf16" --benchmark_mode comm
```

```bash gemm + comm + overlap mode
python benchmark.py -b -m 3840 -n 3840 -k 4352 -r 8 --datatype "bf16" --benchmark_mode all
```


# how to run the hipblaslt + rccl benchmark

```bash hipblaslt + rccl + overlap mode
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 benchmark_rccl_allreduce_hipblaslt.py
```

