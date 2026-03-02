# GEMM + AllScatter Benchmark Results

## Hardware & Configuration

- **Hardware**: 8x AMD MI300X (304 CUs each)
- **Datatype**: fp16
- **GPUs**: 8
- **Tiling**: BLK_M=256, BLK_N=64, BLK_K=64, gsize_m=6
- **Pipeline stages**: 2
- **Kernel**: `examples/23_gemm_all_scatter_tracing/` (persistent GEMM + all-scatter via `ctx.put`)
- **Benchmark script**: `benchmark/examples/benchmark_gemm_all_scatter.py`
- **Config file**: `dataset/gemm_all_scatter.json`

## Results

N=4096, K=14336 sweep over M (typical LLM FF layer dimensions).

> **Note**: Total ms is measured end-to-end (including inter-GPU barriers) via `iris.do_bench`.
> Kernel ms is the per-GPU CUDA-event average for the fused GEMM+AllScatter kernel.

| M    | N    | K     | Total ms | TFLOPS  | Kernel ms |
|------|------|-------|----------|---------|-----------|
| 1    | 4096 | 14336 | 0.390    | 0.301   | 0.259     |
| 2    | 4096 | 14336 | 0.429    | 0.548   | 0.270     |
| 4    | 4096 | 14336 | 0.463    | 1.015   | 0.269     |
| 8    | 4096 | 14336 | 0.450    | 2.087   | 0.271     |
| 16   | 4096 | 14336 | 0.401    | 4.683   | 0.272     |
| 32   | 4096 | 14336 | 0.412    | 9.113   | 0.273     |
| 64   | 4096 | 14336 | 0.430    | 17.477  | 0.285     |
| 128  | 4096 | 14336 | 0.501    | 30.002  | 0.345     |
| 256  | 4096 | 14336 | 0.600    | 50.142  | 0.423     |
| 512  | 4096 | 14336 | 0.585    | 102.786 | 0.436     |
| 1024 | 4096 | 14336 | 0.696    | 172.791 | 0.479     |

## Observations

- **Small M (≤ 32)**: Total time is dominated by launch overhead and communication latency (~0.39–0.46 ms). TFLOPS are low because there isn't enough compute work to saturate the GPU.
- **Moderate M (64–256)**: Increasing TFLOPS as the GEMM starts to become compute-bound; kernel time rises from 0.285 ms to 0.423 ms.
- **Large M (512–1024)**: Strong scaling into compute-bound territory. At M=1024 we achieve **172.8 TFLOPS** with a total end-to-end time of only **0.696 ms**, demonstrating effective communication/computation overlap via the fused persistent GEMM+AllScatter kernel.
