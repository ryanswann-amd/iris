#!/usr/bin/env python3
"""
Iris MLP Benchmark - EXACT copy of bench_mlp.py but with Iris communication
The ONLY difference: Uses Iris for DP<->EP conversion instead of Triton's symmetric memory
"""

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import torch.distributed as dist
import argparse
import sys
import os

# Add parent directory to path for local triton_kernels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import triton_kernels.roofline as roofline
from triton_kernels.matmul import matmul
from triton_kernels.target_info import get_cdna_version
from triton_kernels.topk import topk
from triton_kernels.reduce import reduce
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata
from triton_kernels.distributed import make_expt_dict_uniform, make_expt_assignment, symm_mem_pool

# Import reference utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))
import distributed as triton_dist
from bench_utils import prepare_mlp_numerics, resolve_x_dtype
import tempfile

# Import Iris
import iris

# Import Iris MoE conversion functions
from moe_iris_v2 import convert_dp_to_ep_iris, convert_ep_to_dp_iris


def bench_iris_mlp(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP):
    """
    EXACT copy of bench_mlp but using Iris for DP<->EP conversion
    """
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    rank, world_size = triton_dist.setup()
    dev = f"cuda:{rank}"
    DP = world_size
    batch = batch_per_expt * n_expts_tot // n_expts_act

    assert n_expts_tot % EP == 0, f"{n_expts_tot=}, {EP=}, n_expts_tot must be divisible by EP"
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # Initialize Iris (only for MoE, not dense)
    shmem = None
    if n_expts_tot > 1:
        # Iris requires distributed to be initialized, even for single GPU
        if not dist.is_initialized():
            # Initialize dummy process group for single GPU
            dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12355", rank=0, world_size=1)
        shmem = iris.iris()

    # -- init data --
    # weights
    wg = triton_dist.broadcast(torch.randn((dim1, n_expts_tot), device=dev))
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = triton_dist.broadcast(torch.randn((n_expts_tot,), device=dev))
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_indx = (rank // TP) % EP
    groups = [list(range(ep * TP, (ep + 1) * TP)) for ep in range(EP)]
    b2 = triton_dist.broadcast(b2, src=ep_indx * TP, groups=groups, group_idx=ep_indx)

    # -- numerics --
    numerics = prepare_mlp_numerics(batch, w_dtype, wg, w1, w2)
    wg, w1, w2 = numerics.wg, numerics.w1, numerics.w2
    pcg, pc1, pc2, act = numerics.pcg, numerics.pc1, numerics.pc2, numerics.activation

    # -- benchmark --
    x_dtype = resolve_x_dtype(x_dtype)

    input_x = torch.randn((batch // DP, dim1), device=dev)
    expt_assignment = triton_dist.create_expt_assignment(EP, n_expts_tot, torch.device(dev))
    triton_dist.initialize_matmul(batch, dim1, dim2, n_expts_act, n_expts_tot, input_x.dtype)

    # run layer
    fpath = Path(tempfile.mktemp())
    proton.start(str(fpath), hook="triton")
    input_x = input_x.to(x_dtype)
    xg = input_x.to(wg.dtype if n_expts_tot > 1 else input_x.dtype)
    for i in range(100):
        if n_expts_tot > 1:  # sparse (MoE)
            logits = matmul(xg, wg, bg, precision_config=pcg)

            # ===== IRIS ROUTING (ONLY DIFFERENCE) =====
            # Use Iris for DP->EP conversion
            ep_rank = (rank // TP) % EP  # Expert parallelism rank
            expt_map = expt_assignment.expt_map[ep_rank, :]
            logits_global = topk(
                logits,
                n_expts_act,
                apply_softmax=True,
                y_indx=None,
                all_gather=True,
            )
            active_indx = logits_global.indx
            expt_sizes = logits_global.mask_metadata.col_sum
            dispatch_indx = logits_global.mask_metadata.row_sorted_indx
            combine_indx = logits_global.mask_metadata.col_sorted_indx
            logits_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])

            # Allocate Iris symmetric memory buffers
            n_tokens_global = batch
            dp_to_ep_buf = shmem.zeros((n_tokens_global * n_expts_act, dim1), dtype=input_x.dtype)
            ep_to_dp_buf = shmem.zeros((batch // DP, dim2), dtype=input_x.dtype)

            # DP -> EP using Iris
            x = convert_dp_to_ep_iris(input_x, expt_assignment, active_indx, dispatch_indx, shmem, dp_to_ep_buf)
            rdata = remap_ragged_tensor_metadata(logits_global_metadata, expt_map)
            gather_indx = None
            scatter_indx = None
            # ===== END IRIS ROUTING =====

        else:  # dense
            x = triton_dist.all_gather(input_x, dim=0)
            rdata, gather_indx, scatter_indx = None, None, None

        if x.nelement() > 0:
            x = matmul(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
            x = matmul(x, w2, b2 if rank % TP == 0 else None, rdata, scatter_indx=scatter_indx, precision_config=pc2)

        if n_expts_tot > 1:  # sparse (MoE)
            # ===== IRIS EP->DP (ONLY DIFFERENCE) =====
            x = convert_ep_to_dp_iris(x, expt_assignment, active_indx, combine_indx, shmem, ep_to_dp_buf)
            # ===== END IRIS EP->DP =====
            # Weighted average
            x = x.view(-1, n_expts_act, x.shape[-1])
            x, _ = reduce(x, dim=1)
        else:
            # For dense case, use standard reduce_scatter
            x = triton_dist.reduce_scatter(x, n_expts_act, metadata=None, expt_assignment=None)

    proton.finalize()
    triton_dist.cleanup_matmul()
    return roofline.parse_profile(fpath.with_suffix(".hatchet"), useful_op_regex=".*matmul.*")


def roofline_mlp(batch_sizes, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, name="", verbose=True):
    """EXACT copy from reference"""
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-TP{TP}-EP{EP}/")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = roofline.compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        TP,
        EP,  # fixed args
        bench_fn=bench_iris_mlp,  # function to benchmark
        intensity_proxy_name="batch_per_expt",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        verbose=verbose,  # options
        out_path=out_path.with_suffix(".csv"),
    )  # output path
    png_path = roofline.plot_roofline(
        series=[csv_path],  # roofline data to plot
        flops_dtype=x_dtype,  # dtype to use for FLOPS roof
        xlabel="batch_per_expt",
        title=out_path,  # plot option
        out_path=out_path.with_suffix(".png"),  # output path
        max_tbps="memset",
        max_tflops="cublas",
    )  # hardware limits

    return png_path


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    batch_sizes_dense = [*range(128, 8192, 128)]
    batch_ranges_moe = [(2 ** (2 + k), 2 ** (3 + k), min(2**k, 32)) for k in range(8)]
    batch_sizes_moe = list(chain(*[range(*r) for r in batch_ranges_moe]))
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    rank, world_size = triton_dist.setup()
    if world_size > 1:
        # Running all workloads at once may cause OOM on some GPUs such as H100 80GB.
        # Thus we request users to run each workload separately.
        # For example, all eligible combinations of options are listed below when four GPUs are used:
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 2 --ep 2 --name iris-gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 1 --ep 4 --name iris-gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 4 --ep 1 --name iris-gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 4 --ep 1 --name iris-dense
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 2 --ep 2 --name iris-gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 1 --ep 4 --name iris-gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 4 --ep 1 --name iris-gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_iris_mlp.py --tp 4 --ep 1 --name iris-dense --quantized
        parser = argparse.ArgumentParser()
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--ep", type=int, default=1)
        parser.add_argument("--name", type=str, default="")
        parser.add_argument("--quantized", action="store_true")
        args = parser.parse_args()
        TP, EP = args.tp, args.ep
        x_dtype, w_dtype = quantized_dtypes if args.quantized else dense_dtypes
        name = args.name
        assert TP * EP == world_size, f"TP * EP = {TP} * {EP} = {TP * EP} != {world_size}"
        if name == "dense":
            # dense
            roofline_mlp(
                batch_sizes_dense, 6144, 6144, 1, 1, x_dtype, w_dtype, TP, EP, name="iris-dense", verbose=False
            )
        elif name:
            # sparse
            roofline_mlp(
                batch_sizes_moe, 6144, 24576, 128, 2, x_dtype, w_dtype, TP, EP, name=f"iris-{name}", verbose=False
            )
        else:
            assert False, "Please specify --name"
    else:
        # single GPU workload - run all configurations
        for x_dtype, w_dtype in [dense_dtypes, quantized_dtypes]:
            roofline_mlp(
                batch_sizes_dense, 6144, 6144, 1, 1, x_dtype, w_dtype, TP=1, EP=1, name="iris-dense", verbose=False
            )
            roofline_mlp(
                batch_sizes_moe,
                6144,
                24576,
                128,
                2,
                x_dtype,
                w_dtype,
                TP=1,
                EP=1,
                name="iris-gpt-oss-x2",
                verbose=False,
            )
