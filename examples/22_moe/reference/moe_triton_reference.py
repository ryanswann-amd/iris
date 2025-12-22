#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reference Triton MoE - using standard Triton distributed kernels
No Iris dependency - uses Triton's native symmetric memory
"""

import torch
import torch.distributed as dist
import sys
import os

# Add parent directory to path for triton_kernels import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triton_kernels.distributed import (
    make_expt_dict_uniform,
    make_expt_assignment,
    symm_mem_pool,
    convert_dp_to_ep,
    convert_ep_to_dp,
)
from triton_kernels.reduce import reduce
from triton_kernels.topk import topk
from triton_kernels.matmul import matmul
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata


def mixture_of_expt_triton(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act):
    """
    Standard Triton MoE implementation using Triton's symmetric memory
    """
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    expt_map = expt_assignment.expt_map[rank, :]
    n_tokens_local, d_model = x_dp_local.shape
    n_tokens_global = n_tokens_local * n_ranks

    # Use Triton for routing
    l_global_active = topk(l_dp_local, n_expts_act, apply_softmax=True, dim=1, y_indx=None, all_gather=True)
    active_indx = l_global_active.indx
    expt_sizes = l_global_active.mask_metadata.col_sum
    dispatch_indx = l_global_active.mask_metadata.row_sorted_indx
    combine_indx = l_global_active.mask_metadata.col_sorted_indx
    x_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])

    # Convert DP → EP using Triton's native implementation
    y_ep_local = convert_dp_to_ep(x_dp_local, expt_assignment, active_indx, dispatch_indx)
    y_ep_local_metadata = remap_ragged_tensor_metadata(x_global_metadata, expt_map)

    # Use Triton's optimized matmul (tl.dot)
    y_ep_local = matmul(y_ep_local, w_ep_local, b_ep_local, a_ragged_metadata=y_ep_local_metadata)

    # Convert EP → DP using Triton's native implementation
    y_dp_local = convert_ep_to_dp(y_ep_local, expt_assignment, active_indx, combine_indx)

    # Weighted average
    y_dp_local = y_dp_local.view(-1, n_expts_act, y_dp_local.shape[-1])
    z_dp_local, _ = reduce(y_dp_local, dim=1)

    return z_dp_local


def moe_triton_reference(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act):
    """
    Reference Triton MoE - wrapper for standard implementation
    """
    return mixture_of_expt_triton(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act)


__all__ = ["moe_triton_reference"]
