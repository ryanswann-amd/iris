################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Part of the code adapted from
# https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_sp_decode_attn.py
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


import sys
from pathlib import Path
import pytest
from typing import List, Optional
from argparse import Namespace

import torch
import iris


pytestmark = pytest.mark.multi_rank_required

project_root = Path(__file__).resolve()
while not (project_root / "tests").is_dir() or not (project_root / "examples").is_dir():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent
print(f"Project Root: {project_root}")

module_dir = project_root / "examples" / "13_flash_decode"
print(f"Module Directory: {module_dir}")

target_file = module_dir / "flash_decode_fused_layer.py"
if module_dir.exists():
    sys.path.insert(0, str(module_dir))
    print(f"'{module_dir}' was added to sys.path.")
else:
    print("ERROR: Target directory not found")

from flash_decode_fused_layer import flash_decode_fused_layer  # noqa: E402
from utils import print_correctness_report  # noqa: E402


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens_per_rank: List[int],
    block_tables: torch.Tensor,
    scale: float,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_cpu = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len, kv_len = query_lens[i], kv_lens_per_rank[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            gqa_ratio = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, gqa_ratio, dim=1)
            v = torch.repeat_interleave(v, gqa_ratio, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=query.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if soft_cap is not None and soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)


def prepare_correctness_data(cfg, args, num_query_heads, num_kv_heads, NUM_BLOCKS):
    head_dim = cfg["head_dim"]
    if args.rank == 0:
        query = torch.randn(cfg["num_seqs"], num_query_heads, head_dim, dtype=cfg["dtype"]) / 10
        key_value_cache = torch.randn(NUM_BLOCKS, 2, cfg["block_size"], num_kv_heads, head_dim, dtype=cfg["dtype"]) / 10
    else:
        query = torch.empty(cfg["num_seqs"], num_query_heads, head_dim, dtype=cfg["dtype"])
        key_value_cache = torch.empty(NUM_BLOCKS, 2, cfg["block_size"], num_kv_heads, head_dim, dtype=cfg["dtype"])

    query = torch.from_numpy(args.shmem.broadcast(query.cpu().numpy(), source_rank=0)).to(query.device)
    key_value_cache = torch.from_numpy(args.shmem.broadcast(key_value_cache.cpu().numpy(), source_rank=0)).to(
        key_value_cache.device
    )

    return {"query": query, "key_value_cache": key_value_cache}



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize("head_dim", [128])