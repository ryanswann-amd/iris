# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import torch

import json
import numpy as np
import os

# Communication Algorithms
NONE = tl.constexpr(0)  # TODO: None is bad here
ALL_SCATTER = tl.constexpr(1)
ALL_REDUCE = tl.constexpr(2)
ONE_SHOT = tl.constexpr(3)
ONE_SHOT_V1 = tl.constexpr(4)
ONE_SHOT_V2 = tl.constexpr(5)
ALL_GATHER = tl.constexpr(6)


dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    try:
        return dtype_map[datatype]
    except KeyError:
        print(f"Unknown datatype: {datatype}")
        exit(1)


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    return list(dtype_map.keys())[list(dtype_map.values()).index(dtype)]


class JSONWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)

    def add_field(self, key, value):
        self.data[key] = value

    def _write_to_file(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def display(self):
        print(json.dumps(self.data, indent=4))

    def flush(self):
        self._write_to_file()


class Timestamps:
    def __init__(self, num_tiles):
        self.max_ts = torch.iinfo(torch.int64).max
        self.min_ts = 0
        self.mm_begin_timestamp = torch.empty(num_tiles, dtype=torch.int64, device="cuda")
        self.mm_end_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")

        self.comm_begin_timestamp = torch.empty(num_tiles, dtype=torch.int64, device="cuda")
        self.comm_middle_min_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")
        self.comm_middle_max_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")
        self.comm_end_timestamp = torch.zeros(num_tiles, dtype=torch.int64, device="cuda")

    def reset(self):
        self.mm_begin_timestamp.fill_(self.max_ts)
        self.mm_end_timestamp.fill_(self.min_ts)

        self.comm_begin_timestamp.fill_(self.max_ts)
        self.comm_middle_min_timestamp.fill_(self.max_ts)
        self.comm_middle_max_timestamp.fill_(self.min_ts)
        self.comm_end_timestamp.fill_(self.min_ts)

    def to_json(self, filename, gpu_freq):
        cycles_to_us = lambda cycles: (cycles / gpu_freq)

        gemm_begin_us = cycles_to_us(self.mm_begin_timestamp.cpu().numpy())
        gemm_end_us = cycles_to_us(self.mm_end_timestamp.cpu().numpy())

        comm_begin_us = cycles_to_us(self.comm_begin_timestamp.cpu().numpy())
        poll_end_us = cycles_to_us(self.comm_middle_max_timestamp.cpu().numpy())
        op_begin_us = cycles_to_us(self.comm_middle_min_timestamp.cpu().numpy())
        op_end_us = cycles_to_us(self.comm_end_timestamp.cpu().numpy())

        min_timestamp = min(
            np.min(gemm_begin_us),
            np.min(gemm_end_us),
            np.min(comm_begin_us),
            np.min(poll_end_us),
            np.min(op_begin_us),
            np.min(op_end_us),
        )

        gemm_begin_us = gemm_begin_us - min_timestamp
        gemm_end_us = gemm_end_us - min_timestamp
        comm_begin_us = comm_begin_us - min_timestamp
        poll_end_us = poll_end_us - min_timestamp
        op_begin_us = op_begin_us - min_timestamp
        op_end_us = op_end_us - min_timestamp

        data = [
            {
                "tile_id": i,
                "gemm_begin": int(gemm_begin),
                "gemm_end": int(gemm_end),
                "poll_begin": int(comm_begin),
                "poll_end": int(poll_end),
                "op_begin": int(op_begin),
                "op_end": int(
                    op_end,
                ),
                "comm_begin": int(comm_begin),
                "comm_end": int(
                    op_end,
                ),
            }
            for i, (
                gemm_begin,
                gemm_end,
                comm_begin,
                poll_end,
                op_begin,
                op_end,
            ) in enumerate(
                zip(
                    gemm_begin_us,
                    gemm_end_us,
                    comm_begin_us,
                    poll_end_us,
                    op_begin_us,
                    op_end_us,
                )
            )
        ]
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


def is_triton_interpret_set():
    return "TRITON_INTERPRET" in os.environ


@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="""s_waitcnt vmcnt(0)
        s_memrealtime $0
        s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp


@triton.jit
def apply_xcd_reordering(pid, NUM_XCDS: tl.constexpr, NUM_SMS: tl.constexpr):
    """
    Apply XCD (compute die) space-filling curve reordering to program ID.

    This function reorders program IDs to improve locality when multiple compute
    dies (XCDs) are present. It ensures that consecutive PIDs are distributed
    across different XCDs before moving to the next set of programs within an XCD.

    Args:
        pid: The original program ID from tl.program_id(0)
        NUM_XCDS: Number of compute dies (XCDs) in the system
        NUM_SMS: Total number of streaming multiprocessors

    Returns:
        Reordered program ID that optimizes for XCD locality
    """
    if NUM_XCDS != 1:
        return (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    return pid


@triton.jit
def compute_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    """
    Compute 2D tile coordinates (pid_m, pid_n) from linear tile_id using swizzling.

    This function implements a space-filling curve that groups tiles along the M
    dimension to improve memory coalescing and cache locality. Tiles are organized
    into groups of size GROUP_SIZE_M along the M dimension.

    Args:
        tile_id: Linear tile index
        num_pid_m: Number of tiles in the M dimension
        num_pid_n: Number of tiles in the N dimension
        GROUP_SIZE_M: Size of tile groups along M dimension for swizzling

    Returns:
        Tuple of (pid_m, pid_n) representing the 2D coordinates of the tile
    """
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n
