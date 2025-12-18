# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for get_accumulator_dtype utility function.
"""

import pytest
import triton.language as tl
import iris


def test_get_accumulator_dtype_int8():
    """Test that int8 promotes to int32."""
    result = iris.get_accumulator_dtype(tl.int8)
    assert result == tl.int32


def test_get_accumulator_dtype_int16():
    """Test that int16 promotes to int32."""
    result = iris.get_accumulator_dtype(tl.int16)
    assert result == tl.int32


def test_get_accumulator_dtype_int32():
    """Test that int32 stays int32."""
    result = iris.get_accumulator_dtype(tl.int32)
    assert result == tl.int32


def test_get_accumulator_dtype_float16():
    """Test that float16 (fp16) promotes to float32."""
    result = iris.get_accumulator_dtype(tl.float16)
    assert result == tl.float32


def test_get_accumulator_dtype_bfloat16():
    """Test that bfloat16 (bf16) promotes to float32."""
    result = iris.get_accumulator_dtype(tl.bfloat16)
    assert result == tl.float32


def test_get_accumulator_dtype_float32():
    """Test that float32 stays float32."""
    result = iris.get_accumulator_dtype(tl.float32)
    assert result == tl.float32


def test_get_accumulator_dtype_float64():
    """Test that float64 stays float64."""
    result = iris.get_accumulator_dtype(tl.float64)
    assert result == tl.float64


@pytest.mark.parametrize(
    "input_dtype,expected_output",
    [
        (tl.int8, tl.int32),
        (tl.int16, tl.int32),
        (tl.int32, tl.int32),
        (tl.float16, tl.float32),
        (tl.bfloat16, tl.float32),
        (tl.float32, tl.float32),
        (tl.float64, tl.float64),
    ],
)
def test_get_accumulator_dtype_parametrized(input_dtype, expected_output):
    """Parametrized test for all supported dtype promotions."""
    result = iris.get_accumulator_dtype(input_dtype)
    assert result == expected_output


def test_get_accumulator_dtype_precision_loss_prevention():
    """Test that half precision types promote to prevent precision loss."""
    # fp16 and bf16 should promote to fp32 to prevent precision loss in accumulation
    assert iris.get_accumulator_dtype(tl.float16) == tl.float32
    assert iris.get_accumulator_dtype(tl.bfloat16) == tl.float32
    # fp32 and fp64 should stay as is (sufficient precision)
    assert iris.get_accumulator_dtype(tl.float32) == tl.float32
    assert iris.get_accumulator_dtype(tl.float64) == tl.float64


def test_get_accumulator_dtype_overflow_prevention():
    """Test that small integer types promote to prevent overflow."""
    # int8 and int16 should promote to int32 to prevent overflow
    assert iris.get_accumulator_dtype(tl.int8) == tl.int32
    assert iris.get_accumulator_dtype(tl.int16) == tl.int32
    # int32 is already wide enough
    assert iris.get_accumulator_dtype(tl.int32) == tl.int32
