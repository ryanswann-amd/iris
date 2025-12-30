# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Common tensor operations for Iris implementations.

This module contains shared tensor creation methods used by both
Triton and Gluon backends.
"""

import torch


def create_zeros(iris_instance, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """
    Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
    The tensor is allocated on the Iris symmetric heap.

    Args:
        iris_instance: The Iris instance (Iris or IrisGluon)
        *size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword Arguments:
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see torch.set_default_dtype()).
        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
            Default: False.

    Returns:
        torch.Tensor: Zero-initialized tensor
    """
    iris_instance.debug(f"zeros: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

    # Use global default dtype if None is provided
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Use current device if none specified
    if device is None:
        device = iris_instance.device

    # Validate device compatibility with Iris
    iris_instance._throw_if_invalid_device(device)

    # Parse size and calculate number of elements
    size, num_elements = iris_instance._parse_size(size)

    # If out is provided, use it; otherwise allocate new tensor
    if out is not None:
        iris_instance._throw_if_invalid_output_tensor(out, num_elements, dtype)
        # Fill with zeros
        out.zero_()
        # Create a reshaped view of the out tensor
        tensor = out.view(size)
    else:
        tensor = iris_instance._allocate(num_elements=num_elements, dtype=dtype)
        # Fill with zeros
        tensor.zero_()
        # Reshape to the desired size
        tensor = tensor.reshape(size)

    # Apply the requested layout
    tensor = iris_instance._apply_layout(tensor, layout)

    # Set requires_grad if specified
    if requires_grad:
        tensor.requires_grad_()

    return tensor


def create_ones(iris_instance, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """
    Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
    The tensor is allocated on the Iris symmetric heap.

    Args:
        iris_instance: The Iris instance (Iris or IrisGluon)
        *size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword Arguments:
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see torch.set_default_dtype()).
        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
            Default: False.

    Returns:
        torch.Tensor: Ones-initialized tensor
    """
    iris_instance.debug(f"ones: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

    # Use global default dtype if None is provided
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Use current device if none specified
    if device is None:
        device = iris_instance.device

    # Validate device compatibility with Iris
    iris_instance._throw_if_invalid_device(device)

    # Parse size and calculate number of elements
    size, num_elements = iris_instance._parse_size(size)

    # If out is provided, use it; otherwise allocate new tensor
    if out is not None:
        iris_instance._throw_if_invalid_output_tensor(out, num_elements, dtype)
        # Fill with ones
        out.fill_(1)
        # Create a reshaped view of the out tensor
        tensor = out.view(size)
    else:
        tensor = iris_instance._allocate(num_elements=num_elements, dtype=dtype)
        # Fill with ones
        tensor.fill_(1)
        # Reshape to the desired size
        tensor = tensor.reshape(size)

    # Apply the requested layout
    tensor = iris_instance._apply_layout(tensor, layout)

    # Set requires_grad if specified
    if requires_grad:
        tensor.requires_grad_()

    return tensor


def create_full(
    iris_instance, size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
):
    """
    Creates a tensor of size size filled with fill_value. The tensor's dtype is inferred from fill_value.
    The tensor is allocated on the Iris symmetric heap.

    Args:
        iris_instance: The Iris instance (Iris or IrisGluon)
        size (int...): a list, tuple, or torch.Size of integers defining the shape of the output tensor.
        fill_value (Scalar): the value to fill the output tensor with.

    Keyword Arguments:
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see torch.set_default_dtype()).
        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
            Default: False.

    Returns:
        torch.Tensor: Tensor filled with fill_value
    """
    iris_instance.debug(
        f"full: size = {size}, fill_value = {fill_value}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}"
    )

    # Infer dtype from fill_value if not provided
    if dtype is None:
        if isinstance(fill_value, (int, float)):
            if isinstance(fill_value, float):
                dtype = torch.get_default_dtype()
            else:
                dtype = torch.int64
        else:
            # For other types (like tensors), use their dtype
            dtype = torch.get_default_dtype()

    # Use current device if none specified
    if device is None:
        device = iris_instance.device

    # Validate device compatibility with Iris
    iris_instance._throw_if_invalid_device(device)

    # Parse size and calculate number of elements
    size, num_elements = iris_instance._parse_size(size)

    # If out is provided, use it; otherwise allocate new tensor
    if out is not None:
        iris_instance._throw_if_invalid_output_tensor(out, num_elements, dtype)
        # Fill with the specified value
        out.fill_(fill_value)
        # Create a reshaped view of the out tensor
        tensor = out.view(size)
    else:
        tensor = iris_instance._allocate(num_elements=num_elements, dtype=dtype)
        # Fill with the specified value
        tensor.fill_(fill_value)
        # Reshape to the desired size
        tensor = tensor.reshape(size)

    # Apply the requested layout
    tensor = iris_instance._apply_layout(tensor, layout)

    # Set requires_grad if specified
    if requires_grad:
        tensor.requires_grad_()

    return tensor


def create_zeros_like(
    iris_instance,
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format,
):
    """
    Returns a tensor filled with the scalar value 0, with the same size as input,
    allocated on the Iris symmetric heap.

    Args:
        iris_instance: The Iris instance (Iris or IrisGluon)
        input (Tensor): the size of input will determine size of the output tensor.

    Keyword Arguments:
        dtype (torch.dtype, optional): the desired data type of returned Tensor.
            Default: if None, defaults to the dtype of input.
        layout (torch.layout, optional): the desired layout of returned tensor.
            Default: if None, defaults to the layout of input. Note: Iris tensors are always contiguous (strided).
        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, defaults to the device of input. Must be compatible with this Iris instance.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
            Default: False.
        memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
            Default: torch.preserve_format.

    Returns:
        torch.Tensor: Zero-initialized tensor with same shape as input
    """
    iris_instance.debug(
        f"zeros_like: input_shape = {input.shape}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}"
    )

    # Use input's properties as defaults if not specified
    if dtype is None:
        dtype = input.dtype
    if layout is None:
        layout = input.layout
    if device is None:
        device = input.device

    # Validate device compatibility with Iris
    iris_instance._throw_if_invalid_device(device)

    # Get the size from input tensor
    size = input.size()
    num_elements = input.numel()

    # Allocate new tensor with the same size
    new_tensor = iris_instance._allocate(num_elements, dtype)
    new_tensor.zero_()

    # Reshape to match input size
    new_tensor = new_tensor.reshape(size)

    # Apply the requested layout
    new_tensor = iris_instance._apply_layout(new_tensor, layout)

    # Set requires_grad if specified
    if requires_grad:
        new_tensor.requires_grad_()

    return new_tensor
