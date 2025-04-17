#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d_pytorch(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies input filter(s) to the input image.

    Args:
        image: Tensor of shape (1, d1, h1, w1)
        kernel: Tensor of shape (N, d1/groups, k, k) to be applied to the image
    Returns:
        filtered_image: Tensor of shape (1, d2, h2, w2) where
           d2 = N
           h2 = (h1 - k + 2 * padding) / stride + 1
           w2 = (w1 - k + 2 * padding) / stride + 1

    HINTS:
    - You should use the 2d convolution operator from torch.nn.functional.
    - In PyTorch, d1 is `in_channels`, and d2 is `out_channels`
    - Make sure to pad the image appropriately (it's a parameter to the
      convolution function you should use here!).
    - You can assume the number of groups is equal to the number of input channels.
    - You can assume only square filters for this function.
    """

    # 获取输入通道数 d1
    in_channels = image.shape[1]  # d1

    # 卷积核大小 k
    kernel_size = kernel.shape[-1]  # k

    # 为了输出与输入保持相同的空间大小，padding 设为 k // 2
    padding = kernel_size // 2

    # 调用 PyTorch 的 2D 卷积函数
    # groups=in_channels 表示按通道分组卷积，每个输入通道独立卷积
    # bias=None 表示不加偏置，stride=1 表示步幅为1
    filtered_image = F.conv2d(
        input=image,
        weight=kernel,
        bias=None,
        stride=1,
        padding=padding,
        groups=in_channels
    )

    return filtered_image
