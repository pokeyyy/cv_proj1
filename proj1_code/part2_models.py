#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proj1_code.part1 import create_Gaussian_kernel_2D


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff frequency.

        PyTorch requires the kernel to be of a particular shape in order to
        apply it to an image. Specifically, the kernel needs to be of shape
        (c, 1, k, k) where c is the # channels in the image. Start by getting a
        2D Gaussian kernel using your implementation from Part 1, which will be
        of shape (k, k). Then, let's say you have an RGB image, you will need to
        turn this into a Tensor of shape (3, 1, k, k) by stacking the Gaussian
        kernel 3 times.

        Args
            cutoff_frequency: int specifying cutoff_frequency
        Returns
            kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel_2D() function from part1.py in
          this function.
        - Since the # channels may differ across each image in the dataset,
          make sure you don't hardcode the dimensions you reshape the kernel
          to. There is a variable defined in this class to give you channel
          information.
        - You can use np.reshape() to change the dimensions of a numpy array.
        - You can use np.tile() to repeat a numpy array along specified axes.
        - You can use torch.Tensor() to convert numpy arrays to torch Tensors.
        """

        # 创建二维高斯核，形状为 (k, k)
        kernel_2d = create_Gaussian_kernel_2D(cutoff_frequency)
        k = kernel_2d.shape[0]  # 核的大小 k

        # 首先将 (k, k) 扩展为 (1, 1, k, k)
        kernel = np.reshape(kernel_2d, (1, 1, k, k))
        # 然后沿通道维度重复 c 次，得到 (c, 1, k, k)
        kernel = np.tile(kernel, (self.n_channels, 1, 1, 1))
        # 转换为 torch 张量
        kernel = torch.Tensor(kernel)

        return kernel

    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image.

        Args:
            x: Tensor of shape (b, c, m, n) where b is batch size
            kernel: low pass filter to be applied to the image
        Returns:
            filtered_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the
          filter will be applied to.
        """

       # 计算填充大小，保持输出尺寸与输入相同
        k = kernel.shape[2]  # 核的大小 k
        padding = k // 2  # 填充大小

        # 应用 2D 卷积
        filtered_image = F.conv2d(
            x,
            kernel,
            padding=padding,
            groups=self.n_channels  # 按通道分组 groups 参数针对的是 shape 的 第 2 个维度（索引为 1）
        )

        return filtered_image

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the
        hybrid image.

        Args:
            image1: Tensor of shape (b, c, m, n)
            image2: Tensor of shape (b, c, m, n)
            cutoff_frequency: Tensor of shape (b)
        Returns:
            low_frequencies: Tensor of shape (b, c, m, n)
            high_frequencies: Tensor of shape (b, c, m, n)
            hybrid_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function
          in this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You
          can use torch.clamp().
        - If you want to use images with different dimensions, you should
          resize them in the HybridImageDataset class using
          torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        batch_size = image1.shape[0]       # 获取批量大小

        low_frequencies_list = []
        high_frequencies_list = []
        hybrid_image_list = []

        for i in range(batch_size):
            # 获取当前图像对和对应的截止频率
            img1 = image1[i].unsqueeze(0)  # 形状 (1, c, m, n)
            img2 = image2[i].unsqueeze(0)  # 形状 (1, c, m, n)
            cf = cutoff_frequency[i].item()
            
            # 生成高斯核
            kernel = self.get_kernel(cf)

            # 对 img1 应用低通滤波
            low_freq = self.low_pass(img1, kernel)

            # 对 img2 应用低通滤波并计算高频内容
            low_freq_img2 = self.low_pass(img2, kernel)
            high_freq = img2 - low_freq_img2

            # 计算混合图像
            hybrid = low_freq + high_freq

            # 将结果添加到列表中
            low_frequencies_list.append(low_freq)
            high_frequencies_list.append(high_freq)
            hybrid_image_list.append(hybrid)

        # 将列表中的张量堆叠成批次张量
        low_frequencies = torch.cat(low_frequencies_list, dim=0)
        high_frequencies = torch.cat(high_frequencies_list, dim=0)
        hybrid_image = torch.cat(hybrid_image_list, dim=0)

        # 将像素值裁剪到 [0, 1] 范围内
        low_frequencies = torch.clamp(low_frequencies, 0, 1)
        #high_frequencies = torch.clamp(high_frequencies, 0, 1)
        hybrid_image = torch.clamp(hybrid_image, 0, 1)

        return low_frequencies, high_frequencies, hybrid_image