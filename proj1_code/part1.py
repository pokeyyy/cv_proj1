#!/usr/bin/python3

from typing import Tuple

import numpy as np


def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.
    
    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1
    
    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution
    
    Returns:
        kernel: 1d column vector of shape (k,1)
    
    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """
    # 输入验证：确保 ksize 和 sigma 是有效的
    if ksize <= 0:
        raise ValueError("ksize 必须是一个正整数")
    if sigma <= 0:
        raise ValueError("sigma 必须大于 0")
    
    # 计算均值位置：核的中心
    mu = ksize // 2  # 使用整数除法取整，确保 mu 是整数
    
    # 生成位置数组：从 0 到 ksize-1，表示每个元素的位置
    x = np.arange(ksize)
    
    # 计算高斯值：使用高斯分布的指数部分
    gaussian_values = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    # 归一化：使核的元素之和为 1
    kernel_sum = np.sum(gaussian_values)  # 计算高斯值的总和
    kernel = gaussian_values / kernel_sum  # 除以总和进行归一化
    
    # 重塑为列向量：将一维数组变为 (ksize, 1) 的形状
    kernel = kernel.reshape((ksize, 1))
    
    return kernel

def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each 
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability 
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    ############################
# 计算核的尺寸 k，例如 cutoff_frequency=1 时，k=5
    k = cutoff_frequency * 4 + 1
    
    # 生成一维高斯核（列向量），中心在 floor(k / 2)，和为 1
    kernel_1d = create_Gaussian_kernel_1D(ksize=k, sigma=cutoff_frequency)
    
    # 通过一维核与其转置的外积生成二维高斯核
    kernel = np.outer(kernel_1d, kernel_1d.T)

    return kernel


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.
    
    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    # 获取输入图像尺寸和滤波器尺寸
    m, n, c = image.shape  # 图像高度、宽度、通道数
    k, j = filter.shape     # 滤波器高度、宽度

    # 计算零填充量（保持输出尺寸不变）
    pad_h = (k - 1) // 2  # 垂直填充量
    pad_w = (j - 1) // 2  # 水平填充量

    # 对图像进行零填充（所有通道同时处理）
    # 填充格式：((上, 下), (左, 右), (通道前, 通道后))，此处通道不填充
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    # 生成滑动窗口视图（形状为 (m, n, k, j, c)）
    # 在高度和宽度维度滑动，窗口尺寸为(k, j)，保留通道维度
    strides = padded_image.strides  # 获取填充后图像的步幅
    window_strides = (strides[0], strides[1], strides[0], strides[1], strides[2])
    shape = (m, n, k, j, c)
    image_blocks = np.lib.stride_tricks.as_strided(
        padded_image, 
        shape=shape, 
        strides=window_strides
    )

    # 调整滤波器形状以匹配广播 (k, j) -> (k, j, 1)
    filter_reshaped = filter.reshape(*filter.shape, 1)

    # 执行卷积：窗口与滤波器相乘后求和（高度、宽度维度求和）
    # 结果形状 (m, n, c) -> 直接作为输出
    filtered_image = np.sum(image_blocks * filter_reshaped, axis=(-3, -2))

    return filtered_image
  
  
def my_conv2d_numpy_v2(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image. Notably, this is the optimized revision of `my_conv2d_numpy()`.
    
    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    # 获取输入图像的尺寸 (高度, 宽度, 通道数)
    m, n, c = image.shape
    # 获取卷积核尺寸 (高度, 宽度)
    k, j = filter.shape

    # 计算填充量，保证卷积输出与输入尺寸一致
    pad_h = (k - 1) // 2  # 垂直方向的填充大小
    pad_w = (j - 1) // 2  # 水平方向的填充大小

    # 使用边缘复制进行填充，而非零填充，能够减少黑色边框带来的明显影响
    # 'edge' 模式会复制边缘的像素值，有助于平滑过渡
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

    # 采用 stride_tricks 构造滑动窗口，返回形状为 (m, n, k, j, c) 的视图
    strides = padded_image.strides  # 获取填充图像的步幅信息
    # 构造滑动窗口步幅，最后一个维度保持通道信息
    window_strides = (strides[0], strides[1], strides[0], strides[1], strides[2])
    # 定义滑动窗口的形状
    shape = (m, n, k, j, c)
    image_blocks = np.lib.stride_tricks.as_strided(
        padded_image, 
        shape=shape, 
        strides=window_strides
    )

    # 调整卷积核的形状以便于与图像窗口进行逐元素乘法，扩展为 (k, j, 1)
    filter_reshaped = filter.reshape(k, j, 1)
    
    # 进行元素乘法和求和操作：
    # 对每个窗口将对应位置与卷积核相乘，并在 k 和 j 维度求和，得到每个通道的卷积结果
    filtered_image = np.sum(image_blocks * filter_reshaped, axis=(-3, -2))

    return filtered_image


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: array of dim (m, n, c)
        image2: array of dim (m, n, c)
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    # 对 image1 进行卷积，得到低频部分
    low_frequencies = my_conv2d_numpy(image1, filter)

    # 对 image2 进行低通滤波，以便从 image2 提取高频部分
    image2_low = my_conv2d_numpy(image2, filter)
    # image2 的高频部分通过原始 image2 减去其低频部分得到
    high_frequencies = image2 - image2_low

    # 混合图像：将 image1 的低频和 image2 的高频相加
    hybrid_image = low_frequencies + high_frequencies

    # 将混合图像中所有像素的值限制在 [0, 1] 范围内，避免出现溢出或负值
    hybrid_image = np.clip(hybrid_image, 0, 1)

    return low_frequencies, high_frequencies, hybrid_image
