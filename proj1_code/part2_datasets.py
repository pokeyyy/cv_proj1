#!/usr/bin/python3

"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args:
        path: string specifying the directory containing images
    Returns:
        images_a: list of strings specifying the paths to the images in set A,
           in lexicographically-sorted order
        images_b: list of strings specifying the paths to the images in set B,
           in lexicographically-sorted order
    """

    image_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith('.bmp')
        ]
    
    # 按文件名排序
    image_files.sort()
    
    # 分割为两个集合：偶数索引为A，奇数索引为B
    images_a = image_files[::2]   # 步长为2取元素，如0,2,4...
    images_b = image_files[1::2]  # 步长为2取元素，如1,3,5...
    
    # 确保两个集合长度相同
    assert len(images_a) == len(images_b), "A和B集合图像数量必须相等"
    return images_a, images_b


def get_cutoff_frequencies(path: str) -> List[int]:
    """
    Gets the cutoff frequencies corresponding to each pair of images.

    The cutoff frequencies are the values you discovered from experimenting in
    part 1.

    Args:
        path: string specifying the path to the .txt file with cutoff frequency
        values
    Returns:
        cutoff_frequencies: numpy array of ints. The array should have the same
            length as the number of image pairs in the dataset
    """

    # 读取所有cutoff_frequency值
    with open(path, 'r') as f:
        cutoff_frequencies = [int(line.strip()) for line in f]
    return cutoff_frequencies


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You
        can specify additional transforms (e.g. image resizing) if you want to,
        but it's not necessary for the images we provide you since each pair has
        the same dimensions.

        Args:
            image_dir: string specifying the directory containing images
            cf_file: string specifying the path to the .txt file with cutoff
            frequency values
        """
        images_a, images_b = make_dataset(image_dir)
        cutoff_frequencies = get_cutoff_frequencies(cf_file)
        # 设置变换，将PIL图像转换为torch张量，并将像素值归一化到[0,1]
        self.transform = transforms.ToTensor()

        # 保存图像路径和截止频率
        self.images_a = images_a
        self.images_b = images_b
        self.cutoff_frequencies = cutoff_frequencies

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        return len(self.images_a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff frequency value at
        index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0
        and 1. Make sure you transpose the dimensions so that image_a and
        image_b are of shape (c, m, n) instead of the typical (m, n, c), and
        convert them to torch Tensors.

        Args:
            idx: int specifying the index at which data should be retrieved
        Returns:
            image_a: Tensor of shape (c, m, n)
            image_b: Tensor of shape (c, m, n)
            cutoff_frequency: int specifying the cutoff frequency corresponding
               to (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """

        # 获取图像路径
        image_a_path = self.images_a[idx]
        image_b_path = self.images_b[idx]

        # 使用PIL打开图像
        image_a = PIL.Image.open(image_a_path)
        image_b = PIL.Image.open(image_b_path)

        # 应用变换，将PIL图像转换为torch张量
        # ToTensor()会自动将(h, w, c)转换为(c, h, w)，并将像素值从[0,255]归一化到[0,1]
        image_a = self.transform(image_a)
        image_b = self.transform(image_b)

        # 获取对应的截止频率
        cutoff_frequency = self.cutoff_frequencies[idx]

        return image_a, image_b, cutoff_frequency
