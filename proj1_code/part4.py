from typing import Tuple
import numpy as np
import numpy.fft as fft
from utils import load_image, save_image

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两幅图像的峰值信噪比 PSNR，支持单通道或多通道。
    参数：
      img1, img2: 浮点数组，取值范围 [0,1]，形状 (h,w) 或 (h,w,c)
    返回：
      PSNR 值（单位 dB）
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def lowpass_mask(shape: Tuple[int, int], ratio: float) -> np.ndarray:
    """
    构造低通滤波掩模，仅保留中心低频区域。
    参数：
      shape: 图像的高度和宽度 (h, w)
      ratio: 保留中心低频的比例 (0~1)
    返回：
      掩模数组，形状 (h, w)，中心区域为 1，其余为 0
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * ratio / 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    mask = (dist <= radius).astype(np.float32)
    return mask


def _compress_channel(channel: np.ndarray, retention: float) -> np.ndarray:
    """
    对单通道数据的频域低通压缩与重建。
    """
    # 正向 FFT 并中心化
    F = fft.fft2(channel)
    F_shift = fft.fftshift(F)
    # 构造并应用掩模
    mask = lowpass_mask(channel.shape, retention)
    F_filtered = F_shift * mask
    # 逆中心化并逆变换
    F_ishift = fft.ifftshift(F_filtered)
    recon = fft.ifft2(F_ishift)
    # 取实部并裁剪
    recon = np.real(recon)
    return np.clip(recon, 0, 1)


def compress_image_fft(img: np.ndarray, retention: float) -> np.ndarray:
    """
    对图像在频域做低通压缩，并重建，支持多通道。
    输入 img: 浮点数组，取值范围 [0,1]，形状 (h,w) 或 (h,w,c)
    返回: 重建图像，同形状、同范围。
    """
    if img.ndim == 3:
        channels = [_compress_channel(img[..., c], retention) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return _compress_channel(img, retention)


def main():
    # 输入与输出路径（BMP 格式）
    input_path = '../data/1a_dog.bmp'
    output_prefix = 'lena_fft_retention'

    # 直接使用 load_image，返回 [0,1] 浮点图像
    img = load_image(input_path)  # shape (h,w,c) or (h,w)

    retention_ratios = [0.1, 0.3, 0.5, 0.7]
    psnr_values = {}

    for r in retention_ratios:
        recon = compress_image_fft(img, r)
        val = psnr(img, recon)
        psnr_values[r] = val
        print(f"保留比例 {r:.1f} ，PSNR = {val:.2f} dB")
        # save_image 接受 [0,1] 浮点，无需额外转换
        save_image(f"../results/part4/{output_prefix}_{int(r*100)}.bmp", recon)

    print("PSNR 汇总:")
    for r, v in psnr_values.items():
        print(f"  保留 {r:.1f} -> PSNR: {v:.2f} dB")

if __name__ == '__main__':
    main()
