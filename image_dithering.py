import os
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
import random


def image_dithering_simple(img):
    # 读取图像
    input_image = Image.open(os.path.join(r"/EYE/cy/leiqibing/test/normal_moderate", img))

    # 获取图像的尺寸
    width, height = input_image.size

    # 创建一个新的图像对象，用于存储抖动后的图像
    output_image = Image.new("RGB", (width, height))

    # 遍历图像的每个像素点
    for x in range(width):
        for y in range(height):
            # 获取当前像素点的颜色
            r, g, b = input_image.getpixel((x, y))

            # 生成随机的颜色偏移值
            offset = random.randint(-100, 100)

            # 计算新的颜色值
            new_r = min(max(r + offset, 0), 255)
            new_g = min(max(g + offset, 0), 255)
            new_b = min(max(b + offset, 0), 255)

            # 将新的颜色值设置给输出图像的对应像素点
            output_image.putpixel((x, y), (new_r, new_g, new_b))
    new_name = img.split(".jpg")[0] + "simple_dithering" + ".jpg"
    # 保存抖动后的图像
    output_image.save(os.path.join(r"/EYE/cy/leiqibing/test/auged/normal_moderate", new_name))


def floyd_steinberg_dithering_kernel(image):
    for y in range(image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            old_pixel = image[y, x]
            new_pixel = np.round(old_pixel / 255) * 255
            image[y, x] = new_pixel
            error = old_pixel - new_pixel
            image[y, x + 1] += error * 7 / 16
            image[y + 1, x - 1] += error * 3 / 16
            image[y + 1, x] += error * 5 / 16
            image[y + 1, x + 1] += error * 1 / 16
    return image


#################### Atkinson Dithering ####################

def atkinson_dithering_kernel(image):
    error = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0] - 2):
        for x in range(image.shape[1] - 2):
            old_pixel = image[y, x] + error[y, x]
            new_pixel = np.round(old_pixel / 255) * 255
            image[y, x] = new_pixel
            diff = old_pixel - new_pixel
            error[y, x + 1] += diff * 1 / 8
            error[y, x + 2] += diff * 1 / 8
            error[y + 1, x - 1] += diff * 1 / 8
            error[y + 1, x] += diff * 1 / 8
            error[y + 1, x + 1] += diff * 1 / 8
            error[y + 2, x] += diff * 1 / 8
    return image


if __name__ == '__main__':
    results = Parallel(n_jobs=-1)(
        delayed(atkinson_dithering_kernel)(im) for im in os.listdir(r"/EYE/cy/leiqibing/test/normal_moderate"))
