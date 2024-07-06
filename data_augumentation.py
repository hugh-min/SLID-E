from PIL import ImageEnhance
import os
import numpy as np
import tqdm
from PIL import Image


def brightnessEnhancement(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.4 + 0.4 * np.random.random()  # 取值范围1.1-1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.1 + 0.4 * np.random.random()  # 取值范围1.1-1.5
    # contrast = 2.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.choice([-1, -2, -3, 1, 2, 3]) * 90
    if random_angle == 0:
        rotation_img = img.rotate(-90)  # 旋转角度
    else:
        rotation_img = img.rotate(random_angle)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


def flip(root_path, img_name):  # 翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def createImage(imageDir, saveDir):
    for idx, name in tqdm.tqdm(enumerate(os.listdir(imageDir))):
        saveName2 = "flip" + str(name)
        saveImage2 = flip(imageDir, name)
        saveImage2.save(os.path.join(saveDir, saveName2))


if __name__ == '__main__':
    imageDir = r"/EYE/cy/leiqibing/test/auged/normal_moderate"  # 要改变的图片的路径文件夹
    saveDir = r"/EYE/cy/leiqibing/test/auged/normal_moderate"  # 数据增强生成图片的路径文件夹
    import time

    start_time = time.time()
    createImage(imageDir, saveDir)
    end_time = time.time()
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print(run_time)
