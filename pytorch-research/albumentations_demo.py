import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt

path = 'C:/Users/Alberta/Desktop/set'

names = os.listdir(path)
print(names)

transform = [
    # 水平反转 色调饱和度值
    A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1),
        A.HorizontalFlip(p=1),
    ]),
    # 随机旋转 噪声
    A.Compose([
        A.Rotate(limit=90, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None,
                 always_apply=False, p=1),
        A.GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=1)
    ]),
    # 上下翻转 随机亮度
    A.Compose([
        A.VerticalFlip(p=1),
        A.RandomBrightnessContrast(always_apply=False, p=1)
    ]),
    # 随机亮度 随机对比度 模糊
    A.Compose([
        A.Blur(blur_limit=9, always_apply=False, p=1),
        A.RandomBrightnessContrast(always_apply=False, p=1)
    ]),
    # 模拟图像雾
    A.Compose([
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=1)
    ]),
]

# print(path + '/' + names[1])
# image = cv2.imread(path + '/' + names[0])
# cv2.imshow("da", transform[3](image=image)["image"])
# cv2.waitKey(0)

for name in names:
    image = cv2.imread(path + '/' + name)
    num = 0
    for t in transform:
        cv2.imwrite(path + '/tr' + str(num) + name, t(image=image)["image"])
        num += 1
