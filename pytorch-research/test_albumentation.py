import albumentations as A
from PIL import Image
import cv2 as cv
import numpy as np


image = cv.imread('../image_set/example.png')
out = A.Sharpen(p = 1)(image=image)["image"]
cv.imshow("da", out)
cv.waitKey(0)