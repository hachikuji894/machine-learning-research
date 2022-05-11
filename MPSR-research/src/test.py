from predictor import COCODemo
import cv2
import os
import time

from maskrcnn_benchmark.config import cfg


time_start = time.time()

config_file = "configs/conf_final.yaml"

# update the configs options with the configs file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

file_root = './images/'
file_list = os.listdir(file_root)
save_root = './images_check/'
time_end = time.time()
print('load model totally cost', time_end - time_start)

for img_name in file_list:
    img_path = file_root + img_name
    image = cv2.imread(img_path)
    time_start = time.time()
    predictions = coco_demo.run_on_opencv_image(image)
    save_path = save_root + img_name
    cv2.imwrite(save_path, predictions)
    time_end = time.time()
    print('run on an image totally cost', time_end - time_start)

    # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    # cv2.imshow("img",predictions)
    # cv2.waitKey(1)
