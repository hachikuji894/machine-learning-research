from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import cv2
import os
import time

time_start = time.time()

config_file = "configs/conf_final.yaml"

# update the configs options with the configs file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.6,
)

file_root = './video/'
file_list = os.listdir(file_root)
save_root = './video_check/'
time_end = time.time()
print('totally cost', time_end - time_start)

for video_name in file_list:
    src_path = file_root + video_name
    save_path = save_root + video_name
    capture = cv2.VideoCapture(src_path)
    # image = cv2.imread(img_path)
    # (grabbed, frame) = cap.read()
    # if not grabbed:
    #     break
    fps = capture.get(cv2.CAP_PROP_FPS)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # predictions = coco_demo.run_on_opencv_image(image)
    # writer.write()

    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            img_out = coco_demo.run_on_opencv_image(frame)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'-------------------------------------------')
            writer.write(img_out)
    else:
        print('视频打开失败！')
    writer.release()
