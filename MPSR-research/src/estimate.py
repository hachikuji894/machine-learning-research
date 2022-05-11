# 打印推理的文件
import cv2
import os
import time
import torch
from predictor import COCODemo


from maskrcnn_benchmark.config import cfg

time_start = time.time()

config_file = "configs/conf_final.yaml"

# update the configs options with the configs file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

file_root = './set/'
file_list = os.listdir(file_root)
save_root = './detection_result/'
time_end = time.time()
print('totally cost', time_end - time_start)

for img_name in file_list:

    img_path = file_root + img_name
    image = cv2.imread(img_path)
    predictions = coco_demo.compute_prediction(image)
    top_predictions = coco_demo.select_top_predictions(predictions)
    # get labels and boxes
    labels = top_predictions.get_field("labels")
    scores = top_predictions.get_field("scores")
    boxes = top_predictions.bbox

    f = open(save_root + img_name.split('.')[0] + ".txt", "w")

    for box, label, score in zip(boxes, labels, scores):
        # name = COCODemo.CATEGORIES[label.to(torch.int64)]
        # score = score.to(torch.float64)
        box_list = box.to(torch.int64)[:].tolist()
        f.write("%s %s %s %s %s %s\n" % (
            COCODemo.CATEGORIES[label.to(torch.int64)], str(score.to(torch.float32).item())[:6],
            str(box_list[0]), str(box_list[1]),
            str(box_list[2]), str(box_list[3])))
    f.close()
