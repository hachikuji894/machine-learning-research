import glob
import os
import sys
import xml.etree.ElementTree as ET


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


file_root = './lables/'
file_list = os.listdir(file_root)
save_root = './ground-truth/'

# if not os.path.exists("./input"):
#     os.makedirs("./input")
# if not os.path.exists("./input/ground-truth"):
#     os.makedirs("./input/ground-truth")

for file in file_list:
    id = file.split('.')[0]
    with open(save_root + id + ".txt", "w") as new_f:
        root = ET.parse(file_root + id + ".xml").getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
                if difficult != 'Unspecified' and int(difficult) == 1:
                    difficult_flag = True
            obj_name = obj.find('name').text

            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")
