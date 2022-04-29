import os
import shutil

train_path = 'C:/Users/Alberta/Desktop/train_set'
label_path = 'C:/Users/Alberta/Desktop/labels'
src_path = 'C:/Users/Alberta/Desktop/set'
new_path = 'C:/Users/Alberta/Desktop/few_shot'

names = os.listdir(train_path)
num = 100001
names_set = set()
for name in names:
    names_set.add(name.split('.')[0])

for name in names_set:
    shutil.copy(train_path + '/' + name + '.jpg', new_path + '/' + str(num) + '.jpg')
    shutil.copy(train_path + '/' + name + '.xml', new_path + '/' + str(num) + '.xml')
    num += 1

# for name in names:
#     xml = name.replace('.xml', '.jpg', )
#     shutil.copy(src_path + '/' + xml, train_path + '/' + xml)
