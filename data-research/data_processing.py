# Terminal: 2022-03-29 18:05:28,695 maskrcnn_benchmark.trainer INFO: eta: 8:01:37  iter: 20
# loss: 1.4023 (1.7081)  loss_classifier: 0.0574 (0.2738)  loss_box_reg: 0.0000 (0.0033)  loss_closeup: 0.5914 (0.7208)
# loss_objectness: 0.6733 (0.6699)  loss_rpn_box_reg: 0.0205 (0.0403)  time: 0.6934 (0.8032)  data: 0.0093 (0.0669)
# lr: 0.001800  max mem: 5694


# 正则表达式提取
import re
import matplotlib.pyplot as plt

# path = './log/log.txt'
path = './log/log_finetuning.txt'
# type = 'base'
type = 'noval'

iter = []
loss = []
lr = []

f = open(path, "r")
lines = f.readlines()
for line in lines:
    if re.match(r'(?=.*\bmaskrcnn_benchmark.trainer INFO\b)', line) is None:
        continue
    for one in re.findall(r'iter: ([\d.]+)', line):
        iter.append(int(one))
    for one in re.findall(r'loss: ([\d.]+)', line):
        loss.append(float(one))
    for one in re.findall(r'lr: ([\d.]+)', line):
        lr.append(float(one))

print(iter)
print(loss)
print(lr)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.xlabel(u'iters')
plt.ylabel(u'loss')
plt.title(u'Loss for models in ' + type + ' training')
plt.plot(iter, loss, 'g-')
# plt.show()

plt.subplot(1, 2, 2)
plt.xlabel(u'iters')
plt.ylabel(u'learning rate')
plt.title(u'Learning rate for models in ' + type + ' training')
plt.plot(iter, lr, 'g-')

plt.show()
