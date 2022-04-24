# pytorch-research

研究机器学习、深度学习、计算机视觉所写的 Python 代码，作为学习的记录吧

### conda 环境配置
安装
~~~
conda create -n pytorch python=3.9.10
conda activate pytorch

# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2
~~~
换源，修改配置
~~~
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
show_channel_urls: true
~~~
~~~
conda upgrade --all
~~~
测试
~~~
python
import torch
torch.cuda.is_available()
~~~
### BUG
tensorboard
~~~
pip install tb-nightly
pip install setuptools==59.5.0
~~~
