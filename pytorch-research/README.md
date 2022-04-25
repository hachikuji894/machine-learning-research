# pytorch-research

### conda 环境配置
安装
~~~
conda create -n env_name python=3.6
conda activate env_name
conda deactivate env_name
conda remove -n env_name --all

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
~~~
换源，修改配置 /.condarc
~~~
channels:
  - defaults
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
show_channel_urls: True
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
