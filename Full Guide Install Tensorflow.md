# I. INSTALATION

## 1.Dependencies
`sudo apt-get install htop gcc make cmake git`

## 2.Download and Install [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx)
```
sudo add-apt-repository ppa:graphics-drivers/ppa  
sudo apt-get update
sudo apt-get install nvidia-375
nvidia-smi
reboot
```

## 3.Install docker (Optional)
(Source: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04)
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu zesty stable"
sudo apt-get update
sudo apt-get install docker-ce
```

```
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG
```

```sh
docker ps
docker images
docker start <ID>
docker stop <ID>
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
```

## 4.Install nvidia-docker (Optional)

(Source: http://itproguru.com/expert/2016/10/docker-create-container-change-container-save-as-new-image-and-connect-to-container/)
#### Notice: If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
```sh
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
```

#### Add the package repositories
```sh
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```

#### Install nvidia-docker2 and reload the Docker daemon configuration
```sh
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

#### Test nvidia-smi with the latest official CUDA image
`docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi`

## 5.Install NVIDIA-DIGITS (Optional)

Source: https://github.com/NVIDIA/nvidia-docker/wiki/DIGITS

`docker run --runtime=nvidia --name digits -d -p 5000:5000 -v /opt/mnist:/data/mnist nvidia/digits`

OR

```sh
docker run --runtime=nvidia --name digits -d -p 5000:5000 -p 6006:6006 -v digits-jobs:/jobs nvidia/digits:6.0

docker exec -it <ID> /bin/bash

exit
```

## 6.Download and Install [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)
[Tutorial](https://yangcha.github.io/CUDA90/)
#### Uninstall old version:
```sh
sudo apt-get purge cuda
sudo apt-get purge libcudnn6
sudo apt-get purge libcudnn6-dev
```

#### Install CUDA Toolkit 9.0 and cuDNN 7.0 as follows:
```sh
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo apt-get update
sudo apt-get install cuda=9.0.176-1
sudo apt-get install libcudnn7-dev
sudo apt-get install libnccl-dev
```

#### Set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables, also add them to the end of .bashrc file:
```sh
>export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
>export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## 7.Download [Cudnn](https://developer.nvidia.com/cudnn) then copy to CUDA installed folder

## 8.Install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html):

Or by command:
```sh
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
source ~/.bashrc
```

## 9.Conda install pakages:
```commandline
conda search -c fastai fastai
conda create --name fastaiv1 python=3.6
conda install pytorch=1.0.0 torchvision cudatoolkit=9.0 fastai jupyter -c pytorch -c fastai -c conda-forge
```

```
conda create --name ENV_NAME python=3.6
conda install --name ENV_NAME python=3.6

source activate ENV_NAME

conda install -c conda-forge pip nose h5py ipython jupyter matplotlib numpy pandas pillow scikit-learn scikit-image flask pylint spyder -y
conda install -c conda-forge statsmodels seaborn scrapy sympy requests dill graphviz python-graphviz tpot tqdm scipy cython opencv -y
conda install -c conda-forge bcolz spacy expat plotnine bokeh-y
conda install -c conda-forge theano -y
conda install pytorch -c pytorch # Or: pip install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install torchvision
conda install -c anaconda tensorflow-gpu -y # Or: pip install --upgrade  tensorflow-gpu
conda install -c anaconda keras-gpu -y # Or: pip install Keras
pip install vtk
pip install mayavi
```

Optional:
```
pip install tflearn
conda install -c blaze blaze -y
conda install -c creditx sklearn_pandas
conda install -c mwcraig vpython -y
conda install mingw libpython -y 
conda install -c conda-forge gunicorn -y
conda install -c conda-forge plotly -y
conda install -c anaconda numbapro -y
```

## 10.VERIFY SUCCESSFUL INSTALLATION

```python
import platform
print(platform.python_version())
```

```python
import torch
torch.cuda.get_device_name(0)
```

```python
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

```python
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
```

```python
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session()
print(sess.run(c))
sess.close()
```

```python
# check installed fastai framework
python -m fastai.utils.show_install
```


# II. UTILITIES

## 1.Check and Free GPU
```sh
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>
```

## 2.Install Pycharm
Download Pycharm Professional from JetBrain

start PyCharm from a command-line: 
```
cd /path/to/pycharm/bin
sh pycharm.sh
```

invoke "Tools | Create Desktop Entry"
exit PyCharm
restart a session (*)
launch PyCharm from Dash

## 3.Conda commands:

#### CLONE environment
`conda create --name NEW_ENV --clone OLD_ENV`

#### EXPORT environment
`conda env export | cut -f 1 -d '=' | grep -v "^prefix: " > environment.yml`

Or for pip:

`conda list -e > requirements.txt`

#### CREATE environment from SHARED environment file
`conda env create -f environment.yml`

Or for pip:

`pip install -r requirements.txt`

#### REMOVE environment
`conda remove --name ENV_NAME --all`

#### REMOVE package in environment
`conda remove --name ENV_NAME PACKAGE_NAME`

#### UPDATE in conda
```sh
conda update python
conda update conda
```

References:
 
https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415
 
https://conda.io/docs/user-guide/tasks/manage-environments.html
 
https://medium.com/@vivek.yadav/deep-learning-setup-for-ubuntu-16-04-tensorflow-1-2-keras-opencv3-python3-cuda8-and-cudnn5-1-324438dd46f0
 
Course:

https://medium.com/@Rapchik/running-google-s-deep-learning-course-material-under-windows-82d468b6d5be










