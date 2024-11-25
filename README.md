# Neural Network Parameter Diffusion



## Environment
We support all versions of `pytorch>=2.0.0`.
But we recommend to use `python==3.11` and `pytorch==2.5.1`, which we have fully tested.
```shell
conda create -n pdiff python=3.11
conda activate pdiff
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
git clone -b develop https://github.com/NUS-HPC-AI-Lab/Neural-Network-Parameter-Diffusion.git --depth=1
cd Neural-Network-Parameter-Diffusion
pip install -r requirements.txt
```
<font color="red">delete "-b develop" when publishing.</font>  


## Quick Start

This will run three steps sequentially: preparing the dataset, training p-diff, and evaluating.
Then the results will be saved in the root directory and save checkpoint in `./checkpoint`
```shell
cd workspace
bash run_all.sh main cifar100_resnet18 0
# bash run_all <category> <tag> <device>
```

## Detailed Usage

Prepare checkpoints dataset.
```shell
cd ./dataset/main/cifar100_resnet18
rm performance.cache  # optional
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=0 python finetune.py
```
Train pdiff and generate models.
```shell
cd ../../../workspace
bash launch.sh main cifar100_resnet18 0
# bash launch <category> <tag> <device>
CUDA_VISIBLE_DEVICES=0 python generate.py main cifar100_resnet18
# CUDA_VISIBLE_DEVICES=<device> python generate.py <category> <tag>
```
Test original checkpoints and generated checkpoints and their similarity.
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py main cifar100_resnet18
# CUDA_VISIBLE_DEVICES=<device> python evaluate.py <category> <tag>
```

All our `<category>` and `<tag>` can be found in `./dataset/<category>/<tag>`.


## Register Your Own Dataset

1. Create a directory that mimics the dataset folder and contains three contents:  
```shell
mkdir ./dataset/main/<tag>
cd ./dataset/main/<tag>
```
`checkpoint`: A directory contains many `.pth` files, which contain dictionaries of parameters.  
`generated`: An empty directory, where the generated model will be stored.  
`test.py`: A test script to test the checkpoints. It should be callable as follows:  
```shell
CUDA_VISIBLE_DEVICES=0 python test.py ./checkpoint/checkpoint001.pth
# CUDA_VISIBLE_DEVICES=<device> python test.py <checkpoint_file>
```

2. Register a dataset.  
Add a class to the last line of the dataset file.
```shell
cd ../../../dataset
vim __init__.py  
# This __init__.py is the dataset file.
```
```diff
# on line 392
+ class <Tag>(MainDataset): pass
```

3. Create your launch script.  
You can change other hyperparameters here.
```shell
cd ../workspace/main
cp cifar10_resnet18.py main_<tag>.py
vim main_<tag>.py
```
```diff
# on line 33
- from dataset import Cifar100_ResNet18 as Dataset
+ from dataset import <Tag> as Dataset
```

3. Train pdiff and generate models.  
Following Section "Detail Usage".  


4. Test original ckpt and generated ckpt and their similarity.  
Following Section "Detail Usage".  

