# DGMN2 for Image Classification

This folder contains the implementation of DGMN2 for image classification.


## Results

#### ImageNet-1K

| Method       | Params (M) | FLOPs (G) | Top1 Acc (%) | Download |
|--------------|:----------:|:---------:|:------------:|:--------:|
| DGMN2-Tiny   |    12.1    |    2.3    |     78.7     | [model](https://drive.google.com/file/d/1H21VwFOzkv47GIXnV2a47F2K98wn3s0a/view?usp=sharing) |
| DGMN2-Small  |    21.0    |    4.3    |     81.7     |          |
| DGMN2-Medium |    35.8    |    7.1    |     82.5     |          |
| DGMN2-Large  |    48.3    |   10.4    |     83.3     |          |


## Getting Started

Clone the repository locally:
```
git clone https://github.com/fudan-zvg/DGMN2
```


### Installation

a. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/). Here we use PyTorch 1.8.1 and torchvision 0.9.1.
```
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge
```

b. Install [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models). Here we use PyTorch Image Models 0.4.5.
```
pip install timm==0.4.5
```

c. Build the extension.
```
cd dcn
python setup.py build_ext --inplace
```


### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


### Training
To train DGMN2-Tiny on ImageNet on a single node with 8 GPUs for 300 epochs run:

```
dist_train.sh dgmn2_tiny 8 /path/to/checkpoint_root --data-path /path/to/imagenet
```


### Evaluation
To evaluate a pre-trained DGMN2-Tiny on ImageNet val with a single GPU run:
```
dist_train.sh dgmn2_tiny 1 /path/to/checkpoint_root --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```
