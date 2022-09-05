# DGMN2 with Sparse R-CNN

This folder contains the implementation of DGMN2 for object detection with [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).


## Results

#### COCO validation set

|   Method   |   Backbone   | Lr schd |   AP   | Config | Download |
|:-----------|:-------------|:-------:|:------:|:------:|:--------:|
| Sparse R-CNN | DGMN2-Small |   3x   |  48.2  | [config](configs/sparsercnn.dgmn2small.300pro.3x.yaml) | [model](https://drive.google.com/file/d/1EvzDIdc_zNPhUlDweP_g9-Iqf2_XXMr6/view?usp=sharing) |


## Getting Started

Clone the repository locally:
```
git clone https://github.com/fudan-zvg/DGMN2
```


### Installation

a. Install Detectron2 following the [official instructions](https://github.com/facebookresearch/detectron2). Here we use Detectron2 0.4.

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

First, prepare COCO dataset according to the guidelines in [Detectron2](https://github.com/facebookresearch/detectron2).

Then, download the [weights](https://github.com/fudan-zvg/DGMN2) pretrained on ImageNet, and put them in a folder `pretrained/`.


### Training
To train DGMN2-Small + Sparse R-CNN using 300 learnable proposals on COCO train2017 on a single node with 8 GPUs for 36 epochs run:

```
python train_net.py --num-gpus 8 --config-file configs/sparsercnn.dgmn2small.300pro.3x.yaml
```


### Evaluation
To evaluate DGMN2-Tiny + RetinaNet on COCO val2017 on a single node with 8 GPUs run:
```
python train_net.py --num-gpus 8 --config-file --config-file configs/sparsercnn.dgmn2small.300pro.3x.yaml --eval-only MODEL.WEIGHTS path/to/checkpoint_file
```
