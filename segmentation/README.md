# DGMN2 for Semantic Segmentation

This folder contains the implementation of DGMN2 for semantic segmentation.

Here, we take [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) as an example, applying DGMN2-Tiny to SETR-Naive, SETR-PUP head and SETR-MLA head.


## Results

#### Cityscapes validation set

| Method       |   Backbone   |  Iters  |  mIoU   | mIoU (ms + flip) |
|--------------|:------------:|:-------:|:-------:|:----------------:|
| Semantic FPN | DGMN2-Tiny   |   40K   |  78.09  |      79.40       |
| Semantic FPN | DGMN2-Small  |   40K   |  80.65  |      81.58       |
| Semantic FPN | DGMN2-Medium |   40K   |  80.60  |      81.79       |
| Semantic FPN | DGMN2-Large  |   40K   |  81.75  |      82.64       |
| SETR-Naive   | DGMN2-Tiny   |   40K   |  77.23  |      78.23       |
| SETR-Naive   | DGMN2-Small  |   40K   |  80.31  |      81.04       |
| SETR-Naive   | DGMN2-Medium |   40K   |  80.83  |      81.39       |
| SETR-Naive   | DGMN2-Large  |   40K   |  81.80  |      82.61       |
| SETR-PUP     | DGMN2-Tiny   |   40K   |  78.25  |      79.26       |
| SETR-PUP     | DGMN2-Small  |   40K   |  79.78  |      80.73       |
| SETR-PUP     | DGMN2-Medium |   40K   |  80.96  |      81.80       |
| SETR-PUP     | DGMN2-Large  |   40K   |  81.58  |      82.27       |
| SETR-MLA     | DGMN2-Tiny   |   40K   |  78.25  |      79.32       |
| SETR-MLA     | DGMN2-Small  |   40K   |  80.79  |      81.62       |
| SETR-MLA     | DGMN2-Medium |   40K   |  81.09  |      82.00       |
| SETR-MLA     | DGMN2-Large  |   40K   |  81.55  |      81.98       |


## Getting Started

Clone the repository locally:
```
git clone https://github.com/fudan-zvg/DGMN2
```


### Installation

a. Install MMSegmentation following the [official instructions](https://github.com/open-mmlab/mmsegmentation). Here we use MMDetection 0.17.0.

b. Install [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models). Here we use PyTorch Image Models 0.4.5.
```
pip install timm==0.4.5
```

c. Build the extension
```
cd dcn
python setup.py build_ext --inplace
```


### Data preparation

First, prepare Cityscapes dataset according to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

Then, download the [weights](https://github.com/fudan-zvg/DGMN2) pretrained on ImageNet, and put them in a folder `pretrained/`


### Training
To train DGMN2-Tiny + SETR-PUP head on Cityscapes training set on a single node with 8 gpus for 40K iterations run:

```
dist_train.sh configs/setr_pup_dgmn2_tiny_40k_cityscapes.py 8
```


### Evaluation
To evaluate DGMN2-Tiny + SETR-PUP head on Cityscapes validation set on a single node with 8 gpus run:
```
dist_test.sh configs/setr_pup_dgmn2_tiny_40k_cityscapes.py /path/to/checkpoint_file 8 --eval mIoU
```
