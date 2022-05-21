# DGMN2 for Object Detection

This folder contains the implementation of DGMN2 for object detection.

Here, we take [MMDetection](https://github.com/open-mmlab/mmdetection) as an example, applying DGMN2 to RetinaNet and Mask R-CNN.


## Results

#### COCO validation set

| Method           |   Backbone   | Lr schd | box AP | mask AP | Config | Download |
|------------------|:------------:|:-------:|:------:|:-------:|:------:|:--------:|
| RetinaNet        | DGMN2-Tiny   |   1x    |  39.7  |    -    | [config](configs/retinanet_dgmn2_tiny_fpn_8x2_1x_coco.py) | [model](https://drive.google.com/file/d/14gjw75Cz8iytUFDQP9ioIfiMni6e-xRl/view?usp=sharing) |
| RetinaNet        | DGMN2-Small  |   1x    |  42.5  |    -    | [config](configs/retinanet_dgmn2_small_fpn_8x2_1x_coco.py) |          |
| RetinaNet        | DGMN2-Medium |   1x    |  43.7  |    -    | [config](configs/retinanet_dgmn2_medium_fpn_8x2_1x_coco.py) |          |
| RetinaNet        | DGMN2-Large  |   1x    |  44.7  |    -    | [config](configs/retinanet_dgmn2_large_fpn_8x2_1x_coco.py) |          |
| Mask R-CNN       | DGMN2-Tiny   |   1x    |  40.1  |  37.2   | [config](configs/mask_rcnn_dgmn2_tiny_fpn_8x2_1x_coco.py) | [model](https://drive.google.com/file/d/17vGTzN1dazQ1Euu5mpBMvEro5HafAedT/view?usp=sharing) |
| Mask R-CNN       | DGMN2-Small  |   1x    |  43.4  |  39.7   | [config](configs/mask_rcnn_dgmn2_small_fpn_8x2_1x_coco.py) |          |
| Mask R-CNN       | DGMN2-Medium |   1x    |  44.4  |  40.2   | [config](configs/mask_rcnn_dgmn2_medium_fpn_8x2_1x_coco.py) |          |
| Mask R-CNN       | DGMN2-Large  |   1x    |  46.2  |  41.6   | [config](configs/mask_rcnn_dgmn2_large_fpn_8x2_1x_coco.py) |          |


## Getting Started

Clone the repository locally:
```
git clone https://github.com/fudan-zvg/DGMN2
```


### Installation

a. Install MMDetection following the [official instructions](https://github.com/open-mmlab/mmdetection). Here we use MMDetection 2.12.0.

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

First, prepare COCO dataset according to the guidelines in [MMDetection](https://github.com/open-mmlab/mmdetection).

Then, download the [weights](https://github.com/fudan-zvg/DGMN2) pretrained on ImageNet, and put them in a folder `pretrained/`


### Training
To train DGMN2-Tiny + RetinaNet on COCO train2017 on a single node with 8 GPUs for 12 epochs run:

```
dist_train.sh configs/retinanet_dgmn2_tiny_fpn_8x2_1x_coco.py 8
```


### Evaluation
To evaluate DGMN2-Tiny + RetinaNet on COCO val2017 on a single node with 8 GPUs run:
```
dist_test.sh configs/retinanet_dgmn2_tiny_fpn_8x2_1x_coco.py /path/to/checkpoint_file 8 --eval bbox
```
