# DGMN2

This repository contains the implementation of [Dynamic Graph Message Passing Networks](https://arxiv.org/abs/1908.06955).


## Main results
### Image Classification

#### ImageNet-1K

|    Method    | Params (M) | FLOPs (G) | Top1 Acc (%) |
|:-------------|:----------:|:---------:|:------------:|
| DGMN2-Tiny   |    12.1    |    2.3    |     78.7     |
| DGMN2-Small  |    21.0    |    4.3    |     81.7     |
| DGMN2-Medium |    35.8    |    7.1    |     82.5     |
| DGMN2-Large  |    48.3    |   10.4    |     83.3     |


### Object Detection

#### COCO validation set

|      Method      |   Backbone   | Lr schd | box AP | mask AP |
|:-----------------|:-------------|:-------:|:------:|:-------:|
| RetinaNet        | DGMN2-Tiny   |   1x    |  39.7  |    -    |
| RetinaNet        | DGMN2-Small  |   1x    |  42.5  |    -    |
| RetinaNet        | DGMN2-Medium |   1x    |  43.7  |    -    |
| RetinaNet        | DGMN2-Large  |   1x    |  44.7  |    -    |
| Mask R-CNN       | DGMN2-Tiny   |   1x    |  40.2  |  37.2   |
| Mask R-CNN       | DGMN2-Small  |   1x    |  43.4  |  39.7   |
| Mask R-CNN       | DGMN2-Medium |   1x    |  44.4  |  40.2   |
| Mask R-CNN       | DGMN2-Large  |   1x    |  46.2  |  41.6   |
| Deformable DETR  | DGMN2-Tiny   |   50e   |  44.4  |    -    |
| Deformable DETR  | DGMN2-Small  |   50e   |  47.3  |    -    |
| Deformable DETR  | DGMN2-Medium |   50e   |  48.4  |    -    |
| Deformable DETR+ | DGMN2-Small  |   50e   |  47.3  |    -    |
| Sparse R-CNN     | DGMN2-Small  |   3x    |  48.2  |    -    |


### Semantic Segmentation

#### Cityscapes validation set

|    Method    |   Backbone   | Iters |  mIoU  | mIoU (ms + flip) |
|:-------------|:-------------|:-----:|:------:|:----------------:|
| Semantic FPN | DGMN2-Tiny   |  40K  |  78.09 |      79.40       |
| Semantic FPN | DGMN2-Small  |  40K  |  80.65 |      81.58       |
| Semantic FPN | DGMN2-Medium |  40K  |  80.60 |      81.79       |
| Semantic FPN | DGMN2-Large  |  40K  |  81.75 |      82.64       |
| SETR-Naive   | DGMN2-Tiny   |  40K  |  77.23 |      78.23       |
| SETR-Naive   | DGMN2-Small  |  40K  |  80.31 |      81.04       |
| SETR-Naive   | DGMN2-Medium |  40K  |  80.83 |      81.39       |
| SETR-Naive   | DGMN2-Large  |  40K  |  81.80 |      82.61       |
| SETR-PUP     | DGMN2-Tiny   |  40K  |  78.25 |      79.26       |
| SETR-PUP     | DGMN2-Small  |  40K  |  79.78 |      80.73       |
| SETR-PUP     | DGMN2-Medium |  40K  |  80.96 |      81.80       |
| SETR-PUP     | DGMN2-Large  |  40K  |  81.58 |      82.27       |
| SETR-MLA     | DGMN2-Tiny   |  40K  |  78.25 |      79.32       |
| SETR-MLA     | DGMN2-Small  |  40K  |  80.79 |      81.62       |
| SETR-MLA     | DGMN2-Medium |  40K  |  81.09 |      82.00       |
| SETR-MLA     | DGMN2-Large  |  40K  |  81.55 |      81.98       |


## Getting Started

 - For image classification, please see [classification](classification/).
 - For object detection, please see [detection](detection/).
 - For semantic segmentation, please see [segmentation](segmentation/).



## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Reference

```
@inproceedings{zhang2020dynamic,
  title={Dynamic Graph Message Passing Networks},
  author={Zhang, Li and Xu, Dan and Arnab, Anurag and Torr, Philip H.S.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
```
@article{zhang2022dynamic,
  title={Dynamic Graph Message Passing Networks},
  author={Zhang, Li and Chen, Mohan and Arnab, Anurag and Xue, Xiangyang and Torr, Philip H.S.},
  journal={arXiv preprint arXiv:1908.06955},
  year={2022}
}
```


## Acknowledgement
Thanks to previous open-sourced repo:  
[PVT](https://github.com/whai362/PVT)  
[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)  
[MMDetection](https://github.com/open-mmlab/mmdetection)  
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)  
