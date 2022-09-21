# DGMN2

This repository contains the implementation of [Dynamic Graph Message Passing Networks for Visual Recognition](https://arxiv.org/abs/2209.09760).


## Main results
### Image Classification

#### ImageNet-1K

|    Method    | Params (M) | FLOPs (G) | Top1 Acc (%) | Download |
|:-------------|:----------:|:---------:|:------------:|:--------:|
| DGMN2-Tiny   |    12.1    |    2.3    |     78.7     | [model](https://drive.google.com/file/d/1H21VwFOzkv47GIXnV2a47F2K98wn3s0a/view?usp=sharing) |
| DGMN2-Small  |    21.0    |    4.3    |     81.7     | [model](https://drive.google.com/file/d/1bOhpFnZLO8Va4LJccrlnazD1aL61wm5b/view?usp=sharing) |
| DGMN2-Medium |    35.8    |    7.1    |     82.5     | [model](https://drive.google.com/file/d/13iDrUlynBs83pdhUoFmwQoZbAf4oDvTm/view?usp=sharing) |
| DGMN2-Large  |    48.3    |   10.4    |     83.3     | [model](https://drive.google.com/file/d/1nXvXHXJZpsScPnI2VQe8ZgrGXuMpbHia/view?usp=sharing) |


### Object Detection

#### COCO validation set

|      Method      |   Backbone   | Lr schd | box AP | mask AP | Download |
|:-----------------|:-------------|:-------:|:------:|:-------:|:--------:|
| RetinaNet        | DGMN2-Tiny   |   1x    |  39.7  |    -    | [model](https://drive.google.com/file/d/14gjw75Cz8iytUFDQP9ioIfiMni6e-xRl/view?usp=sharing) |
| RetinaNet        | DGMN2-Small  |   1x    |  42.5  |    -    | [model](https://drive.google.com/file/d/1JIIuf7iNA9-tJoUefUNc1O1laJSW30Hx/view?usp=sharing) |
| RetinaNet        | DGMN2-Medium |   1x    |  43.7  |    -    | [model](https://drive.google.com/file/d/1WU4Kv1Z0Q4b3VMIcJPi7LSmUSYqlywWU/view?usp=sharing) |
| RetinaNet        | DGMN2-Large  |   1x    |  44.7  |    -    | [model](https://drive.google.com/file/d/1kws1Q6Ccwipaimour9F67EAg2bMzXlAj/view?usp=sharing) |
| Mask R-CNN       | DGMN2-Tiny   |   1x    |  40.2  |  37.2   | [model](https://drive.google.com/file/d/17vGTzN1dazQ1Euu5mpBMvEro5HafAedT/view?usp=sharing) |
| Mask R-CNN       | DGMN2-Small  |   1x    |  43.4  |  39.7   | [model](https://drive.google.com/file/d/1g1lp7kUIM5gvxROfTAvI2EVjpPL-HW6r/view?usp=sharing) |
| Mask R-CNN       | DGMN2-Medium |   1x    |  44.4  |  40.2   | [model](https://drive.google.com/file/d/1MO0BLtIrRohAW7BPEnr5G63mCBDA4Yey/view?usp=sharing) |
| Mask R-CNN       | DGMN2-Large  |   1x    |  46.2  |  41.6   | [model](https://drive.google.com/file/d/1DFkSQmfHI9z6IKag21BzRb7LhWvHotXm/view?usp=sharing) |
| Deformable DETR  | DGMN2-Tiny   |   50e   |  44.4  |    -    | [model](https://drive.google.com/file/d/1FZ9HIzQ9ty3TUeW30TCLxI9zKopVcHHG/view?usp=sharing) |
| Deformable DETR  | DGMN2-Small  |   50e   |  47.3  |    -    | [model](https://drive.google.com/file/d/1cy9-HftCSkX5gSMH22ThjJNn8Br8EPke/view?usp=sharing) |
| Deformable DETR  | DGMN2-Medium |   50e   |  48.4  |    -    | [model](https://drive.google.com/file/d/1bXNPKpWSAu_i0vtQCcBcoP7yUJQUIHN3/view?usp=sharing) |
| Deformable DETR+ | DGMN2-Small  |   50e   |  48.5  |    -    | [model](https://drive.google.com/file/d/1svE9XGe6bwvGtiauQenYIjpnaOkQ0x9e/view?usp=sharing) |
| Sparse R-CNN     | DGMN2-Small  |   3x    |  48.2  |    -    | [model](https://drive.google.com/file/d/1EvzDIdc_zNPhUlDweP_g9-Iqf2_XXMr6/view?usp=sharing) |


### Semantic Segmentation

#### Cityscapes validation set

|    Method    |   Backbone   | Iters |  mIoU  | mIoU (ms + flip) | Download |
|:-------------|:-------------|:-----:|:------:|:----------------:|:--------:|
| Semantic FPN | DGMN2-Tiny   |  40K  |  78.09 |      79.40       | [model](https://drive.google.com/file/d/1sg6Mlzg12uTIQvE5XEebzNkd-Tino0eU/view?usp=sharing) |
| Semantic FPN | DGMN2-Small  |  40K  |  80.65 |      81.58       | [model](https://drive.google.com/file/d/19H1ynlczV3dKC5qvTl3l2r-ygp26g5ho/view?usp=sharing) |
| Semantic FPN | DGMN2-Medium |  40K  |  80.60 |      81.79       | [model](https://drive.google.com/file/d/1IWXPpx6ra7n1svK0yJHD9E4lr7M0lhVU/view?usp=sharing) |
| Semantic FPN | DGMN2-Large  |  40K  |  81.75 |      82.64       | [model](https://drive.google.com/file/d/1gtE2IhkUtsjY6HPajUwlz3pE2szyjNuS/view?usp=sharing) |
| SETR-Naive   | DGMN2-Tiny   |  40K  |  77.23 |      78.23       | [model](https://drive.google.com/file/d/1RuPaccxYYcpTJ38hpTzUmThL4yRlgT6o/view?usp=sharing) |
| SETR-Naive   | DGMN2-Small  |  40K  |  80.31 |      81.04       | [model](https://drive.google.com/file/d/1Vrg25uAGUf0ASeLHgUQCLsMzYWAO-BQO/view?usp=sharing) |
| SETR-Naive   | DGMN2-Medium |  40K  |  80.83 |      81.39       | [model](https://drive.google.com/file/d/1HoGC-t51bhMqLYr35vDkAIHDlVPJQ8Wy/view?usp=sharing) |
| SETR-Naive   | DGMN2-Large  |  40K  |  81.80 |      82.61       | [model](https://drive.google.com/file/d/1hYqUr0nF9tKKDhIuJ-heAQwtFCKtPMp9/view?usp=sharing) |
| SETR-PUP     | DGMN2-Tiny   |  40K  |  78.25 |      79.26       | [model](https://drive.google.com/file/d/1Fz_VgIDvX7WcrEJcdYiB85KWOUGz7qPC/view?usp=sharing) |
| SETR-PUP     | DGMN2-Small  |  40K  |  79.78 |      80.73       | [model](https://drive.google.com/file/d/1w-XcAmeTUIAQG1WjYubmTRm88_IMPigP/view?usp=sharing) |
| SETR-PUP     | DGMN2-Medium |  40K  |  80.97 |      81.80       | [model](https://drive.google.com/file/d/1zQM9CUxXDVfGZsXSkoRt1u-Yak3g_9jN/view?usp=sharing) |
| SETR-PUP     | DGMN2-Large  |  40K  |  81.58 |      82.27       | [model](https://drive.google.com/file/d/1-ZbdeKIFGGBo73QZjGsIBJTA8vdbeZBL/view?usp=sharing) |
| SETR-MLA     | DGMN2-Tiny   |  40K  |  78.25 |      79.32       | [model](https://drive.google.com/file/d/1EpgGGDzRzwMslW07ML93fO0VgIKbqFWr/view?usp=sharing) |
| SETR-MLA     | DGMN2-Small  |  40K  |  80.79 |      81.62       | [model](https://drive.google.com/file/d/1SxjazS3tTwC43H-t1fs1--I3LpgwH9ty/view?usp=sharing) |
| SETR-MLA     | DGMN2-Medium |  40K  |  81.09 |      82.00       | [model](https://drive.google.com/file/d/1L_VZSB3TjnEUtzz1cXTY2iR5Grsl-yzM/view?usp=sharing) |
| SETR-MLA     | DGMN2-Large  |  40K  |  81.55 |      81.98       | [model](https://drive.google.com/file/d/1yRz3yjk6Aox-weRZLQL7-buW0G_RqiPX/view?usp=sharing) |


## Getting Started

 - For image classification, please see [classification](classification/).
 - For object detection, please see [detection](detection/).
 - For semantic segmentation, please see [segmentation](segmentation/).
 - For Deformable DETR, please see [deformable_detr](deformable_detr/).
 - For Sparse R-CNN, please see [sparsercnn](sparsercnn/).


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
  title={Dynamic Graph Message Passing Networks for Visual Recognition},
  author={Zhang, Li and Chen, Mohan and Arnab, Anurag and Xue, Xiangyang and Torr, Philip H.S.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```


## Acknowledgement
Thanks to previous open-sourced repo:  
[PVT](https://github.com/whai362/PVT)  
[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)  
[MMDetection](https://github.com/open-mmlab/mmdetection)  
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)  
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)  
[Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN)  
[Detectron2](https://github.com/facebookresearch/detectron2)  
