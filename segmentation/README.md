# DGMN2 for Semantic Segmentation

This folder contains the implementation of DGMN2 for semantic segmentation.

Here, we take [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) as an example, applying DGMN2-Tiny to SETR-Naive, SETR-PUP head and SETR-MLA head.


## Results

#### Cityscapes validation set

|    Method    |   Backbone   | Iters |  mIoU  | mIoU (ms + flip) | Config | Download |
|:-------------|:-------------|:-----:|:------:|:----------------:|:------:|:--------:|
| Semantic FPN | DGMN2-Tiny   |  40K  |  78.09 |      79.40       | [config](configs/fpn_dgmn2_tiny_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1sg6Mlzg12uTIQvE5XEebzNkd-Tino0eU/view?usp=sharing) |
| Semantic FPN | DGMN2-Small  |  40K  |  80.65 |      81.58       | [config](configs/fpn_dgmn2_small_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/19H1ynlczV3dKC5qvTl3l2r-ygp26g5ho/view?usp=sharing) |
| Semantic FPN | DGMN2-Medium |  40K  |  80.60 |      81.79       | [config](configs/fpn_dgmn2_medium_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1IWXPpx6ra7n1svK0yJHD9E4lr7M0lhVU/view?usp=sharing) |
| Semantic FPN | DGMN2-Large  |  40K  |  81.75 |      82.64       | [config](configs/fpn_dgmn2_large_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1gtE2IhkUtsjY6HPajUwlz3pE2szyjNuS/view?usp=sharing) |
| SETR-Naive   | DGMN2-Tiny   |  40K  |  77.23 |      78.23       | [config](configs/setr_naive_dgmn2_tiny_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1RuPaccxYYcpTJ38hpTzUmThL4yRlgT6o/view?usp=sharing) |
| SETR-Naive   | DGMN2-Small  |  40K  |  80.31 |      81.04       | [config](configs/setr_naive_dgmn2_small_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1Vrg25uAGUf0ASeLHgUQCLsMzYWAO-BQO/view?usp=sharing) |
| SETR-Naive   | DGMN2-Medium |  40K  |  80.83 |      81.39       | [config](configs/setr_naive_dgmn2_medium_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1HoGC-t51bhMqLYr35vDkAIHDlVPJQ8Wy/view?usp=sharing) |
| SETR-Naive   | DGMN2-Large  |  40K  |  81.80 |      82.61       | [config](configs/setr_naive_dgmn2_large_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1hYqUr0nF9tKKDhIuJ-heAQwtFCKtPMp9/view?usp=sharing) |
| SETR-PUP     | DGMN2-Tiny   |  40K  |  78.25 |      79.26       | [config](configs/setr_pup_dgmn2_tiny_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1Fz_VgIDvX7WcrEJcdYiB85KWOUGz7qPC/view?usp=sharing) |
| SETR-PUP     | DGMN2-Small  |  40K  |  79.78 |      80.73       | [config](configs/setr_pup_dgmn2_small_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1w-XcAmeTUIAQG1WjYubmTRm88_IMPigP/view?usp=sharing) |
| SETR-PUP     | DGMN2-Medium |  40K  |  80.97 |      81.80       | [config](configs/setr_pup_dgmn2_medium_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1zQM9CUxXDVfGZsXSkoRt1u-Yak3g_9jN/view?usp=sharing) |
| SETR-PUP     | DGMN2-Large  |  40K  |  81.58 |      82.27       | [config](configs/setr_pup_dgmn2_large_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1-ZbdeKIFGGBo73QZjGsIBJTA8vdbeZBL/view?usp=sharing) |
| SETR-MLA     | DGMN2-Tiny   |  40K  |  78.25 |      79.32       | [config](configs/setr_mla_dgmn2_tiny_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1EpgGGDzRzwMslW07ML93fO0VgIKbqFWr/view?usp=sharing) |
| SETR-MLA     | DGMN2-Small  |  40K  |  80.79 |      81.62       | [config](configs/setr_mla_dgmn2_small_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1SxjazS3tTwC43H-t1fs1--I3LpgwH9ty/view?usp=sharing) |
| SETR-MLA     | DGMN2-Medium |  40K  |  81.09 |      82.00       | [config](configs/setr_mla_dgmn2_medium_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1L_VZSB3TjnEUtzz1cXTY2iR5Grsl-yzM/view?usp=sharing) |
| SETR-MLA     | DGMN2-Large  |  40K  |  81.55 |      81.98       | [config](configs/setr_mla_dgmn2_large_4x2_769x769_40k_cityscapes.py) | [model](https://drive.google.com/file/d/1yRz3yjk6Aox-weRZLQL7-buW0G_RqiPX/view?usp=sharing) |


## Getting Started

Clone the repository locally:
```
git clone https://github.com/fudan-zvg/DGMN2
```


### Installation

a. Install MMSegmentation following the [official instructions](https://github.com/open-mmlab/mmsegmentation). Here we use MMSegmentation 0.16.0.

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

First, prepare Cityscapes dataset according to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

Then, download the [weights](https://github.com/fudan-zvg/DGMN2) pretrained on ImageNet, and put them in a folder `pretrained/`.


### Training
To train DGMN2-Tiny + SETR-PUP head on Cityscapes training set on a single node with 4 GPUs for 40K iterations run:

```
dist_train.sh configs/setr_pup_dgmn2_tiny_4x2_769x769_40k_cityscapes.py 4
```


### Evaluation
To evaluate DGMN2-Tiny + SETR-PUP head on Cityscapes validation set on a single node with 4 GPUs run:
```
dist_test.sh configs/setr_pup_dgmn2_tiny_4x2_769x769_40k_cityscapes.py /path/to/checkpoint_file 4 --eval mIoU
```
