# DGMN2 with Deformable DETR

This folder contains the implementation of DGMN2 for object detection with [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).


## Results

#### COCO validation set

|   Method   |   Backbone   | Lr schd |   AP   | Download |
|:-----------|:-------------|:-------:|:------:|:--------:|
| Deformable DETR  | DGMN2-Tiny   |   50e   |  44.4  | [model](https://drive.google.com/file/d/1FZ9HIzQ9ty3TUeW30TCLxI9zKopVcHHG/view?usp=sharing) |
| Deformable DETR  | DGMN2-Small  |   50e   |  47.3  | [model](https://drive.google.com/file/d/1cy9-HftCSkX5gSMH22ThjJNn8Br8EPke/view?usp=sharing) |
| Deformable DETR  | DGMN2-Medium |   50e   |  48.4  | [model](https://drive.google.com/file/d/1bXNPKpWSAu_i0vtQCcBcoP7yUJQUIHN3/view?usp=sharing) |
| Deformable DETR+ | DGMN2-Small  |   50e   |  48.5  | [model](https://drive.google.com/file/d/1svE9XGe6bwvGtiauQenYIjpnaOkQ0x9e/view?usp=sharing) |

Note: "+" indicates using iterative bounding box refinement and two-stage in Deformable DETR.


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

d. Install other requirements.
```bash
pip install -r requirements.txt
```

e. Compile CUDA operators in Deformable DETR.
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```


### Data preparation

First, prepare [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
data/
    coco/
        train2017/
        val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json
```

Then, download the [weights](https://github.com/fudan-zvg/DGMN2) pretrained on ImageNet, and put them in a folder `pretrained/`.


### Training
To train DGMN2-Tiny + Deformable DETR on COCO train2017 on a single node with 8 GPUs run:

```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/dgmn2_deformable_detr.sh --backbone dgmn2_tiny
```


### Evaluation
To evaluate DGMN2-Tiny + Deformable DETR on COCO val2017 on a single node with 8 GPUs run:
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/dgmn2_deformable_detr.sh --backbone dgmn2_tiny --resume <path to checkpoint_file> --eval
```
