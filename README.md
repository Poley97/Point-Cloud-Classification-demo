# PointNet and PointNet++ cls on ModelNet40  

This repo is  a unofficial implementation for PointNet and PointNet++ classification on ModelNet40

## Install
The latest codes are tested on CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification (ModelNet40)

### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)

Split ModelNet40 into train set, validation set and test set. 

Default split ratio (**train set : validation set : test set = 0.8 : 0.1 : 0.1**)
```shell
python ./dataset/data_split.py --dataset_path <ModelNet40 path> --split_ratio <split_ratio>
```
### Run
You can run PointNet or PointNet++ on ModelNet40 cls task.
 
* use `--model pointnet++` or `--model pointnet`



### Performance
Our results are got by 30 epochs training.

| Model | Accuracy |
|--|--|
| PointNet (Official without T-Net) |  87.1|
| PointNet2 (Official) | 91.9 |
| PointNet (Pytorch without normal and T-Net) |  88.6|
| PointNet2_MSG (Pytorch without normal) |  91.9|


## Reference By
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>

## Environments
see requirement.txt


