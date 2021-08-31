## Neural Scene Flow Prior

Submitted to Thirty-fifth Conference on Neural Information Processing Systems (2021).


### Prerequisites
This code is based on PyTorch implementation, and tested on torch=1.6.0 with CUDA 10.1 **OR** torch=1.7.1 with CUDA 10.2. 

For a detailed installation guide, please go to ```requirements.txt```.


### Dataset
We provide four datasets we used in our paper.
You may download datasets used in the paper from these anonymous links:

| KITTI (266MB) |
|:-:|
| https://drive.google.com/file/d/1PUuOycduXKbla3bPN1a2XL5ZTt27Y1Or/view?usp=sharing |

| Argoverse (5.3GB) |
|:-:|
| https://drive.google.com/file/d/1q5twc14tyRqk63npSzfkxmcjejkKHRNK/view?usp=sharing |

| nuScenes (436MB) |
|:-:|
| https://drive.google.com/file/d/1xNXIfwlhlMkTqDpVYy4BZDXVqW-Dpe6u/view?usp=sharing |

| FlyingThings3D (436MB) |
|:-:|
| https://drive.google.com/file/d/15s6TN1ucKSH2fKW098ddIlHyMPuY-qZ1/view?usp=sharing |

<!-- | KITTI (266MB) | Argoverse (5.3GB) |
|:-:|:-:|
| https://drive.google.com/file/d/1PUuOycduXKbla3bPN1a2XL5ZTt27Y1Or/view?usp=sharing | https://drive.google.com/file/d/1q5twc14tyRqk63npSzfkxmcjejkKHRNK/view?usp=sharing |

| nuScenes (436MB) | FlyingThings3D (436MB) |
|:-:|:-:|
| https://drive.google.com/file/d/1xNXIfwlhlMkTqDpVYy4BZDXVqW-Dpe6u/view?usp=sharing | https://drive.google.com/file/d/15s6TN1ucKSH2fKW098ddIlHyMPuY-qZ1/view?usp=sharing | -->
<!-- 
| KITTI (266MB) | Argoverse (5.3GB) | nuScenes (436MB) | FlyingThings3D (436MB) |
|:-:|:-:|:-:|:-:|
| https://drive.google.com/file/d/1PUuOycduXKbla3bPN1a2XL5ZTt27Y1Or/view?usp=sharing | https://drive.google.com/file/d/1q5twc14tyRqk63npSzfkxmcjejkKHRNK/view?usp=sharing | https://drive.google.com/file/d/1xNXIfwlhlMkTqDpVYy4BZDXVqW-Dpe6u/view?usp=sharing | https://drive.google.com/file/d/15s6TN1ucKSH2fKW098ddIlHyMPuY-qZ1/view?usp=sharing | -->

After you download the dataset, you can create a symbolic link in the ./dataset folder as ```./dataset/kitti```, ```./dataset/argoverse```, ```./dataset/nuscenes```, and ```./dataset/flyingthings```.


### Optimization
Since we use neural scene flow prior for runtime optimization, our method does not include any training. 

Just run following lines for a simple optimization on a small KITTI Scene Flow dataset (only 50 testing samples)
```
python optimization.py \
--dataset KITTISceneFlowDataset \
--dataset_path dataset/kitti \
--exp_name KITTI_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--num_points 2048 \
--hidden_units 128 \
--lr 0.008 \
--backward_flow \
--early_patience 70 \
--visualize
```

You can then play with these configurations.
We provide commands we used to generate results in the small point coud (2048 points) experiments and large point cloud (all points included) experiments.

#### 1. small point cloud (2048 points)

#### KITTI Scene Flow
```
python optimization.py \
--dataset KITTISceneFlowDataset \
--dataset_path dataset/kitti \
--exp_name KITTI_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--num_points 2048 \
--hidden_units 128 \
--lr 0.008 \
--backward_flow \
--early_patience 70 \
--visualize
```

#### Argoverse Scene Flow
```
python optimization.py \
--dataset ArgoverseSceneFlowDataset \
--dataset_path dataset/argoverse \
--exp_name Argoverse_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--num_points 2048 \
--hidden_units 128 \
--lr 0.008 \
--backward_flow \
--early_patience 30 \
--visualize
```

#### nuScenes Scene Flow
```
python optimization.py \
--dataset NuScenesSceneFlowDataset \
--dataset_path dataset/nuscenes \
--exp_name Argoverse_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--num_points 2048 \
--hidden_units 128 \
--lr 0.008 \
--backward_flow \
--early_patience 30 \
--visualize
```

#### FlyingThings3D
```
python optimization.py \
--dataset FlyingThings3D \
--dataset_path dataset/flyingthings \
--exp_name FlyingThings_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--num_points 2048 \
--hidden_units 128 \
--lr 0.008 \
--backward_flow \
--early_patience 30 \
--visualize
```

#### 2. dense point cloud (all points included)

#### KITTI Scene Flow
```
python optimization.py \
--dataset KITTISceneFlowDataset \
--dataset_path dataset/kitti \
--exp_name KITTI_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--use_all_points \
--hidden_units 128 \
--lr 0.001 \
--early_patience 100 \
--visualize
```

#### Argoverse Scene Flow
```
python optimization.py \
--dataset ArgoverseSceneFlowDataset \
--dataset_path dataset/argoverse \
--exp_name Argoverse_2048_points \
--batch_size 1 \
--iters 5000 \
--compute_metrics \
--use_all_points \
--hidden_units 128 \
--lr 0.003 \
--backward_flow \
--early_patience 100 \
--visualize
```
