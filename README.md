
SegMesh: 3D Mesh Segmentation
============

A fast CUDA-accelerated (GPU) method that uses novel mesh convolutions (spherical harmonics) and neural networks (machine learning/NN) for efficient scene segmentation.

### [Paper](https://arxiv.org/abs/2112.01801)

### Based on Mesh Convolution With Continuous Filters for 3-D Surface Parsing (IEEE Transactions on Neural Networks and Learning Systems 2023)

<p align="center">
<img width="900" src=./image/scene_segmentation.png>
</p>

&nbsp;

## Introduction

SegMesh introduces a range of modular operations tailored for 3D triangular mesh segmentation. These include innovative mesh segmentation techniques, GPU-accelerated mesh simplification, and precise mesh pooling/unpooling methods.  

Our approach uses spherical harmonics for creating  segmentation filters. The GPU-accelerated mesh simplification efficiently processes batched meshes, while our pooling operations cater to varying mesh resolutions. SegMesh also encompasses a neural network designed for 3D surface segmentation, exhibiting exceptional performance in shape analysis and scene segmentation on major 3D benchmarks.

[This journal work](https://arxiv.org/abs/2112.01801) is a sigificant extension of [our original work](https://arxiv.org/abs/2103.15076) presented in CVPR 2021.

## 1. Setup
Tested on Ubuntu 22.04.2 LTS, Cuda 11.7.1, Pytorch 2.0.1. To run within a conda environment, and the following hardware: GPU 3090, 4090 for multiple versions of python /pytorch /cuda /cudnn /ubuntu. Requires GPU support for blocks of 1024 threads.
```
conda create --prefix .conda python==3.8 -y
conda activate ./.conda
conda install -c "nvidia/label/cuda-11.7.1" cuda -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python setup.py install
```

## 2. Data Preparation

### Dataset(s) supported by Segmesh

| Dataset Name | Nature of Dataset | Task                         | Number of Samples | Categories/Classes | Evaluation (%) | Terms of Usage |
|--------------|-------------------|------------------------------|-------------------|--------------------|----------------|----------------|
| ShapeNetCore | Synthetic         | Shape Classification         | ~51,000           | 55                 | 87.3           | Academic       |
| SHREC        | Synthetic         | Shape Classification         | 600 (20 samples/class) | 30 | 100            | Commercial     |
| CUBE         | Synthetic         | Shape Classification         | 4,400 (200 samples/class) | 22 | 100          | Commercial     |
| COSEG        | Synthetic         | Semantic Labelling           | Varies per category | 3 (aliens, chairs, vases) | 98.8, 99.5, 95.6 | Commercial |
| HUMAN        | Synthetic         | Semantic Labelling           | 399 (381 training, 18 test) | 8 (body parts) | 91.5            | Commercial    |
| FAUST        | Synthetic         | 3D Manifold Correspondence   | 100 (80 training, 20 test) | 10 subjects with 10 poses each | 100 | Academic    |
| S3DIS        | Real              | Shape Segmentation           | 6 areas (Area 5 for testing) | 13 (indoor areas features) | 91.3(OA), 77.2(mAcc), 71.0(mIoU) | Academic |
| ScanNet      | Real              | Shape Segmentation           | 1,613 (1,213 training, 300 validation) | 40 | 69.2(mIoU)      | Academic     |


### Downloading the Datasets

Datasets can be downloaded following these instructions:

- **SHREC, CUBE, COSEG, HUMAN**: These datasets are available and can be downloaded from the provided links in the ./data directory.
  - Coseg: [Download](https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz)
  - Human: [Download](https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz)
  - Cubes: [Download](https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz)
  - Shrec16: [Download](https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz)

Alternatively, you can extract these shared datasets of MeshCNN using [download_extract_shapes.py](./data/download_extract_shapes.py) script.
```
  cd data
  python download_extract_shapes.py
```

- **ShapeNetCore, FAUST, S3DIS, ScanNet**: These datasets are restricted to academic use. Follow the links to their respective websites for download instructions and adhere to their usage terms.
  - ShapeNetCore: [License & Download](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main)
  - FAUST: [License & Download](https://faust-leaderboard.is.tuebingen.mpg.de/license)
  - S3DIS: [Terms & Access Form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1&pli=1)
  - ScanNet: [License & Download](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)

Please ensure to follow the respective license terms and agreements when using these datasets.

Unzip the datasets after downloading.

After extracting its compressed file, follow the steps in the Preprocess section to prepare the labels for these datasets.

## 3. Usage of Segmesh

### 1. Preprocess

To preprocess datasets for training, run the preprocess script with the specific dataset you want to prepare. Update the main function in              [preprocess.py](./segmesh/datasets/preprocess.py) if needed. Here are the commands for each supported dataset:

```
python -m segmesh.datasets.preprocess --dataset cubes --config_file ./config/cubes.yaml
python -m segmesh.datasets.preprocess --dataset shrec --config_file ./config/shrec.yaml
python -m segmesh.datasets.preprocess --dataset coseg_aliens --config_file ./config/coseg_aliens.yaml
python -m segmesh.datasets.preprocess --dataset coseg_chairs --config_file ./config/coseg_chairs.yaml
python -m segmesh.datasets.preprocess --dataset coseg_vases --config_file ./config/coseg_vases.yaml
python -m segmesh.datasets.preprocess --dataset humanseg --config_file ./config/human.yaml
python -m segmesh.datasets.preprocess --dataset faust --config_file ./config/faust_match.yaml
```

Make sure to update the paths in the config file before you run the command.

### 2. Training

Shape Classification
```
python -m train.train_cubes
python -m train.train_shrec
python -m train.train_shapenetcore
python -m train.train_faust_match
```

Before you start the training for coseg dataset, update the config file path in the train/train_coseg.py to train a specific the coseg dataset (chairs/aliens/vases).

Shape Segmentation
```
python -m train.train_human
python -m train.train_coseg
```

Scene Segmentation
```
python -m train.train_s3dis_render
python -m train.train_scannet_render
```
### 3. Support for custom datasets

To add support for a new or custom dataset, follow the steps:-

1. To add custom support for dataloaders and collate functions, create a Dataset class similar to [shape_dataset.py](./segmesh/datasets/shape_dataset.py) and load in the training script
2. To add custom augmentations and normalize functionality, update the [augmentations.py](./segmesh/utils/augmentations.py) and [normalize.py](./segmesh/utils/normalize.py) with the dataset specific class definitions.
3. Replicate the training code in a new file and update the BaseTrainer and DatasetHandler classes. Load other custom classes.
4. Finally, update the input/outputs and loss functions in train_one_epoch function of [fit.py](./fit.py) if required.

### 4. Inference

SegMesh provides a inference framework for various tasks, including scene segmentation, shape classification, and shape segmentation.

#### Inference Code Structure

The inference code is organized into several Python files, each handling a specific task:

- `inference_scene_seg.py`: Performs scene segmentation on 3D scenes. It processes 3D scene data and outputs segmented results.
- `inference_shape_cls.py`: Conducts shape classification on 3D shapes. This script takes a 3D shape and classifies it into predefined categories.
- `inference_shape_seg.py`: Deals with shape segmentation, where it segments different parts of a given 3D shape.
- `__init__.py`: Initialization file for the inference module.

#### Running Inference

To run the inference for a specific task, use the following command structure:

1. **Scene Segmentation**:
```bash
  python -m inference.inference_scene_seg --config path/to/scene_seg_config.yaml --dataset dataset_name --model_path path/to/model --mesh_path path/to/mesh_file.h5 --label_path path/to/label_file.txt
```
Example command to run inference for a S3DIS dataset test sample:
```bash
python -m inference.inference_scene_seg --config ./config/s3dis.yaml --dataset s3dis --model_path ./runs_scenes/s3dis_render_20230913_184719/model_epoch_50 --mesh_file ./data/S3DIS_3cm_hdf5_Rendered/Area_6/conferenceRoom_1.h5 --label_file ./data/S3DIS_3cm_hdf5_Rendered/Area_6/conferenceRoom_1.txt
```

2. **Shape Classification**:
```bash
python -m inference.inference_shape_cls --config path/to/shape_cls_config.yaml --model_path path/to/model --mesh_file path/to/mesh_file.obj --label_file path/to/label_file.txt
```

3. **Shape Segmentation**:
```bash
python -m inference.inference_shape_seg --config path/to/shape_seg_config.yaml --model_path path/to/model --mesh_file path/to/mesh_file.obj --label_file path/to/label_file.txt
```
Example command to run inference for Human dataset sample:
```bash
python -m inference.inference_shape_seg --dataset human --model_path './runs_shapes/human_20230922_121456/model_epoch_100' --mesh_file './data/human_seg/test/shrec__10.obj' --label_file './data/human_seg/face_label/shrec__10.txt'
```

Replace `path/to/config_file.yaml` with the path to your dataset-specific configuration file, `path/to/model` with the path to your trained model, and `path/to/mesh_file` and `path/to/label_file` with the appropriate paths to your data files.

### 5. Evaluation

SegMesh provides an evaluation code structure, allowing easy integration and evaluation of various 3D datasets like S3DIS and ScanNet. The evaluation structure is designed to be flexible, enabling the addition of new datasets with minimal changes.

#### Evaluation Code Structure

The evaluation code is organized into several key subclass files:

- `base_evaluator.py`: Contains the `BaseEvaluator` class, which provides common functionalities for all datasets.
- `{dataset_name}_evaluator.py`: Dataset-specific evaluator classes (e.g., `s3dis_evaluator.py`, `scannet_evaluator.py`) that inherit from `BaseEvaluator`.
- `transform_texture.py`: Handles dataset-specific data transformations.
- `main.py`: The main script to select and run the appropriate evaluator based on the dataset.

#### Running the Evaluation

To run the evaluation for a specific dataset, run the `main.py` script with the required arguments from the root project directory:

```bash
python -m eval.main --config path/to/config_file.yaml --dataset [s3dis | scannet | ...]
```

Replace `path/to/config_file.yaml` with the path to your dataset-specific configuration file, and `[s3dis | scannet | coseg | ...]` with the desired dataset name.
  
## Citation

If you find our work useful in your research, please consider citing:

```
@article{lei2023mesh,
  title={Mesh Convolution With Continuous Filters for 3-D Surface Parsing},
  author={Lei, Huan and Akhtar, Naveed and Shah, Mubarak and Mian, Ajmal},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
```
@inproceedings{lei2021picasso,
  title={Picasso: A CUDA-based Library for Deep Learning over 3D Meshes},
  author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13854--13864},
  year={2021}
} 
```
Please also cite the original dataset if you use their data or our reprocessed data, and follow their original terms of use.


## Licensing

SegMesh is available for non-commercial internal research use by academic institutions or not-for-profit organisations only, free of charge. Please, see the [license](./license.txt) for further details. To the extent permitted by applicable law, your use is at your own risk and our liability is limited. Interested in a commercial license? For commercial queries, please email <aimlshop@adelaide.edu.au> with subject line "SegMesh Commercial License". 

This is an [AIML Shop](https://aiml.shop) project.
