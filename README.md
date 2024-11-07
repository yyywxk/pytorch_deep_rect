<h2 align="center">pytorch_deep_rect</h2>
<p align="center">
    <!-- <a href="https://github.com/yyywxk/pytorch_deep_rect/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/yyywxk/pytorch_deep_rect/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/yyywxk/pytorch_deep_rect">
    </a>
    <a href="https://github.com/yyywxk/pytorch_deep_rect/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/yyywxk/pytorch_deep_rect">
    </a>
    <a href="https://github.com/yyywxk/pytorch_deep_rect/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/yyywxk/pytorch_deep_rect?color=pink">
    </a>
    <a href="https://github.com/yyywxk/pytorch_deep_rect">
        <img alt="issues" src="https://img.shields.io/github/stars/yyywxk/pytorch_deep_rect">
    </a>
    <a href="mailto: qiulinwei@buaa.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

## Introduction

This is the a PyTorch implementation of papers

- [Deep Rectangling for Image stitching: A Learning Baseline](https://arxiv.org/abs/2203.03831) CVPR 2022 (Oral)
- [RecRecNet: Rectangling Rectified Wide-Angle Images by Thin-Plate Spline Model and DoF-based Curriculum Learning](https://arxiv.org/abs/2301.01661) ICCV2023

## Requirements

- Packages
  
  The code was tested with Anaconda and Python 3.10.13. The Anaconda environment is:
  
  - pytorch = 2.1.1
  - torchvision = 0.16.1
  - cudatoolkit = 11.8
  - tensorboard = 2.17.0
  - tensorboardX = 2.6.2.2
  - opencv-python = 4.9.0.80
  - numpy = 1.26.4
  - pillow = 10.3.0

Install dependencies:

- For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.
- For custom dependencies:
  
  ```bash
  conda install tensorboard tensorboardx
  pip install tqdm opencv-python thop scikit-image lpips scipy
  ```
- We implement this work with Ubuntu 18.04, NVIDIA Tesla V100, and CUDA11.8.

## Datasets

- Put data in `../dataset` folder or  configure your dataset path in the `my_path` function of  `dataloaders/__inint__.py`.
- The details of the dataset AIRD can be found in our paper ([IEEE Xplore](https://ieeexplore.ieee.org/document/10632108)). You can download it at [Baidu Cloud](https://pan.baidu.com/s/1oklVqzmjfluqJdwq1R_xlw?pwd=1234) (Extraction code: 1234).
- These codes also support the DIR-D (Deep Rectangling for Image stitching: A Learning Baseline ([paper](https://arxiv.org/abs/2203.03831))). You can download it at [Google Drive](https://drive.google.com/file/d/1KR5DtekPJin3bmQPlTGP4wbM1zFR80ak/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1aNpHwT8JIAfX_0GtsxsWyQ)(Extraction code: 1234).

## Model Training

Follow steps below to train your model:

1. Input arguments: (see full input arguments via `python train.py --help` or `python train_RecRec.py --help`):
2. To train `deep_rect` using DIR-D with one GPU:
   
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py --lr 1e-4 --dataset DIR-D --epochs 70 --batch_size 4 --workers 4 --loss-type 8terms --GRID_W 8 --GRID_H 6
   ```
3. To train `RecRecNet` using DIR-D with one GPU:
   
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_RecRec.py --lr 1e-4 --dataset DIR-D --epochs 260 --batch_size 4 --workers 4 --GRID_W 8 --GRID_H 6
   ```
4. You can change the dataset from DIR-D  to AIRD.

## Model Testing

1. Input arguments: (see full input arguments via `python test.py --help` or `python test_RecRec.py --help`):
2. Run the  `deep_rect`  testing script.
   
   ```bash
   python test.py --model_path {path/to/your/checkpoint} --save_path {path/to/the/save/result}
   ```
3. Run the  `RecRecNet`  testing script.
   
   ```bash
   python test_RecRec.py --model_path {path/to/your/checkpoint} --save_path {path/to/the/save/result}
   ```

## Inference

1. Input arguments: (see full input arguments via `python inference.py --help`):
2. You can use this script to obtain your own results.
   
   ```bash
   python inference.py --model_path {path/to/your/checkpoint} --save_path {path/to/the/save/result} --input_path {path/to/the/input/data}
   ```
3. Make sure to put the data files as the following structure:
   
   ```
   inference
   ├── input
   |   ├── 001.png
   │   ├── 002.png
   │   ├── 003.png
   │   ├── 004.png
   │   ├── ...
   |
   ├── mask
   |   ├── 001.png
   │   ├── 002.png
   │   ├── 003.png
   │   ├── 004.png
   |   ├── ...
   ```

## Citation

If our work is useful for your research, please consider citing:

```tex
@misc{qiu2024pytorch_deep_rect,
  author = {Qiu, Linwei},
  title = {pytorch_deep_rect},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yyywxk/pytorch_deep_rect}}
}
```

## Questions

Please contact [qiulinwei@buaa.edu.cn](mailto:qiulinwei@buaa.edu.cn).

## Acknowledgement

[UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching)

[UDIS2](https://github.com/nie-lang/UDIS2)

[DeepRectangling](https://github.com/nie-lang/DeepRectangling)

[RecRecNet](https://github.com/KangLiao929/RecRecNet)

[IWKFormer](https://github.com/yyywxk/IWKFormer)

