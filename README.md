# dcaPF-torch
A PyTorch prototype for Difference-of-Convex Privacy Funnel

Accepted for ISIT 2024 Learn to compress workshop

[Paper link](https://arxiv.org/abs/2403.04778)

## Overview
The source code implements a differnce-of-convex solver for the privacy funnel
The algorithm is implemented with a neural network. It is a supervised learning algorithm.

## Requirements
- python == 3.7.12
- PyTorch == 1.11.0
- torchvision == 0.12.0
- umap-learn == 0.5.3
- pandas == 1.3.5
- scikit-learn == 1.0.2
- scipy == 1.7.3
- matplotlib == 3.5.3
- cvxpy == 1.5.2

## Create a Conda Environment
```
conda env create -f environment.yml
```

## Run the MNIST example
```
python train_rt.py mnist
```

## Run the Fashion-MNIST example
```
python train_rt.py fashion
```

## Load a trained model
```
python load_test.py path/to/model.pth
```


## Developer
Teng-Hui Huang


Electrical and Computer Engineering


University of Sydney, NSW, Australia


email: tenghui[DOT]huang@sydney[DOT]edu[DOT]au

