# {{cookiecutter.project_name}}

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&color=gray)  
![Test Coverage](https://img.shields.io/codecov/c/github/yourusername/pytorch-framework?logo=codecov)

A simple template for building and training deep learning models using PyTorch. This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author :** {{ cookiecutter.author_name }} <br />

## Overview

This repository contains a framework built on top of [PyTorch](https://pytorch.org/) and inspired by the deep learning framework from the Institute of Machine Learning in Biomedical Imaging : https://github.com/compai-lab/iml-dl. It abstracts away boilerplate code for training, evaluation, and inference tasks while still offering the flexibility of PyTorch for custom modifications. 

It supports the following:
- Model training pipelines (with data preprocessing)
- Experiment tracking (integration with Weights & Biases)
- Model evaluation 

## Installation

First, the dependencies should be installed using pip or the provided conda environment : 

```bash
pip install -r requirements.txt
```

Or

```bash
conda env create -f env.yaml
conda activate {{cookiecutter.project_name}}
```

## Experiment tracking

By default this template uses WandB as a logging system and tracker for experiment metrics.
To enable this support you should have a free account at [wandb.ai](https://wandb.ai) and login with the CLI using :

```bash
wandb login
```

The CLI will ask for the API key that can be found in your wandb account page.


## Running experiments

You can run the default experiment with :

```bash
python main.py --config_path ./experiments/mnist/config.yaml
```

### Create a custom experiment

You can define your own experiments by simply following the structure of the default experiment folder (cifar10).
Alternatively, if the synforge CLI is installed you can use it to create the necessary files for you :

```bash
synforge generate experiment new_experiment
```

## TODO

- [ ] Save model
- [ ] Change wandb output log dir
- [ ] Add metrics tracking with WandB
- [ ] Add hyperparameter tuning (Optuna, RayTune ...)
- [ ] Add more default architectures (ResNet, UNet ...)
- [ ] Add fine-tuning utilities
- [ ] Add tensorboard support to replace W&B
