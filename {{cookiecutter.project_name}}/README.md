# CompressedUNET

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&color=gray)  
![Test Coverage](https://img.shields.io/codecov/c/github/yourusername/pytorch-framework?logo=codecov)

A simple template for building and training deep learning models using PyTorch. This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author :** Brad Niepceron <br />

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
conda env create -f environment.yaml
conda activate torch-env
```

## Experiment tracking

By default this template uses WandB as a logging system and tracker for experiment metrics.
To enable this support you should have a free account at [wandb.ai](https://wandb.ai) and login with the CLI using :

```bash
wandb login
```

The CLI will ask for the API key that can be found in your wandb account page.


## Running tasks

You can run the default experiment with :

```bash
python main.py --config_path ./tasks/mnist/config.yaml
```

### Create a custom task

You can define your own tasks by simply following the structure of the default experiment folder (cifar10).
Alternatively, if the synforge CLI is installed you can use it to create the necessary files for you :

```bash
synforge generate task
```