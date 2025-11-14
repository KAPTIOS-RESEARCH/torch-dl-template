# {{cookiecutter.project_name}}

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&color=gray)  

A simple template for building and training deep learning models using PyTorch. This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author :** {{cookiecutter.author_name}} <br />

## Overview

This repository contains a framework built on top of [PyTorch](https://pytorch.org/) and inspired by the deep learning framework from the Institute of Machine Learning in Biomedical Imaging : https://github.com/compai-lab/iml-dl. It abstracts away boilerplate code for training, evaluation, and inference tasks while still offering the flexibility of PyTorch for custom modifications. 

It supports the following:
- Model training pipelines (with data preprocessing)
- Experiment tracking (integration with Weights & Biases)
- Model evaluation 
- Energy consumption tracking

## Installation

First, the dependencies should be installed the provided conda environment depending on your OS : 

```bash
conda env create -f environment.yml
conda activate torch-env
```

## Features

### Running a training task

You can run an task by pointing to its configuration file like :

```bash
python main.py --config_path ./tasks/mnist/train.yaml
```

### Export a saved model

The framework supports exporting a saved PyTorch model to ONNX.
To do so, an export config yaml file should be given as flag to the ```export.py``` script.

```bash
python export.py --export_config_path ./tasks/default/export.yaml
```

This file should look like : 

```yaml
export_path: './exports'
model_path: './saved_models/best_model.pth'
quantization_dataset:
  module_name: src.data.sets.super_resolution
  class_name: FastMRISuperResolutionDataReader
  parameters:
    data_folder: /path/to/dataset
    num_samples: 100
model:
  class_name: SRResUNet
  module_name: src.models.super_resolution
  parameters:
```

By default, the export also saves a quantized version of the model. For this to work, a Calibration Dataset should be passed using the ```quantization_dataset``` key.

## Tracking


### Experiment tracking

This template uses [Comet](https://www.comet.com) as a logging system and tracker for experiment metrics.
To enable this support you should have an account and set the api key and workspace in a .env file (see ```.env.example```).

### Energy consumption tracking

This template uses [CodeCarbon](https://github.com/mlco2/codecarbon) for tracking energy consumption and carbon emissions during the training tasks.
We used the ```EmissionsTracker``` explicit object with a default configuration in the ```./src/core/experiment.py``` file : 

```bash
self.codeCarbonTracker = EmissionsTracker(
    experiment_id=self.id,
    experiment_name=self.name,
    output_dir=self.log_dir,
    output_file='emissions.csv',
    log_level='error',
    measure_power_secs=10,
    save_to_file=True,
)
```

In this configuration, metrics are not sent to the CodeCarbon API but saved in a .csv file.
This configuration can be overwritten by a ```.codecarbon.config``` file, you can refer to there documentation to set this file.
The tracking process also automatically saves summary metrics in the Comet ML experiment. The ```emissions.csv``` file can be found in the ```Assets&Artefacts``` Comet resource.

For each experiment, two tasks are also tracked and saved in the ```Others``` Comet resource :

- data (to track the energy consumption when loading the datasets)
- training (to track the energy consumtion when training/validating/testing a model)
