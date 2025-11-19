# {{cookiecutter.project_name}}

[](https://pytorch.org/)
[](https://www.comet.com)
[](https://codecarbon.io/)
[](https://opensource.org/licenses/MIT)

A simple template for building and training deep learning models using PyTorch. This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author:** {{cookiecutter.author_name}}

-----

## 📖 Overview

This repository contains a framework built on top of [PyTorch](https://www.google.com/search?q=httpss://pytorch.org/) that abstracts away boilerplate code for training, evaluation, and inference. It is inspired by the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: [compai-lab/iml-dl](https://github.com/compai-lab/iml-dl).

### ✨ Key Features

  * **Config-Driven:** Define your entire experiment—from data loading to model parameters—in a single `.yaml` file.
  * **Training Pipelines:** Robust training, validation, and testing loops right out of the box.
  * **📈 Experiment Tracking:** Automatic logging of metrics, parameters, and artifacts with [Comet ML](https://www.comet.com).
  * **🌍 Energy Tracking:** Monitor energy consumption and CO2 emissions during training using [CodeCarbon](https://github.com/mlco2/codecarbon).
  * **📦 Model Export:** Easily export your trained PyTorch models to **ONNX** with built-in quantization support.

-----

## 🚀 Getting Started

### 1\. Installation

First, clone the repository and create the Conda environment from the provided file:

```bash
git clone https://github.com/your-username/{{cookiecutter.project_name}}.git
cd {{cookiecutter.project_name}}

conda env create -f environment.yml
conda activate torch-env
```

### 2\. Configure Tracking (Optional)

This template uses **Comet** for experiment tracking. To enable it, create a `.env` file from the example and add your credentials:

```bash
cp .env.example .env
nano .env  # Add your COMET_API_KEY and COMET_WORKSPACE
```

### 3\. Run a Training Task

You can start a training task by pointing the `main.py` script to a configuration file.

```bash
python main.py --config_path ./tasks/mnist/train.yaml
```

-----

## 🛠️ Usage

### 📦 Exporting a Model to ONNX

The framework supports exporting a saved PyTorch model (`.pth`) to **ONNX**. This process is also config-driven.

1.  **Create an export config** (e.g., `export.yaml`). This file specifies the paths and model architecture.
2.  **Run the `export.py` script.**

<!-- end list -->

```bash
python export.py --export_config_path ./tasks/default/export.yaml
```

An example `export.yaml` file looks like this:

```yaml
# Path to save the exported ONNX model
export_path: './exports'

# Path to the saved PyTorch model checkpoint
model_path: './saved_models/best_model.pth'

# Model definition (must match the saved model)
model:
  module_name: src.models.super_resolution
  class_name: SRResUNet
  parameters:
    # ... (model-specific parameters)

# --- Optional: Quantization ---
# By default, the export script also saves a quantized ONNX model.
# This requires a 'quantization_dataset' for calibration.
quantization_dataset:
  module_name: src.data.sets.super_resolution
  class_name: FastMRISuperResolutionDataReader
  parameters:
    data_folder: /path/to/calibration-dataset
    num_samples: 100 # Number of samples to use for calibration
```

> **Note:** If you do not want to create a quantized model, simply remove the `quantization_dataset` key from your export config.

-----

## 📊 Tracking & Logging

### 📈 Experiment Tracking with Comet ML

This template is fully integrated with [Comet ML](https://www.comet.com). When you provide a valid API key in your `.env` file, the framework will automatically:

  * Log all hyperparameters from your `.yaml` config.
  * Track training, validation, and test metrics (e.g., loss, accuracy) in real-time.
  * Save your model checkpoints.
  * Upload generated artifacts (like the `emissions.csv` from CodeCarbon).

### 🌍 Energy Consumption Tracking

This template uses [CodeCarbon](https://github.com/mlco2/codecarbon) to track energy consumption and estimate carbon emissions. This is implemented via the `EmissionsTracker` in `./src/core/experiment.py`:

```python
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

**How it works:**

  * **Local CSV:** By default, metrics are saved locally to an `emissions.csv` file within your experiment's log directory. This file is also automatically uploaded to your Comet experiment's **Assets & Artifacts** tab.
  * **Comet Integration:** The tracker also saves summary metrics (like `total_energy_kwh` and `total_co2_emissions`) in the **Others** tab of your Comet experiment.
  * **Task-Specific Tracking:** It automatically tracks energy for two distinct tasks:
      * `data`: Energy consumed during data loading.
      * `training`: Energy consumed during the main training, validation, and testing loops.

> You can customize this behavior by creating a `.codecarbon.config` file in the project's root directory. See the [CodeCarbon documentation](https://www.google.com/search?q=https://mlco2.github.io/codecarbon/configuration.html) for details.