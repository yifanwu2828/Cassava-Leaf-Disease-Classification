# ECE228 Team-4 - Cassava Leaf Disease Classification
![GitHub repo size](https://img.shields.io/github/repo-size/yifanwu2828/ECE_228-Team-4)
![GitHub contributors](https://img.shields.io/github/contributors/yifanwu2828/ECE_228-Team-4)
![GitHub last commit](https://img.shields.io/github/last-commit/yifanwu2828/ECE_228-Team-4)

Identify the type of disease present on a Cassava Leaf image

## Table of contents
* [Introduction](#Introduction)
* [Features](#Features)
* [Prerequisites](#Prerequisites)
* [Setup](#setup)
* [Usage](#Usage)
* [Contribute](#Contribute)
* [Sources](#Sources)

## Introduction
This project is simple Lorem ipsum dolor generator.

## Features
* ...
* ...
### TODO:
- [x] Set up project virtual environment
- [x] Implement trainer
- [x] Implement resnet example
- [ ] Implement more NN Acrh example
- [x] Add visualization for learned reward during training
- [ ] Add Optuna to automate hyperparameter search
- [ ] Check tensorboard video logger for visualization
- [x] Add requirements.txt (need update)
- [ ] Clear TODOs in code 

## Prerequisites
Before you continue, ensure you have met the following requirements:

* You have installed the version of`<python >= 3.6>` 

* You are using a `<Linux/MacOS>` machine. Windows is not currently recommended.

## Setup
To run this project, there are two options:

(Recommended) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at [Link to Conda environment setup](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
This install will modify the `PATH` variable in your bashrc.
You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).
```
2. Create a conda environment that will contain python 3:
```
conda create -n `<env_name>` python=3.6
```
3. activate the environment (do this every time you open a new terminal and want to run code):
```
conda activate `<env_name>`
```

4. Install the requirements into this conda environment
```
pip install --user -r requirements.txt
```

## Usage
TODO...

## Contribute
Thanks to the following people who have contributed to this project:

TODO...

## Sources
[Dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification)

[Albumentations](https://github.com/albumentations-team/albumentations#i-am-new-to-image-augmentation)

[Imbalanced Dataset Sampler](https://github.com/ufoym/imbalanced-dataset-sampler)

[Optuna](https://optuna.org/)