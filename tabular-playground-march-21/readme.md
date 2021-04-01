# Tabular Playground Series - Mar 2021
* [Kaggle Competition Link](https://www.kaggle.com/c/tabular-playground-series-mar-2021/overview)  
* Submission Date: 03-31-2021
* Score: 534/1495 (Top 36%)

## Resources
---
* [AWS 10-Minute Tutorials - Create a machine learning model automatically with Amazon SageMaker Autopilot](https://aws.amazon.com/getting-started/hands-on/create-machine-learning-model-automatically-sagemaker-autopilot/)

## Overview
---
The aim of this project was to test the effectiveness of using AWS Autopilot. Work was done in a SageMaker Studio environment.
* **Goal**: Create a model that predicts the probability of a positive outcome.
* **Preprocessing**: AWS Autopilot Experiment (default settings)
* **Modeling**: AWS Autopilot Experiment (default settings)

## Project
---
```
├── input                                                    <- Source data
├── working
│   ├── 0-create-dataset.ipynb                               <- Format and save datasets to S3
│   ├── 1-create-submission.ipynb                            <- Perform inference and create submission
│   ├── SageMakerAutopilotCandidateDefinitionNotebook.ipynb  <- AWS Autopilot generated notebook
│   ├── SageMakerAutopilotDataExplorationNotebook.ipynb      <- AWS Autopilot generated notebook
│   ├── config.py
├── readme.md
```

## Setup
---
### Download Competition Input
```
pip install kaggle --upgrade
kaggle competitions download -c tabular-playground-series-mar-2021
```
### Unzip Competition Input
```
sudo yum install epel-release
sudo yum --enablerepo=epel install p7zip
7za x tabular-playground-series-mar-2021.zip
```
