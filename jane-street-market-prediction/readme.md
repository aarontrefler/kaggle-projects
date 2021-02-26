# Jane Street Market Prediction
* [Kaggle Competition Link](https://www.kaggle.com/c/jane-street-market-prediction/)  
* Submission Date: 02-22-2021
* Score: TBD

## Resources
---
* Based on [Kaggle Notebook - Market Prediction: XGBoost with GPU (Fit in 1min)](https://www.kaggle.com/hamditarek/market-prediction-xgboost-with-gpu-fit-in-1min)

## Overview
---
* **Goal**: Create a model that chooses whether to accept or reject stock trading opportunities. Evaluation is performed on future market returns after the competition submission deadline.
* **Preprocessing**: Investment outcome binary label was created. Training examples were filtered.
* **Modeling**: XGBoost classifier model was trained to determine whether to execute on a trade. Temporal train-validation split was used for limited manual hyper-parameter tuning.

## Project
---
```
├── input        <- Source data
├── submissions  <- Kaggle submission scripts
├── working      <- Jupyter notebooks
├── readme.md
```

## Setup
---
### Download Competition Input
```
pip install kaggle --upgrade
kaggle competitions download --wp -c jane-street-market-prediction 
```
### Unzip Competition Input
```
sudo yum install epel-release
sudo yum --enablerepo=epel install p7zip
7za x jane-stree-market-prediction.zip
```